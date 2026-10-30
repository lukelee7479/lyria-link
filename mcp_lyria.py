#!/usr/bin/env python3
"""
MCP Lyria 3 audio Generator - A Model Context Protocol server for Lyria 3 audio generation
Usage:
  python mcp_Lyria3.py --output-dir ~/audios/Generated
"""

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
import httpx
import base64
import re
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime
from urllib.parse import urlparse, unquote, parse_qs
from io import BytesIO
from PIL import Image as PILImage

from fastmcp import FastMCP, Context
from pydantic import BaseModel
from dotenv import load_dotenv

from mcp_s3 import upload_file, BUCKET, s3, args as s3_args
from botocore.exceptions import BotoCoreError, ClientError

try:
    from google import genai
    from google.genai import errors as genai_errors
except ImportError:
    genai = None

# Load environment variables from .env file
load_dotenv()

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True, help="Directory to save generated audios")
parser.add_argument("--api-key", help="Gemini API key (overrides .env)")
parser.add_argument("--model", required=False, help="in case of omni deprecation, manually input the new model name")
args, unknown = parser.parse_known_args()

OUTPUT_DIR = os.path.abspath(os.path.expanduser(args.output_dir))
INTERACTION_ID_MARKER = "__iid_"

# Get API key from CLI args or environment
API_KEY = args.api_key 
if not API_KEY:
  concat_envs = os.getenv("NANOBANANA_API_KEY")
  try:
      banana_envs = [value.strip() for value in concat_envs.split("++", 2)]
      API_KEY = banana_envs[0]
  except:
      API_KEY = concat_envs
  
if not API_KEY:
    raise ValueError("Gemini API key must be provided via --api-key argument or NANOBANANA_API_KEY in .env file")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-Lyria3")

# Initialize FastMCP
mcp = FastMCP("MCP Lyria 3 audio Generator")

# Initialize Gemini client
if not genai:
    raise ImportError("google-genai package not installed. Run: pip install google-genai")

gemini_client = genai.Client(api_key=API_KEY)


class audioGenerationResponse(BaseModel):
    audio_path: str
    filename: str
    model: str
    prompt: str
    generation_time: float
    file_size: int
    lyrics: Optional[str] = None
    url: str


class audioListResponse(BaseModel):
    audios: list[dict]
    total_count: int
    output_dir: str


class audioInfoResponse(BaseModel):
    filename: str
    path: str
    size: int
    created: str
    modified: str

class audioSubmissionRateLimitError(RuntimeError):
    """음악 생성 요청이 한도 초과 429로 거절된 경우"""

def safe_join(root: str, user_path: str) -> str:
    """Safely join paths and prevent directory traversal"""
    abs_path = os.path.abspath(os.path.join(root, user_path))
    if not abs_path.startswith(root):
        raise ValueError("Path escapes allowed root")
    return abs_path

HTTPX_CLIENT: Optional[httpx.AsyncClient] = None

def get_httpx_client():
    global HTTPX_CLIENT
    if HTTPX_CLIENT is None:
        HTTPX_CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    return HTTPX_CLIENT

def detect_image_mime(image_data: bytes) -> str:
    if not image_data:
        raise ValueError("Downloaded image is empty")

    with PILImage.open(BytesIO(image_data)) as img:
        mime_type = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
        }.get(img.format)

        if mime_type is None:
            raise ValueError(
                "Only JPEG and PNG input images are accepted"
            )

        img.verify()

    return mime_type
  
def encode_interaction_id(interaction_id: str) -> str:
    """Encode an opaque interaction ID into a filename-safe string."""
    encoded = base64.urlsafe_b64encode(
        interaction_id.encode("utf-8")
    ).decode("ascii")

    return encoded.rstrip("=")


def decode_interaction_id(encoded_id: str) -> Optional[str]:
    """Decode a filename-safe interaction ID."""
    try:
        padding = "=" * (-len(encoded_id) % 4)

        return base64.urlsafe_b64decode(
            encoded_id + padding
        ).decode("utf-8")

    except Exception:
        return None
      
def extract_interaction_id_from_audio_url(
    audio_url: str
) -> Optional[str]:
    """Extract an encoded interaction ID from an MCP-generated audio URL."""
    try:
        clean_url = audio_url.strip().replace("&amp;", "&")
        parsed = urlparse(clean_url)

        filename = unquote(
            Path(parsed.path).name
        )

        match = re.search(
            r"__iid_([A-Za-z0-9_-]+)\.mp3$",
            filename
        )

        if not match:
            return None

        encoded_id = match.group(1)

        return decode_interaction_id(encoded_id)

    except Exception:
        return None

def _download_edit_audio(
    audio_url: str,
    destination: Path,
    max_bytes: int = 30 * 1024 * 1024,
) -> None:
    """Download an mp3 from the configured S3 bucket or CloudFront host."""
    clean_url = audio_url.strip().replace("&amp;", "&")
    parsed = urlparse(clean_url)
    host = (parsed.hostname or "").lower()
    bucket = (BUCKET or "").lower()
    key = None

    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError(
            "User information and fragments are not supported"
        )

    if parsed.scheme not in ("https", "s3"):
        raise ValueError(
            "Use an HTTPS S3/CloudFront URL or s3://bucket/key"
        )

    if parsed.port not in (None, 443):
        raise ValueError("Only the standard HTTPS port is supported")

    s3_endpoint = r"s3(?:[.-][a-z0-9-]+)*\.amazonaws\.com(?:\.cn)?"

    if parsed.scheme == "s3":
        if not bucket or host != bucket:
            raise ValueError("The S3 bucket must match --bucket")

        key = unquote(parsed.path[1:])

    elif (
        bucket
        and host.startswith(bucket + ".")
        and re.fullmatch(s3_endpoint, host[len(bucket) + 1:])
    ):
        key = unquote(parsed.path[1:])

    elif re.fullmatch(s3_endpoint, host):
        prefix = f"/{bucket}/"

        if not bucket or not parsed.path.startswith(prefix):
            raise ValueError("The S3 bucket must match --bucket")

        key = unquote(parsed.path[len(prefix):])

    else:
        configured = (
            getattr(s3_args, "cloudfront", None) or ""
        ).strip()

        cf_host = urlparse(
            configured
            if "://" in configured
            else "https://" + configured
        ).hostname

        if not cf_host or host != cf_host.lower():
            raise ValueError(
                "URL must use the configured S3 bucket "
                "or --cloudfront host"
            )

    if key == "" or not parsed.path or parsed.path.endswith("/"):
        raise ValueError("The URL must identify a audio object")

    def save_chunks(chunks, content_length=None):
        if (
            content_length is not None
            and int(content_length) > max_bytes
        ):
            raise ValueError(
                "Input audio exceeds the configured download size limit"
            )

        total = 0

        with destination.open("wb") as output:
            for chunk in chunks:
                if not chunk:
                    continue

                total += len(chunk)

                if total > max_bytes:
                    raise ValueError(
                        "Input audio exceeds the configured "
                        "download size limit"
                    )

                output.write(chunk)

        if total == 0:
            raise ValueError("Downloaded audio is empty")

    if key is not None:
        try:
            request = {
                "Bucket": BUCKET,
                "Key": key
            }

            version = parse_qs(parsed.query).get("versionId")
            if version:
                request["VersionId"] = version[0]

            response = s3.get_object(**request)
            body = response["Body"]

            try:
                save_chunks(
                    body.iter_chunks(chunk_size=1024 * 1024),
                    response.get("ContentLength")
                )
            finally:
                body.close()

            return

        except (BotoCoreError, ClientError) as exc:
            if parsed.scheme == "s3":
                raise

            logger.warning(
                "S3 SDK download failed (%s); "
                "trying the supplied HTTPS URL",
                type(exc).__name__
            )

    # Do not reconstruct or discard the signed query string.
    try:
        with httpx.stream(
            "GET",
            clean_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
            follow_redirects=False,
        ) as response:
            response.raise_for_status()

            content_type = (
                response.headers.get("content-type", "")
                .split(";")[0]
                .lower()
            )

            if content_type not in (
                "",
                "audio/mp3",
                "application/mp3",
                "application/octet-stream",
            ):
                raise ValueError(
                    "The URL did not return an mp3 audio"
                )

            save_chunks(
                response.iter_bytes(chunk_size=1024 * 1024),
                response.headers.get("content-length")
            )

    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"audio download failed: HTTP {exc.response.status_code}"
        ) from None

    except httpx.RequestError as exc:
        raise RuntimeError(
            f"audio download failed: {type(exc).__name__}"
        ) from None

async def _wait_for_edit_audio(
    audio_file,
    ctx: Context,
    poll_interval: float = 2.0,
    max_wait: float = 60.0,
):
    """Wait for an uploaded Google file to become ACTIVE."""
    deadline = time.monotonic() + max_wait

    while True:
        state = getattr(audio_file, "state", None)
        state = getattr(state, "name", state)

        if state == "ACTIVE":
            if not getattr(audio_file, "uri", None):
                raise RuntimeError(
                    "Google file is ACTIVE but has no URI"
                )

            return audio_file

        if state == "FAILED":
            raise RuntimeError(
                "Google could not process the uploaded audio"
            )

        if state != "PROCESSING":
            raise RuntimeError(
                f"Unexpected Google file state: {state}"
            )

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            raise TimeoutError(
                "Timed out waiting for Google audio processing"
            )

        await ctx.info(
            "Waiting for Google to process the uploaded audio..."
        )

        await asyncio.sleep(min(poll_interval, remaining))

        audio_file = await asyncio.to_thread(
            gemini_client.files.get,
            name=audio_file.name
        )

async def generate_music_with_progress(
    prompt: str,
    model: str,
    ctx: Context,
    image_data: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
    previous_interaction_id: Optional[str] = None
) -> dict:
    """Generate a audio using Google Lyria"""

    start_time = time.time()
  
    if args.model:
        model = args.model

    try:
        await ctx.info(f"Starting audio generation with model: {model}")
        await ctx.info(f"Prompt: {prompt[:100]}...")

        await ctx.report_progress(progress=5, total=100)

        try:
            if image_data is not None:
                base64_image = base64.b64encode(image_data).decode("utf-8")

                interaction_input = [
                    {
                        "type": "image",
                        "data": base64_image,
                        "mime_type": image_mime_type
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            else:
                interaction_input = prompt

            response_format = {
                "type": "audio"
            }

            interaction = await asyncio.to_thread(
                gemini_client.interactions.create,
                model=model,
                input=interaction_input,
                response_format=response_format,
                store=True
            )
                

        except genai_errors.APIError as e:
            if e.code == 429:
                raise audioSubmissionRateLimitError(
                    f"{model}: generation request rejected with HTTP 429"
                ) from e
            raise

        await ctx.report_progress(progress=90, total=100)

        if (
            not hasattr(interaction, "output_audio")
            or interaction.output_audio is None
            or not interaction.output_audio.data
        ):
            raise RuntimeError(
                "audio generation failed - no audio data in response"
            )

        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        interaction_id=getattr(interaction, "id", None)
        if interaction_id:
            encoded_interaction_id = encode_interaction_id(interaction_id)
            filename = f"lyria_music_{timestamp}{INTERACTION_ID_MARKER}{encoded_interaction_id}.mp3"
        else:
            filename = f"lyria_music_{timestamp}_{uuid.uuid4().hex}.mp3"
          
        output_path = output_dir / filename

        await ctx.info(f"Saving audio to: {output_path}")

        audio_bytes = base64.b64decode(
            interaction.output_audio.data
        )

        def _save_audio():
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

        await asyncio.to_thread(_save_audio)

        await ctx.report_progress(progress=100, total=100)

        file_size = output_path.stat().st_size if output_path.exists() else 0
        generation_time = time.time() - start_time

        await ctx.info(
            f"audio generation completed in {generation_time:.1f} seconds"
        )

        lyrics = interaction.output_text

        return {
            "audio_path": str(output_path),
            "filename": filename,
            "model": model,
            "prompt": prompt,
            "generation_time": generation_time,
            "file_size": file_size,
            "lyrics": lyrics
        }

    except Exception as e:
        logger.exception("audio generation failed")

        try:
            await ctx.report_progress(progress=0, total=100)
            await ctx.error(f"audio generation failed: {str(e)}")
        except Exception:
            logger.warning(
                "Failed to report audio generation error",
                exc_info=True
            )

        raise

@mcp.tool()
async def generate_music(
    prompt: str,
    ctx: Context,
    model: str = "lyria-3.5"
) -> audioGenerationResponse:
    """Generate a audio using lyria from a text prompt
    
    Args:
        prompt: Text prompt describing the audio to generate
        model: audio model to use 
    
    Returns:
        audioGenerationResponse with audio path, metadata, and generation info
    """
    
    await ctx.info(f"Starting audio generation with prompt: {prompt[:100]}...")
    
    if not prompt.strip():
        await ctx.error("Prompt cannot be empty")
        raise ValueError("Prompt cannot be empty")

    result = None

    try:
        result = await generate_music_with_progress(
            prompt=prompt,
            model=model,
            ctx=ctx
        )
        
        logger.info(f"audio generated successfully: {result['filename']}")
      
    except audioSubmissionRateLimitError as e:
        logger.error(f"Model quota exceeded. {e}")
        await ctx.error(f"Generation failed due to quota limit: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"audio generation failed: {e}")
        await ctx.error(f"Generation failed: {str(e)}")
        raise
            
    try:
        logger.info("Trying S3 upload")
          
        s3_response = await upload_file(local_path=result['filename'], ctx=ctx)
        result["url"] = s3_response.url
        logger.info("s3 upload done")

        return audioGenerationResponse(**result)
        
    except Exception as e:
        await ctx.error(f"audio generated. but upload failed: {str(e)}")
        raise ValueError(f"audio generated. but upload failed: {str(e)}")
            

@mcp.tool(
    description=(
        "Generate a music from image"
    )
)
async def generate_music_from_image(
    prompt: str,
    image_path: str,
    ctx: Context,
    model: str = "lyria-3.5"
) -> audioGenerationResponse:
    """Generate a audio using Google lyria from an image and text prompt
    
    Args:
        prompt: Text prompt describing the audio motion/action
        image_path: Path to the starting image file
        model: Lyria model to use 
    
    Returns:
        audioGenerationResponse with audio path, metadata, and generation info
    """
    
    await ctx.info(f"Starting image-to-audio generation: {image_path}")
    
    if not prompt.strip():
        await ctx.error("Prompt cannot be empty")
        raise ValueError("Prompt cannot be empty")
    
    if not image_path.strip():
        await ctx.error("Image path cannot be empty")
        raise ValueError("Image path cannot be empty")
    
    # Image download
    try:
      
        image_data = None
        content_type = "image/jpg"

        clean_url = image_path.strip().replace("&amp;", "&")

        if BUCKET and BUCKET in clean_url and "amazonaws.com" in clean_url:
            logger.info("s3 url detected. trying boto3 download")
            try:
                parsed_url = urlparse(clean_url)
                s3_key = unquote(parsed_url.path.lstrip('/'))
                def fetch_from_s3():
                    resp = s3.get_object(Bucket=BUCKET, Key=s3_key)
                    return resp['Body'].read(), resp.get('ContentType', 'image/jpeg')
                        
                image_data, content_type = await asyncio.to_thread(fetch_from_s3)
                logger.info("image downloaded from s3")
            except Exception as e:
                logger.warning(f"download from s3 failed: {e}")
        if not image_data:
            max_retries = 3

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "image/*,*/*;q=0.8"
            }

            http_client = get_httpx_client()
            for attempt in range(max_retries):
                try:
                    # await를 사용하여 비동기적으로 GET 요청 전송
                    response = await http_client.get(
                        clean_url,
                        headers=headers,
                        timeout=30.0,
                        follow_redirects=True
                    )
                    response.raise_for_status()

                    res_content_type = response.headers.get('content-type', '').lower()
                    if not any(img_type in res_content_type for img_type in ['image/', 'application/octet-stream']):
                        raise ValueError(f"URL does not point to an image. Content-Type: {res_content_type}")

                    image_data = response.content
                    content_type = res_content_type
                    break

                except httpx.HTTPStatusError as e:
                    if attempt == max_retries -1:
                        raise ValueError(f"HTTP download status error: {e.response.status_code}")
                    await asyncio.sleep(2 ** attempt)

                except httpx.TimeoutException:
                    if attempt == max_retries - 1:
                        raise ValueError("Image download timed out after multiple attempts")
                    logger.warning(f"Download attempt {attempt + 1} timed out, retrying...")
                    await asyncio.sleep(2 ** attempt)

                except httpx.RequestError as e:
                    if attempt == max_retries - 1:
                        raise ValueError(f"Connection error during image download: {str(e)}")
                    logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(2 ** attempt)

        if not image_data:
            raise ValueError("No image data downloaded")

        safe_mime_type = await asyncio.to_thread(detect_image_mime, image_data)

    except Exception as e:
        logger.exception(f"Unexpected error during image download/open: {e}")
        raise RuntimeError(f"Failed to download or open image: {str(e)}")

    result = None

    try:
        result = await generate_music_with_progress(
            prompt=prompt,
            model=model,
            ctx=ctx,
            image_data=image_data,
            image_mime_type=safe_mime_type,
        )
        
        logger.info(f"Image-to-audio generation successful: {result['filename']}")

    except audioSubmissionRateLimitError as e:
        logger.error(f"Model quota exceeded: {e}")
        await ctx.error(f"Generation failed due to quota limit:{str(e)}")
        raise

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        await ctx.error(f"Generation failed: {str(e)}")
        raise

    try:    
        logger.info("Trying S3 upload")
          
        
        s3_response = await upload_file(local_path=result['filename'], ctx=ctx)
        result["url"] = s3_response.url
        logger.info("s3 upload done")

        return audioGenerationResponse(**result)
        
    except Exception as e:
        await ctx.error(f"Image-to-audio generated. but upload failed: {str(e)}")
        raise ValueError(f"Image-to-audio generated. but upload failed: {str(e)}")


def main():
    """Main entry point for the MCP Lyria 3 server"""
    mcp.run()


if __name__ == "__main__":
    main()
