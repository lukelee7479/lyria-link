#!/usr/bin/env python3
"""
MCP Veo 3 Video Generator - A Model Context Protocol server for Veo 3 video generation
Usage:
  python mcp_veo3.py --output-dir ~/Videos/Generated
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
parser.add_argument("--output-dir", required=True, help="Directory to save generated videos")
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
logger = logging.getLogger("mcp-veo3")

# Initialize FastMCP
mcp = FastMCP("MCP Veo 3 Video Generator")

# Initialize Gemini client
if not genai:
    raise ImportError("google-genai package not installed. Run: pip install google-genai")

gemini_client = genai.Client(api_key=API_KEY)


class VideoGenerationResponse(BaseModel):
    video_path: str
    filename: str
    model: str
    prompt: str
    negative_prompt: Optional[str] = None
    generation_time: float
    file_size: int
    aspect_ratio: str
    url: str


class VideoListResponse(BaseModel):
    videos: list[dict]
    total_count: int
    output_dir: str


class VideoInfoResponse(BaseModel):
    filename: str
    path: str
    size: int
    created: str
    modified: str

class VideoSubmissionRateLimitError(RuntimeError):
    """영상 생성 요청이 한도 초과 429로 거절된 경우"""

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
      
def extract_interaction_id_from_video_url(
    video_url: str
) -> Optional[str]:
    """Extract an encoded interaction ID from an MCP-generated video URL."""
    try:
        clean_url = video_url.strip().replace("&amp;", "&")
        parsed = urlparse(clean_url)

        filename = unquote(
            Path(parsed.path).name
        )

        match = re.search(
            r"__iid_([A-Za-z0-9_-]+)\.mp4$",
            filename
        )

        if not match:
            return None

        encoded_id = match.group(1)

        return decode_interaction_id(encoded_id)

    except Exception:
        return None

def _download_edit_video(
    video_url: str,
    destination: Path,
    max_bytes: int = 30 * 1024 * 1024,
) -> None:
    """Download an MP4 from the configured S3 bucket or CloudFront host."""
    clean_url = video_url.strip().replace("&amp;", "&")
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
        raise ValueError("The URL must identify a video object")

    def save_chunks(chunks, content_length=None):
        if (
            content_length is not None
            and int(content_length) > max_bytes
        ):
            raise ValueError(
                "Input video exceeds the configured download size limit"
            )

        total = 0

        with destination.open("wb") as output:
            for chunk in chunks:
                if not chunk:
                    continue

                total += len(chunk)

                if total > max_bytes:
                    raise ValueError(
                        "Input video exceeds the configured "
                        "download size limit"
                    )

                output.write(chunk)

        if total == 0:
            raise ValueError("Downloaded video is empty")

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
                "video/mp4",
                "application/mp4",
                "application/octet-stream",
            ):
                raise ValueError(
                    "The URL did not return an MP4 video"
                )

            save_chunks(
                response.iter_bytes(chunk_size=1024 * 1024),
                response.headers.get("content-length")
            )

    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Video download failed: HTTP {exc.response.status_code}"
        ) from None

    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Video download failed: {type(exc).__name__}"
        ) from None

async def _wait_for_edit_video(
    video_file,
    ctx: Context,
    poll_interval: float = 2.0,
    max_wait: float = 60.0,
):
    """Wait for an uploaded Google file to become ACTIVE."""
    deadline = time.monotonic() + max_wait

    while True:
        state = getattr(video_file, "state", None)
        state = getattr(state, "name", state)

        if state == "ACTIVE":
            if not getattr(video_file, "uri", None):
                raise RuntimeError(
                    "Google file is ACTIVE but has no URI"
                )

            return video_file

        if state == "FAILED":
            raise RuntimeError(
                "Google could not process the uploaded video"
            )

        if state != "PROCESSING":
            raise RuntimeError(
                f"Unexpected Google file state: {state}"
            )

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            raise TimeoutError(
                "Timed out waiting for Google video processing"
            )

        await ctx.info(
            "Waiting for Google to process the uploaded video..."
        )

        await asyncio.sleep(min(poll_interval, remaining))

        video_file = await asyncio.to_thread(
            gemini_client.files.get,
            name=video_file.name
        )

async def generate_video_with_progress(
    prompt: str,
    model: str,
    ctx: Context,
    image_data: Optional[bytes] = None,
    image_mime_type: Optional[str] = None,
    task: Optional[str] = None,
    video_uri: Optional[str] = None,
    previous_interaction_id: Optional[str] = None
) -> dict:
    """Generate a video using Gemini Omni Flash"""

    start_time = time.time()
  
    if args.model:
        model = args.model

    try:
        await ctx.info(f"Starting video generation with model: {model}")
        await ctx.info(f"Prompt: {prompt[:100]}...")

        await ctx.report_progress(progress=5, total=100)

        try:
            if task in ("edit", "extend"):
                if previous_interaction_id:
                    interaction_input = None

                elif not video_uri:
                    raise ValueError(
                        f"task='{task}' requires a processed Google video URI"
                    )
                else:
                    if not video_uri.strip():
                        raise ValueError("Google video URI cannot be empty")
                    else:
                        interaction_input = [
                            {
                                "type": "video",
                                "uri": video_uri
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]

            else:
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

            extra_kwargs = {}

            if task is not None:
                extra_kwargs["generation_config"] = {
                    "video_config": {
                        "task": task
                    }
                }

            response_format = {
                "type": "video",
                "resolution": "720p"
            }

            if task in ("text_to_video", "image_to_video", "reference_to_video"):
                response_format["aspect_ratio"] = "16:9"

            if task =="edit" and previous_interaction_id:
                interaction = await asyncio.to_thread(
                    gemini_client.interactions.create,
                    model=model,
                    previous_interaction_id=previous_interaction_id,
                    input=prompt,
                    response_format=response_format,
                    store=True,
                )
            elif task =="extend" and previous_interaction_id:
                interaction = await asyncio.to_thread(
                    gemini_client.interactions.create,
                    model=model,
                    previous_interaction_id=previous_interaction_id,
                    input=prompt,
                    response_format=response_format,
                    store=True,
                    **extra_kwargs
                )
            else:
                interaction = await asyncio.to_thread(
                    gemini_client.interactions.create,
                    model=model,
                    input=interaction_input,
                    response_format=response_format,
                    store=True,
                    **extra_kwargs
                )
                

        except genai_errors.APIError as e:
            if e.code == 429:
                raise VideoSubmissionRateLimitError(
                    f"{model}: generation request rejected with HTTP 429"
                ) from e
            raise

        await ctx.report_progress(progress=90, total=100)

        if (
            not hasattr(interaction, "output_video")
            or interaction.output_video is None
            or not interaction.output_video.data
        ):
            raise RuntimeError(
                "Video generation failed - no video data in response"
            )

        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        interaction_id=getattr(interaction, "id", None)
        if interaction_id:
            encoded_interaction_id = encode_interaction_id(interaction_id)
            filename = f"omni_video_{timestamp}{INTERACTION_ID_MARKER}{encoded_interaction_id}.mp4"
        else:
            filename = f"omni_video_{timestamp}_{uuid.uuid4().hex}.mp4"
          
        output_path = output_dir / filename

        await ctx.info(f"Saving video to: {output_path}")

        video_bytes = base64.b64decode(
            interaction.output_video.data
        )

        def _save_video():
            with open(output_path, "wb") as f:
                f.write(video_bytes)

        await asyncio.to_thread(_save_video)

        await ctx.report_progress(progress=100, total=100)

        file_size = output_path.stat().st_size if output_path.exists() else 0
        generation_time = time.time() - start_time

        await ctx.info(
            f"Video generation completed in {generation_time:.1f} seconds"
        )

        return {
            "video_path": str(output_path),
            "filename": filename,
            "model": model,
            "prompt": prompt,
            "negative_prompt": None,
            "generation_time": generation_time,
            "file_size": file_size,
            "aspect_ratio": "16:9",
        }

    except Exception as e:
        logger.exception("Video generation failed")

        try:
            await ctx.report_progress(progress=0, total=100)
            await ctx.error(f"Video generation failed: {str(e)}")
        except Exception:
            logger.warning(
                "Failed to report video generation error",
                exc_info=True
            )

        raise

@mcp.tool()
async def generate_video(
    prompt: str,
    ctx: Context,
    model: str = "gemini-omni-1.1-flash"
) -> VideoGenerationResponse:
    """Generate a video using omni flash from a text prompt
    
    Args:
        prompt: Text prompt describing the video to generate
        model: video model to use 
    
    Returns:
        VideoGenerationResponse with video path, metadata, and generation info
    """
    
    await ctx.info(f"Starting video generation with prompt: {prompt[:100]}...")
    
    if not prompt.strip():
        await ctx.error("Prompt cannot be empty")
        raise ValueError("Prompt cannot be empty")
    """
    # Validate model
    valid_models = ["veo-3.1-lite-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.1-generate-preview" ]
  
    if model and model in valid_models:
        models_to_try = [model] + [candidate for candidate in valid_models if candidate != model]
    else:
        models_to_try = valid_models
    """

    task = "text_to_video"
    result = None

    try:
        result = await generate_video_with_progress(
            prompt=prompt,
            model=model,
            ctx=ctx,
            task = task
        )
        
        logger.info(f"Video generated successfully: {result['filename']}")
      
    except VideoSubmissionRateLimitError as e:
        logger.error(f"Model quota exceeded. {e}")
        await ctx.error(f"Generation failed due to quota limit: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        await ctx.error(f"Generation failed: {str(e)}")
        raise
            
    try:
        logger.info("Trying S3 upload")
          
        s3_response = await upload_file(local_path=result['filename'], ctx=ctx)
        result["url"] = s3_response.url
        logger.info("s3 upload done")

        return VideoGenerationResponse(**result)
        
    except Exception as e:
        await ctx.error(f"Video generated. but upload failed: {str(e)}")
        raise ValueError(f"Video generated. but upload failed: {str(e)}")
            

@mcp.tool(
    description=(
        "Generate a video using a single input image as the starting scene, "
        "then continue that scene with motion. "
        "Choose this tool when the user requests 'animate this photo', "
        "'start the video from this scene', or 'use this image as the first frame'. "
        "If the user only wants to reference the subject's appearance or visual style "
        "while creating a different scene, use generate_video_from_reference instead. "
        "In prompt, explicitly state that the input image should be used as the first frame, "
        "and describe the subsequent subject motion and camera movement. "
        "Provide exactly one HTTP/HTTPS URL in image_path, pointing to a JPEG or PNG image "
        "that the server can download. Local file paths and multiple images are not supported. "
        "Returns the generated video URL and metadata."
    )
)
async def generate_video_from_image(
    prompt: str,
    image_path: str,
    ctx: Context,
    model: str = "gemini-omni-1.1-flash"
) -> VideoGenerationResponse:
    """Generate a video using Google omni flash from an image and text prompt
    
    Args:
        prompt: Text prompt describing the video motion/action
        image_path: Path to the starting image file
        model: Veo model to use 
    
    Returns:
        VideoGenerationResponse with video path, metadata, and generation info
    """
    
    await ctx.info(f"Starting image-to-video generation: {image_path}")
    
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

    task = "image_to_video"
    result = None

    try:
        result = await generate_video_with_progress(
            prompt=prompt,
            model=model,
            ctx=ctx,
            image_data=image_data,
            image_mime_type=safe_mime_type,
            task=task
        )
        
        logger.info(f"Image-to-video generation successful: {result['filename']}")

    except VideoSubmissionRateLimitError as e:
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

        return VideoGenerationResponse(**result)
        
    except Exception as e:
        await ctx.error(f"Image-to-video generated. but upload failed: {str(e)}")
        raise ValueError(f"Image-to-video generated. but upload failed: {str(e)}")


@mcp.tool(
    description=(
        "Generate a new video using a single input image as a visual reference, "
        "rather than as the starting frame. "
        "Use the image to guide user-specified features such as a person's appearance, "
        "a character's design, a product's appearance, an artistic style, or a color palette. "
        "Choose this tool when the user requests 'show this character in a different location', "
        "'create a new scene featuring this product', or 'make a video in this image's style'. "
        "If the user wants the video to start from the scene shown in the image "
        "or wants to animate the image itself, use generate_video_from_image instead. "
        "In prompt, specify which features to reference and describe the new scene to create. "
        "Explicitly state that the input image is a visual reference, not the starting frame. "
        "Provide exactly one HTTP/HTTPS URL in image_path, pointing to a JPEG or PNG image "
        "that the server can download. Local file paths and multiple images are not supported. "
        "Returns the generated video URL and metadata."
    )
)
async def generate_video_from_reference(
    prompt: str,
    image_path: str,
    ctx: Context,
    model: str = "gemini-omni-1.1-flash"
) -> VideoGenerationResponse:
    """Generate a video using Google omni flash from an reference and text prompt
    
    Args:
        prompt: Text prompt describing the video motion/action
        image_path: Path to the reference image file
        model: Veo model to use 
    
    Returns:
        VideoGenerationResponse with video path, metadata, and generation info
    """
    
    await ctx.info(f"Starting image-to-video generation: {image_path}")
    
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

    task = "reference_to_video"
    result = None

    try:
        result = await generate_video_with_progress(
            prompt=prompt,
            model=model,
            ctx=ctx,
            image_data=image_data,
            image_mime_type=safe_mime_type,
            task=task
        )
        
        logger.info(f"reference-to-video generation successful: {result['filename']}")

    except VideoSubmissionRateLimitError as e:
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

        return VideoGenerationResponse(**result)
        
    except Exception as e:
        await ctx.error(f"reference-to-video generated. but upload failed: {str(e)}")
        raise ValueError(f"reference-to-video generated. but upload failed: {str(e)}")

@mcp.tool(
    description=(
        "Edit an existing MP4 video using a text instruction "
        "and return a new video URL. "
        "Use this tool to change objects, backgrounds, lighting, "
        "or visual effects in an existing clip, not to animate "
        "a still image or generate a video from image references. "
        "Provide one video URL from the configured S3 bucket "
        "or CloudFront host. "
        "The source video must be at most 10 seconds long. "
        "In prompt, describe the requested edit and what "
        "should remain unchanged. "
        "The source video is uploaded to Google for processing. "
        "The original S3 object is not overwritten. "
        "Returns the edited video URL and metadata."
    )
)
async def edit_video(
    prompt: str,
    video_url: str,
    ctx: Context,
    model: str = "gemini-omni-1.1-flash",
) -> VideoGenerationResponse:
    """Edit one uploaded video; --model overrides the tool's model."""

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if not video_url.strip():
        raise ValueError("Video URL cannot be empty")
      
    previous_interaction_id = (
        extract_interaction_id_from_video_url(video_url)
    )

    task = "edit"
  
    #Fast path with stored interaction ID
    if previous_interaction_id:
        await ctx.info(
            "Interaction ID found in video URL. "
            "Trying stateful video editing without downloading."
        )
      
        try:
            result = await generate_video_with_progress(
                prompt=prompt,
                model=model,
                ctx=ctx,
                task=task,
                previous_interaction_id=previous_interaction_id
            )

        except Exception as e:
            error_text = str(e).lower()
        
            error_code = getattr(e, "code", None)
        
            interaction_missing = (
                error_code in (400, 403, 404)
                or "not_found" in error_text
                or "requested entity was not found" in error_text
                or "do not have permission to access the file" in error_text
                or "it may not exist" in error_text
            )
        
            if not interaction_missing:
                raise
        
            logger.warning(
                "Stored interaction unavailable. "
                "Falling back to S3 video upload. "
                "error_type=%s, error_code=%s",
                type(e).__name__,
                error_code,
            )
        
            await ctx.info(
                "Stored interaction is unavailable. "
                "Falling back to uploaded-video editing."
            )

        else:
            s3_response = await upload_file(
            local_path=result["filename"],
            ctx=ctx
            )

            result["url"] = s3_response.url

            return VideoGenerationResponse(**result)


    #Fallback to download
  
    input_dir = Path(OUTPUT_DIR) / ".edit-inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    fd, path = tempfile.mkstemp(
        prefix="source_",
        suffix=".mp4",
        dir=input_dir
    )
    os.close(fd)

    source_path = Path(path)

    google_file_name = None
    edit_started = False
    edit_finished = False
    cancelled = False

    stage = "source download"

    try:
        await ctx.info("Downloading the source video...")

        await asyncio.to_thread(
            _download_edit_video,
            video_url,
            source_path
        )

        stage = "Google file upload"

        await ctx.info(
            "Uploading the source video to Google Files API..."
        )

        video_file = await asyncio.to_thread(
            gemini_client.files.upload,
            file=str(source_path),
            config={"mime_type": "video/mp4"},
        )

        google_file_name = getattr(video_file, "name", None)

        if not google_file_name:
            raise RuntimeError(
                "Google file upload returned no file name"
            )

        stage = "Google file processing"

        video_file = await _wait_for_edit_video(
            video_file,
            ctx
        )

        stage = "video editing"

        edit_started = True

        result = await generate_video_with_progress(
            prompt=prompt,
            model=model,
            ctx=ctx,
            video_uri=video_file.uri,
            task=task,
        )

        edit_finished = True
        stage = "edited video S3 upload"

        await ctx.info("Uploading the edited video to S3...")

        s3_response = await upload_file(
            local_path=result["filename"],
            ctx=ctx
        )

        result["url"] = s3_response.url

        return VideoGenerationResponse(**result)

    except asyncio.CancelledError:
        cancelled = True
        raise

    except Exception as exc:
        logger.exception(
            "Video edit failed during %s",
            stage
        )

        try:
            await ctx.error(
                f"Video edit failed during {stage}: {exc}"
            )
        except Exception:
            logger.warning(
                "Could not report the video edit error"
            )

        raise

    finally:
        # A cancelled to_thread call may still be using this file.
        if not cancelled:
            try:
                source_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not remove the temporary source video"
                )
        else:
            logger.warning(
                "Call cancelled; temporary input retained at %s",
                source_path
            )

        # Do not delete input needed by an uncertain/in-flight edit.
        if (
            google_file_name
            and not cancelled
            and not edit_finished
        ):
            try:
                await asyncio.to_thread(
                    gemini_client.files.delete,
                    name=google_file_name
                )
            except Exception:
                logger.warning(
                    "Could not delete the temporary Google file: %s",
                    google_file_name
                )

@mcp.tool(
    description=(
        "Extend an existing MP4 video by generating a natural continuation "
        "from the end of the source clip. "
        "Use this tool when the user wants the current video to continue beyond "
        "its ending, such as continuing a person's movement, extending camera motion, "
        "or showing what happens next in the same scene. "
        "Do not use this tool to modify objects, colors, backgrounds, or other content "
        "inside the existing portion of the video; use edit_video for those requests. "
        "Provide one video URL from the configured S3 bucket or CloudFront host. "
        "In prompt, describe what should happen after the current ending and specify "
        "any subject motion, camera movement, scene continuity, or elements that "
        "should remain consistent. "
        "The original video is not overwritten. "
        "Returns the extended video URL and metadata."
    )
)
async def extend_video(
    prompt: str,
    video_url: str,
    ctx: Context,
    model: str = "gemini-omni-1.1-flash",
) -> VideoGenerationResponse:
    """Extend one uploaded video; --model overrides the tool's model."""

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    if not video_url.strip():
        raise ValueError("Video URL cannot be empty")
      
    previous_interaction_id = (
        extract_interaction_id_from_video_url(video_url)
    )

    task = "extend"
  
    #Fast path with stored interaction ID
    if previous_interaction_id:
        await ctx.info(
            "Interaction ID found in video URL. "
            "Trying stateful video extension without downloading."
        )
      
        try:
            result = await generate_video_with_progress(
                prompt=prompt,
                model=model,
                ctx=ctx,
                task=task,
                previous_interaction_id=previous_interaction_id
            )

        except Exception as e:
            error_text = str(e).lower()
        
            error_code = getattr(e, "code", None)
        
            interaction_missing = (
                error_code in (400, 403, 404)
                or "not_found" in error_text
                or "requested entity was not found" in error_text
                or "do not have permission to access the file" in error_text
                or "it may not exist" in error_text
            )
        
            if not interaction_missing:
                raise
        
            logger.warning(
                "Stored interaction unavailable. "
                "Falling back to S3 video upload. "
                "error_type=%s, error_code=%s",
                type(e).__name__,
                error_code,
            )
        
            await ctx.info(
                "Stored interaction is unavailable. "
                "Falling back to uploaded-video extension."
            )
        else:
            s3_response = await upload_file(
            local_path=result["filename"],
            ctx=ctx
            )

            result["url"] = s3_response.url

            return VideoGenerationResponse(**result)

    #Fallback to download
  
    input_dir = Path(OUTPUT_DIR) / ".edit-inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    fd, path = tempfile.mkstemp(
        prefix="source_",
        suffix=".mp4",
        dir=input_dir
    )
    os.close(fd)

    source_path = Path(path)

    google_file_name = None
    edit_started = False
    edit_finished = False
    cancelled = False

    stage = "source download"

    try:
        await ctx.info("Downloading the source video...")

        await asyncio.to_thread(
            _download_edit_video,
            video_url,
            source_path
        )

        stage = "Google file upload"

        await ctx.info(
            "Uploading the source video to Google Files API..."
        )

        video_file = await asyncio.to_thread(
            gemini_client.files.upload,
            file=str(source_path),
            config={"mime_type": "video/mp4"},
        )

        google_file_name = getattr(video_file, "name", None)

        if not google_file_name:
            raise RuntimeError(
                "Google file upload returned no file name"
            )

        stage = "Google file processing"

        video_file = await _wait_for_edit_video(
            video_file,
            ctx
        )

        stage = "video extension"

        edit_started = True

        result = await generate_video_with_progress(
            prompt=prompt,
            model=model,
            ctx=ctx,
            video_uri=video_file.uri,
            task=task,
        )

        edit_finished = True
        stage = "extended video S3 upload"

        await ctx.info("Uploading the edited video to S3...")

        s3_response = await upload_file(
            local_path=result["filename"],
            ctx=ctx
        )

        result["url"] = s3_response.url

        return VideoGenerationResponse(**result)

    except asyncio.CancelledError:
        cancelled = True
        raise

    except Exception as exc:
        logger.exception(
            "Video extension failed during %s",
            stage
        )

        try:
            await ctx.error(
                f"Video extension failed during {stage}: {exc}"
            )
        except Exception:
            logger.warning(
                "Could not report the video extension error"
            )

        raise

    finally:
        # A cancelled to_thread call may still be using this file.
        if not cancelled:
            try:
                source_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not remove the temporary source video"
                )
        else:
            logger.warning(
                "Call cancelled; temporary input retained at %s",
                source_path
            )

        # Do not delete input needed by an uncertain/in-flight edit.
        if (
            google_file_name
            and not cancelled
            and not edit_finished
        ):
            try:
                await asyncio.to_thread(
                    gemini_client.files.delete,
                    name=google_file_name
                )
            except Exception:
                logger.warning(
                    "Could not delete the temporary Google file: %s",
                    google_file_name
                )


@mcp.tool()
async def list_generated_videos(ctx: Context) -> VideoListResponse:
    """List all generated videos in the output directory
    
    Returns:
        VideoListResponse with list of videos, count, and directory info
    """
    
    await ctx.info(f"Listing videos in: {OUTPUT_DIR}")
    
    output_dir = Path(OUTPUT_DIR)
    
    if not output_dir.exists():
        await ctx.info(f"Output directory {OUTPUT_DIR} does not exist yet")
        return VideoListResponse(
            videos=[],
            total_count=0,
            output_dir=str(output_dir)
        )
    
    # Find all video files
    video_extensions = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(output_dir.glob(ext))
    
    if not video_files:
        await ctx.info("No video files found")
        return VideoListResponse(
            videos=[],
            total_count=0,
            output_dir=str(output_dir)
        )
    
    # Sort by modification time (newest first)
    video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    videos = []
    for video_file in video_files:
        stat = video_file.stat()
        videos.append({
            "filename": video_file.name,
            "path": str(video_file.absolute()),
            "size": stat.st_size,
            "size_mb": round(stat.st_size / 1024 / 1024, 1),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    
    await ctx.info(f"Found {len(videos)} video files")
    
    return VideoListResponse(
        videos=videos,
        total_count=len(videos),
        output_dir=str(output_dir)
    )


@mcp.tool()
async def get_video_info(video_path: str, ctx: Context) -> VideoInfoResponse:
    """Get detailed information about a video file
    
    Args:
        video_path: Path to the video file (can be relative to output directory)
    
    Returns:
        VideoInfoResponse with file metadata
    """
    
    await ctx.info(f"Getting info for video: {video_path}")
    
    if not video_path.strip():
        await ctx.error("Video path cannot be empty")
        raise ValueError("Video path cannot be empty")
    
    # Resolve video path (allow relative paths within output directory for security)
    if not os.path.isabs(video_path):
        full_video_path = safe_join(OUTPUT_DIR, video_path)
    else:
        full_video_path = video_path
    
    video_file = Path(full_video_path)
    
    if not video_file.exists():
        await ctx.error(f"Video file not found: {full_video_path}")
        raise ValueError(f"Video file not found: {full_video_path}")
    
    stat = video_file.stat()
    created_time = datetime.fromtimestamp(stat.st_ctime).isoformat()
    modified_time = datetime.fromtimestamp(stat.st_mtime).isoformat()
    
    await ctx.info(f"Video info retrieved: {video_file.name} ({stat.st_size:,} bytes)")
    
    return VideoInfoResponse(
        filename=video_file.name,
        path=str(video_file.absolute()),
        size=stat.st_size,
        created=created_time,
        modified=modified_time
    )


def main():
    """Main entry point for the MCP Veo 3 server"""
    mcp.run()


if __name__ == "__main__":
    main()
