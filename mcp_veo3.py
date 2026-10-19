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
from pathlib import Path
from typing import Optional
from datetime import datetime
from urllib.parse import urlparse, unquote
from io import BytesIO
from PIL import Image as PILImage

from fastmcp import FastMCP, Context
from pydantic import BaseModel
from dotenv import load_dotenv

from mcp_s3 import upload_file, BUCKET, s3

try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
except ImportError:
    genai = None
    genai_types = None

# Load environment variables from .env file
load_dotenv()

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True, help="Directory to save generated videos")
parser.add_argument("--api-key", help="Gemini API key (overrides .env)")
args, unknown = parser.parse_known_args()

OUTPUT_DIR = os.path.abspath(os.path.expanduser(args.output_dir))

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

class VeoSubmissionRateLimitError(RuntimeError):
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


async def generate_video_with_progress(
    prompt: str,
    model: str,
    ctx: Context,
    image: Optional[genai_types.Image] = None,
    poll_interval: int = 10,
    max_poll_time: int = 600
) -> dict:
    """Generate a video using Veo 3 with progress tracking"""
    
    start_time = time.time()
    
    try:
        await ctx.info(f"Starting video generation with model: {model}")
        await ctx.info(f"Prompt: {prompt[:100]}...")
        
        # Start video generation - using official API format
        await ctx.report_progress(progress=5, total=100)
        
        try:
            config = genai_types.GenerateVideosConfig(
                resolution = "720p",
                duration_seconds=8
            )
          
            if image is not None:
                operation = gemini_client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    image=image
                    config=config
                )
            else:
                operation = gemini_client.models.generate_videos(
                    model=model,
                    prompt=prompt
                    config=config
                )
        
        except genai_errors.APIError as e:
            if e.code == 429:
                raise VeoSubmissionRateLimitError(
                    f"{model}: generation request rejected with HTTP 429"
                ) from e
            raise
        
        await ctx.report_progress(progress=10, total=100)
        
        # Poll for completion with progress updates
        while not operation.done:
            elapsed = time.time() - start_time
            if elapsed > max_poll_time:
                await ctx.report_progress(progress=0, total=100)  # Reset on timeout
                raise TimeoutError(f"Video generation timed out after {max_poll_time} seconds")
            
            # Calculate progress based on elapsed time (rough estimate)
            # Most generations take 30-300 seconds, so we'll estimate progress
            estimated_progress = min(10 + (elapsed / 300) * 80, 85)  # Cap at 85% until done
            await ctx.report_progress(progress=int(estimated_progress), total=100)
            
            await ctx.info(f"Generating video... ({elapsed:.1f}s elapsed)")
            await asyncio.sleep(poll_interval)
            operation = gemini_client.operations.get(operation)
        
        # Check if generation was successful
        if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
            await ctx.report_progress(progress=0, total=100)  # Reset on error
            raise RuntimeError("Video generation failed - no videos in response")
        
        generated_video = operation.response.generated_videos[0]
        
        await ctx.report_progress(progress=90, total=100)
        
        # Ensure output directory exists
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"veo3_video_{timestamp}_{uuid.uuid4().hex}.mp4"
        output_path = output_dir / filename
        
        # Download the video
        await ctx.info(f"Downloading video to: {output_path}")
        gemini_client.files.download(file=generated_video.video)
        generated_video.video.save(str(output_path))
        
        await ctx.report_progress(progress=100, total=100)
        
        file_size = output_path.stat().st_size if output_path.exists() else 0
        generation_time = time.time() - start_time
        
        await ctx.info(f"Video generation completed in {generation_time:.1f} seconds")
        
        return {
            "video_path": str(output_path),
            "filename": filename,
            "model": model,
            "prompt": prompt,
            "negative_prompt": None,  # Not supported in current API
            "generation_time": generation_time,
            "file_size": file_size,
            "aspect_ratio": "16:9"  # Default for Veo 3
        }
        
    except Exception as e:
        logger.exception("Video generation failed")

        try:
            await ctx.report_progress(progress=0, total=100)  # Reset on error
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
    model: str = "veo-3.1-lite-generate-preview"
) -> VideoGenerationResponse:
    """Generate a video using Google Veo 3 from a text prompt
    
    Args:
        prompt: Text prompt describing the video to generate
        model: Veo model to use 
    
    Returns:
        VideoGenerationResponse with video path, metadata, and generation info
    
    Note: Veo 3 generates 8-second 720p videos with audio. Aspect ratio and other advanced 
    parameters are not currently supported in the public API.
    """
    
    await ctx.info(f"Starting video generation with prompt: {prompt[:100]}...")
    
    if not prompt.strip():
        await ctx.error("Prompt cannot be empty")
        raise ValueError("Prompt cannot be empty")
    
    # Validate model
    valid_models = ["veo-3.1-lite-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.1-generate-preview" ]
  
    if model and model in valid_models:
        models_to_try = [model] + [candidate for candidate in valid_models if candidate != model]
    else:
        models_to_try = valid_models

    result = None
    max_veo_retries = len(models_to_try)
    for attempt in range(max_veo_retries):
        current_model = models_to_try[attempt]
  
        try:
            result = await generate_video_with_progress(
                prompt=prompt,
                model=current_model,
                ctx=ctx
            )
            
            logger.info(f"Video generated successfully: {result['filename']}")
            break
          
        except VeoSubmissionRateLimitError as rle:
            if attempt == max_veo_retries - 1:
                logger.error("All models rejected with 429")
                raise
            logger.warning(f"Model {current_model} quota exceeded. Falling back to next model...")
            await asyncio.sleep(2)
  
        except Exception as e:
            logger.error(f"Non-retryable error with model {current_model}: {e}")
            await ctx.error(f"Generation failed with non-retryable error: {str(e)}")
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
            

@mcp.tool()
async def generate_video_from_image(
    prompt: str,
    image_path: str,
    ctx: Context,
    model: str = "veo-3.1-lite-generate-preview"
) -> VideoGenerationResponse:
    """Generate a video using Google Veo 3 from an image and text prompt
    
    Args:
        prompt: Text prompt describing the video motion/action
        image_path: Path to the starting image file
        model: Veo model to use 
    
    Returns:
        VideoGenerationResponse with video path, metadata, and generation info
        
    Note: Veo 3 generates 8-second 720p videos with audio. Advanced parameters like 
    negative prompts and aspect ratios are not currently supported in the public API.
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
    
    # Validate model
    valid_models = ["veo-3.1-lite-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.1-generate-preview" ]

    if model and model in valid_models:
        models_to_try = [model] + [candidate for candidate in valid_models if candidate != model]
    else:
        models_to_try = valid_models

    result = None
  
    max_veo_retries = len(models_to_try)
    for attempt in range(max_veo_retries):
        current_model = models_to_try[attempt]

        try:
            image_obj = genai_types.Image(
                image_bytes=image_data, 
                mime_type=safe_mime_type
            )
          
            result = await generate_video_with_progress(
                prompt=prompt,
                model=current_model,
                ctx=ctx,
                image=image_obj
            )
            
            logger.info(f"Image-to-video generation successful: {result['filename']}")
            break

        except VeoSubmissionRateLimitError as rle:
            if attempt == max_veo_retries - 1:
                logger.error("All models rejected with 429")
                raise
            logger.warning(f"Model {current_model} quota exceeded. Falling back to next model...")
            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"Non-retryable error with model {current_model}: {e}")
            await ctx.error(f"Generation failed with non-retryable error: {str(e)}")
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
