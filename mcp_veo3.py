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
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastmcp import FastMCP, Context
from pydantic import BaseModel
from dotenv import load_dotenv

from mcp_s3 import upload_file, ROOT, BUCKET, s3

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
args = parser.parse_args()

OUTPUT_DIR = os.path.abspath(os.path.expanduser(args.output_dir))

# Get API key from CLI args or environment
API_KEY = args.api_key or os.getenv("NANOBANANA_API_KEY")
if not API_KEY:
    raise ValueError("Gemini API key must be provided via --api-key argument or GEMINI_API_KEY in .env file")

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
def safe_join(root: str, user_path: str) -> str:
    """Safely join paths and prevent directory traversal"""
    abs_path = os.path.abspath(os.path.join(root, user_path))
    if not abs_path.startswith(root):
        raise ValueError("Path escapes allowed root")
    return abs_path


async def generate_video_with_progress(
    prompt: str,
    model: str,
    ctx: Context,
    image_path: Optional[str] = None,
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
        
        if image_path and os.path.exists(image_path):
            await ctx.info(f"Uploading image: {image_path}")
            image_file = gemini_client.files.upload(path=image_path)
            # For image-to-video, we need to pass the image
            operation = gemini_client.models.generate_videos(
                model=model,
                prompt=prompt,
                image=image_file
            )
        else:
            # For text-to-video, only model and prompt are needed
            operation = gemini_client.models.generate_videos(
                model=model,
                prompt=prompt
            )
        
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
        filename = f"veo3_video_{timestamp}.mp4"
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
        await ctx.report_progress(progress=0, total=100)  # Reset on error
        await ctx.error(f"Video generation failed: {str(e)}")
        raise ValueError(f"Video generation failed: {str(e)}")

@mcp.tool()
async def generate_video(
    prompt: str,
    ctx: Context,
    model: str = "veo-3.0-generate-preview"
) -> VideoGenerationResponse:
    """Generate a video using Google Veo 3 from a text prompt
    
    Args:
        prompt: Text prompt describing the video to generate
        model: Veo model to use (veo-3.0-generate-preview, veo-3.0-fast-generate-preview, veo-2.0-generate-001)
    
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
    valid_models = ["veo-3.1-generate-preview", "veo-3.1-fast-generate-preview", "veo-3.1-lite-generate-preview"]
    if model not in valid_models:
        await ctx.error(f"Invalid model: {model}. Must be one of: {valid_models}")
        raise ValueError(f"Invalid model: {model}")
    
    try:
        result = await generate_video_with_progress(
            prompt=prompt,
            model=model,
            ctx=ctx
        )
        
        await ctx.info(f"Video generated successfully: {result['filename']}")
        
        ##return VideoGenerationResponse(**result)
        
    except Exception as e:
        await ctx.error(f"Video generation failed: {str(e)}")
        raise ValueError(f"Video generation failed: {str(e)}")
     try:
            logger.info("Trying S3 upload")
          
            temp_filename = f"{uuid.uuid4()}.mp4"
            temp_path = os.path.join(ROOT, temp_filename)
            try:
                with open(temp_path, "wb") as f:
                    f.write(result)
                s3_response = await upload_file(local_path=temp_filename, ctx=ctx)
                uploaded_url = s3_response.url
                logger.info("s3 upload done")
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

@mcp.tool()
async def generate_video_from_image(
    prompt: str,
    image_path: str,
    ctx: Context,
    model: str = "veo-3.0-generate-preview"
) -> VideoGenerationResponse:
    """Generate a video using Google Veo 3 from an image and text prompt
    
    Args:
        prompt: Text prompt describing the video motion/action
        image_path: Path to the starting image file
        model: Veo model to use (veo-3.0-generate-preview, veo-3.0-fast-generate-preview, veo-2.0-generate-001)
    
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
    
    # Resolve image path (allow relative paths within output directory for security)
    if not os.path.isabs(image_path):
        full_image_path = safe_join(OUTPUT_DIR, image_path)
    else:
        full_image_path = image_path
    
    if not os.path.exists(full_image_path):
        await ctx.error(f"Image file not found: {full_image_path}")
        raise ValueError(f"Image file not found: {full_image_path}")
    
    # Validate model
    valid_models = ["veo-3.0-generate-preview", "veo-3.0-fast-generate-preview", "veo-2.0-generate-001"]
    if model not in valid_models:
        await ctx.error(f"Invalid model: {model}. Must be one of: {valid_models}")
        raise ValueError(f"Invalid model: {model}")
    
    try:
        result = await generate_video_with_progress(
            prompt=prompt,
            model=model,
            ctx=ctx,
            image_path=full_image_path
        )
        
        await ctx.info(f"Image-to-video generation successful: {result['filename']}")
        
        return VideoGenerationResponse(**result)
        
    except Exception as e:
        await ctx.error(f"Image-to-video generation failed: {str(e)}")
        raise ValueError(f"Image-to-video generation failed: {str(e)}")


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
