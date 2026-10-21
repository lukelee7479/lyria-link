#!/usr/bin/env python3
"""
Basic test script for Veo 3 video generation

This script demonstrates basic text-to-video generation using the MCP Veo 3 server.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import the MCP server
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_omni import generate_video_with_progress, gemini_client

async def test_basic_generation():
    """Test basic video generation"""
    print("🎬 Testing Basic Veo 3 Video Generation")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        return
    
    try:
        # Initialize client
        print("🔧 Initializing Veo 3 client...")
        client = Veo3Client(api_key)
        
        # Configure generation
        config = VideoGenerationConfig(
            model="veo-3.0-fast-generate-preview",  # Use fast model for testing
            aspect_ratio="16:9",
            output_dir="test_videos"
        )
        
        # Test prompt
        prompt = "A peaceful mountain lake reflecting the sunset, with gentle ripples on the water surface"
        
        print(f"📝 Prompt: {prompt}")
        print(f"🎬 Model: {config.model}")
        print(f"📐 Aspect Ratio: {config.aspect_ratio}")
        print("\n🚀 Starting video generation...")
        print("⏱️  This may take 1-6 minutes depending on server load...")
        
        # Generate video
        result = await client.generate_video(prompt, config)
        
        # Display results
        if result["success"]:
            print("\n✅ Video generation completed successfully!")
            print(f"📁 File: {result['video_path']}")
            print(f"⏱️ Generation time: {result['generation_time']:.1f} seconds")
            print(f"📏 File size: {result['file_size'] / 1024 / 1024:.1f} MB")
        else:
            print(f"\n❌ Video generation failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

async def test_with_negative_prompt():
    """Test video generation with negative prompt"""
    print("\n🚫 Testing Video Generation with Negative Prompt")
    print("=" * 50)
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        return
    
    try:
        client = Veo3Client(api_key)
        
        config = VideoGenerationConfig(
            model="veo-3.0-fast-generate-preview",
            aspect_ratio="16:9",
            output_dir="test_videos"
        )
        
        prompt = "A serene forest clearing with tall trees and dappled sunlight"
        negative_prompt = "people, animals, buildings, vehicles, noise"
        
        print(f"📝 Prompt: {prompt}")
        print(f"🚫 Negative Prompt: {negative_prompt}")
        print("\n🚀 Starting video generation with negative prompt...")
        
        result = await client.generate_video(prompt, config, negative_prompt)
        
        if result["success"]:
            print("\n✅ Video with negative prompt completed!")
            print(f"📁 File: {result['video_path']}")
            print(f"⏱️ Generation time: {result['generation_time']:.1f} seconds")
        else:
            print(f"\n❌ Generation failed: {result['error']}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")

async def main():
    """Run all tests"""
    print("🎯 Veo 3 MCP Server Test Suite")
    print("=" * 50)
    
    await test_basic_generation()
    await test_with_negative_prompt()
    
    print("\n🎉 Test suite completed!")
    print("\n💡 Tips:")
    print("- Check the 'test_videos' directory for generated videos")
    print("- Videos are also stored on Google's servers for 2 days")
    print("- Use different models for different speed/quality tradeoffs")

if __name__ == "__main__":
    asyncio.run(main())
