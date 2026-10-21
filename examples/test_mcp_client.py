#!/usr/bin/env python3
"""
MCP Client test script for Veo 3 server

This script demonstrates how to interact with the Veo 3 MCP server
as a client, simulating how other applications would use the server.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

async def run_mcp_server():
    """Start the MCP server as a subprocess"""
    server_path = Path(__file__).parent.parent / "mcp_omni.py"
    
    # Set environment variables
    env = os.environ.copy()
    if not env.get("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        return None
    
    try:
        # Start server process
        process = subprocess.Popen(
            [sys.executable, str(server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=0
        )
        
        print("🚀 MCP Server started")
        return process
    except Exception as e:
        print(f"❌ Failed to start MCP server: {e}")
        return None

async def send_mcp_request(process, request):
    """Send a JSON-RPC request to the MCP server"""
    if not process or process.poll() is not None:
        print("❌ MCP server is not running")
        return None
    
    try:
        # Send request
        request_json = json.dumps(request) + "\n"
        process.stdin.write(request_json)
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline()
        if response_line:
            return json.loads(response_line.strip())
        else:
            print("❌ No response from server")
            return None
            
    except Exception as e:
        print(f"❌ Communication error: {e}")
        return None

async def test_list_tools(process):
    """Test listing available tools"""
    print("\n🛠️ Testing: List Tools")
    print("-" * 30)
    
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list"
    }
    
    response = await send_mcp_request(process, request)
    if response and "result" in response:
        tools = response["result"]["tools"]
        print(f"✅ Found {len(tools)} tools:")
        for tool in tools:
            print(f"   • {tool['name']}: {tool['description']}")
        return True
    else:
        print("❌ Failed to list tools")
        return False

async def test_generate_video(process):
    """Test video generation"""
    print("\n🎬 Testing: Generate Video")
    print("-" * 30)
    
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "generate_video",
            "arguments": {
                "prompt": "A calm ocean wave gently rolling onto a sandy beach at sunset",
                "model": "veo-3.0-fast-generate-preview",
                "aspect_ratio": "16:9",
                "output_dir": "mcp_test_videos"
            }
        }
    }
    
    print("📝 Prompt: A calm ocean wave gently rolling onto a sandy beach at sunset")
    print("🎬 Model: veo-3.0-fast-generate-preview")
    print("⏱️ This may take several minutes...")
    
    response = await send_mcp_request(process, request)
    if response and "result" in response:
        content = response["result"]["content"][0]["text"]
        print("✅ Video generation response:")
        print(content)
        return True
    else:
        print("❌ Video generation failed")
        if response and "error" in response:
            print(f"Error: {response['error']}")
        return False

async def test_list_videos(process):
    """Test listing generated videos"""
    print("\n📁 Testing: List Generated Videos")
    print("-" * 30)
    
    request = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "list_generated_videos",
            "arguments": {
                "output_dir": "mcp_test_videos"
            }
        }
    }
    
    response = await send_mcp_request(process, request)
    if response and "result" in response:
        content = response["result"]["content"][0]["text"]
        print("✅ Video listing response:")
        print(content)
        return True
    else:
        print("❌ Failed to list videos")
        return False

async def test_initialization(process):
    """Test MCP initialization"""
    print("\n🔧 Testing: MCP Initialization")
    print("-" * 30)
    
    # Initialize request
    init_request = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    response = await send_mcp_request(process, init_request)
    if response and "result" in response:
        print("✅ MCP server initialized successfully")
        print(f"Server: {response['result']['serverInfo']['name']} v{response['result']['serverInfo']['version']}")
        return True
    else:
        print("❌ MCP initialization failed")
        return False

async def main():
    """Run MCP client tests"""
    print("🎯 MCP Veo 3 Client Test Suite")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        return
    
    # Start MCP server
    process = await run_mcp_server()
    if not process:
        return
    
    try:
        # Wait for server to start
        await asyncio.sleep(2)
        
        # Run tests
        tests_passed = 0
        total_tests = 4
        
        if await test_initialization(process):
            tests_passed += 1
            
        if await test_list_tools(process):
            tests_passed += 1
            
        if await test_generate_video(process):
            tests_passed += 1
            
        if await test_list_videos(process):
            tests_passed += 1
        
        # Results
        print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
        
        if tests_passed == total_tests:
            print("🎉 All tests passed! MCP Veo 3 server is working correctly.")
        else:
            print("⚠️ Some tests failed. Check the output above for details.")
            
    finally:
        # Clean up
        if process and process.poll() is None:
            process.terminate()
            await asyncio.sleep(1)
            if process.poll() is None:
                process.kill()
            print("🛑 MCP server stopped")

if __name__ == "__main__":
    asyncio.run(main())
