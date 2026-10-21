#!/usr/bin/env python3
"""
Simple test script for the FastMCP Veo 3 server

This script demonstrates basic usage of the updated FastMCP-based Veo 3 server.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def test_server_startup():
    """Test that the server starts without errors"""
    print("🚀 Testing FastMCP Veo 3 Server Startup")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your_api_key_here'")
        return False
    
    print("✅ GEMINI_API_KEY found")
    
    # Test server startup
    server_path = Path(__file__).parent.parent / "mcp_omni.py"
    test_output_dir = Path(__file__).parent / "test_videos"
    test_output_dir.mkdir(exist_ok=True)
    
    try:
        print("🔧 Starting MCP server...")
        
        # Start server process with a timeout
        process = subprocess.Popen(
            [sys.executable, str(server_path), "--output-dir", str(test_output_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Server started successfully!")
            
            # Terminate the server
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ Server terminated cleanly")
            except subprocess.TimeoutExpired:
                process.kill()
                print("⚠️ Server had to be force-killed")
            
            return True
        else:
            # Process died, get error output
            stdout, stderr = process.communicate()
            print("❌ Server failed to start")
            if stderr:
                print(f"Error: {stderr}")
            if stdout:
                print(f"Output: {stdout}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to test server: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available"""
    print("\n📦 Testing Dependencies")
    print("=" * 30)
    
    dependencies = [
        ("fastmcp", "FastMCP framework"),
        ("google.genai", "Google GenAI SDK"),
        ("pydantic", "Pydantic data models"),
        ("dotenv", "Python-dotenv")
    ]
    
    all_good = True
    
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} - not installed")
            all_good = False
    
    return all_good

def test_configuration():
    """Test configuration files"""
    print("\n🔧 Testing Configuration")
    print("=" * 30)
    
    config_files = [
        ("mcp_omni.py", "Main server file"),
        ("requirements.txt", "Dependencies file"),
        ("config.json", "MCP configuration"),
        ("env_example.txt", "Environment template")
    ]
    
    base_dir = Path(__file__).parent.parent
    all_good = True
    
    for filename, description in config_files:
        file_path = base_dir / filename
        if file_path.exists():
            print(f"✅ {description}")
        else:
            print(f"❌ {description} - not found")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🎯 FastMCP Veo 3 Server Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_dependencies():
        tests_passed += 1
        
    if test_configuration():
        tests_passed += 1
        
    if test_server_startup():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The FastMCP Veo 3 server is ready to use.")
        print("\n💡 Next steps:")
        print("1. Add the server to your MCP client configuration")
        print("2. Use the tools: generate_video, generate_video_from_image, list_generated_videos, get_video_info")
        print("3. Check the README.md for detailed usage instructions")
    else:
        print("⚠️ Some tests failed. Please check the output above and fix any issues.")
        print("\n🔧 Common fixes:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Set your API key: export GEMINI_API_KEY='your_key'")
        print("- Check that all files are present")

if __name__ == "__main__":
    main()
