#!/usr/bin/env python3
"""
Setup script for MCP Veo 3 Video Generation Server

This script helps set up the MCP Veo 3 server with proper dependencies
and configuration.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is supported"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_environment():
    """Set up environment configuration"""
    print("\n🔧 Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        # Copy example to .env
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env and add your GEMINI_API_KEY")
        return True
    else:
        print("❌ env_example.txt not found")
        return False

def check_api_key():
    """Check if API key is configured"""
    print("\n🔑 Checking API key configuration...")
    
    # Check environment variable
    if os.getenv("GEMINI_API_KEY"):
        print("✅ GEMINI_API_KEY found in environment")
        return True
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if "GEMINI_API_KEY=" in content and "your_gemini_api_key_here" not in content:
                print("✅ GEMINI_API_KEY found in .env file")
                return True
    
    print("⚠️  GEMINI_API_KEY not configured")
    print("Please:")
    print("1. Get your API key from: https://makersuite.google.com/app/apikey")
    print("2. Edit .env file and replace 'your_gemini_api_key_here' with your actual key")
    print("3. Or set environment variable: export GEMINI_API_KEY='your_key'")
    return False

def test_installation():
    """Test if the installation works"""
    print("\n🧪 Testing installation...")
    
    try:
        # Try importing the main module
        import mcp_omni
        print("✅ MCP omni module imports successfully")
        
        # Try importing dependencies
        import mcp.server.stdio
        print("✅ MCP framework available")
        
        try:
            from google import genai
            print("✅ Google GenAI SDK available")
        except ImportError:
            print("❌ Google GenAI SDK not available - check dependencies")
            return False
            
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Configure your API key in .env file if not already done")
    print("2. Test the installation:")
    print("   python examples/test_basic_generation.py")
    print("3. Run the MCP server:")
    print("   python mcp_omni.py")
    print("4. Add to your MCP client configuration:")
    print("   See README.md for configuration details")
    print("\n📚 Documentation:")
    print("- README.md - Full documentation")
    print("- examples/README.md - Example scripts")
    print("- config.json - MCP server configuration")

def main():
    """Main setup function"""
    print("🚀 MCP Veo 3 Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Set up environment
    if not setup_environment():
        print("❌ Setup failed during environment configuration")
        sys.exit(1)
    
    # Check API key
    api_key_configured = check_api_key()
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during installation test")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()
    
    if not api_key_configured:
        print("\n⚠️  Don't forget to configure your GEMINI_API_KEY!")
        sys.exit(1)
    
    print("\n✅ Setup completed successfully!")

if __name__ == "__main__":
    main()
