#!/usr/bin/env python3
"""
Test script to validate mcp-veo3 package setup for uv/uvx and PyPI
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def run_command(cmd, cwd=None, capture_output=True):
    """Run a command and return result"""
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=capture_output, 
            text=True, 
            check=False
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def test_uv_installation():
    """Test if uv is installed and working"""
    print("🔧 Testing uv installation...")
    
    success, stdout, stderr = run_command(["uv", "--version"])
    if success:
        print(f"✅ uv is installed: {stdout.strip()}")
        return True
    else:
        print("❌ uv is not installed")
        print("Install with: curl -LsSf https://astral.sh/uv/install.sh | sh")
        return False

def test_package_structure():
    """Test package structure and files"""
    print("\n📁 Testing package structure...")
    
    project_root = Path(__file__).parent
    required_files = [
        "pyproject.toml",
        "mcp_lyria.py",
        "__init__.py",
        "__main__.py",
        "README.md",
        "LICENSE",
        "requirements.txt",
        "env_example.txt"
    ]
    
    all_good = True
    for file in required_files:
        file_path = project_root / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - missing")
            all_good = False
    
    return all_good

def test_pyproject_toml():
    """Test pyproject.toml configuration"""
    print("\n⚙️ Testing pyproject.toml configuration...")
    
    project_root = Path(__file__).parent
    pyproject_path = project_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        return False
    
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            print("⚠️ Cannot parse TOML (tomllib/tomli not available)")
            return True
    
    try:
        with open(pyproject_path, 'rb') as f:
            config = tomllib.load(f)
        
        # Check essential fields
        project = config.get('project', {})
        
        checks = [
            ('name', project.get('name') == 'mcp-veo3'),
            ('version', 'version' in project),
            ('description', 'description' in project),
            ('scripts', 'mcp-veo3' in project.get('scripts', {})),
            ('entry-points', 'veo3' in project.get('entry-points', {}).get('mcp.servers', {})),
            ('build-system', 'hatchling' in config.get('build-system', {}).get('requires', [])),
            ('dependencies', len(project.get('dependencies', [])) > 0)
        ]
        
        all_good = True
        for check_name, result in checks:
            if result:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                all_good = False
        
        return all_good
        
    except Exception as e:
        print(f"❌ Error parsing pyproject.toml: {e}")
        return False

def test_uv_sync():
    """Test uv sync functionality"""
    print("\n📦 Testing uv sync...")
    
    project_root = Path(__file__).parent
    success, stdout, stderr = run_command(["uv", "sync"], cwd=project_root)
    
    if success:
        print("✅ uv sync completed successfully")
        return True
    else:
        print("❌ uv sync failed")
        print(f"Error: {stderr}")
        return False

def test_uv_build():
    """Test package building"""
    print("\n🔨 Testing uv build...")
    
    project_root = Path(__file__).parent
    
    # Clean previous builds
    dist_dir = project_root / "dist"
    if dist_dir.exists():
        import shutil
        shutil.rmtree(dist_dir)
    
    success, stdout, stderr = run_command(["uv", "build"], cwd=project_root)
    
    if success:
        print("✅ Package built successfully")
        
        # Check built files
        if dist_dir.exists():
            built_files = list(dist_dir.iterdir())
            print(f"📦 Built files ({len(built_files)}):")
            for file in built_files:
                print(f"   - {file.name}")
            
            # Should have both wheel and source distribution
            has_wheel = any(f.name.endswith('.whl') for f in built_files)
            has_sdist = any(f.name.endswith('.tar.gz') for f in built_files)
            
            if has_wheel and has_sdist:
                print("✅ Both wheel and source distribution created")
                return True
            else:
                print("⚠️ Missing wheel or source distribution")
                return False
        else:
            print("❌ No dist directory created")
            return False
    else:
        print("❌ Package build failed")
        print(f"Error: {stderr}")
        return False

def test_uv_run():
    """Test uv run functionality"""
    print("\n🚀 Testing uv run...")
    
    project_root = Path(__file__).parent
    
    # Test uv run with --help flag (should not require API key)
    success, stdout, stderr = run_command([
        "uv", "run", "mcp-veo3", "--help"
    ], cwd=project_root)
    
    if success:
        print("✅ uv run works")
        if "--output-dir" in stdout:
            print("✅ CLI arguments properly configured")
            return True
        else:
            print("⚠️ CLI arguments may not be configured correctly")
            return False
    else:
        print("❌ uv run failed")
        print(f"Error: {stderr}")
        return False

def test_entry_points():
    """Test entry points configuration"""
    print("\n🔗 Testing entry points...")
    
    project_root = Path(__file__).parent
    
    # Test the script entry point
    success, stdout, stderr = run_command([
        "uv", "run", "python", "-c", 
        "import pkg_resources; print([ep.name for ep in pkg_resources.iter_entry_points('mcp.servers')])"
    ], cwd=project_root)
    
    if success and "veo3" in stdout:
        print("✅ MCP server entry point configured")
        return True
    else:
        print("⚠️ MCP server entry point may not be working")
        return False

def main():
    """Run all tests"""
    print("🎯 MCP Veo 3 Package Setup Test Suite")
    print("=" * 60)
    
    tests = [
        ("UV Installation", test_uv_installation),
        ("Package Structure", test_package_structure),
        ("pyproject.toml", test_pyproject_toml),
        ("UV Sync", test_uv_sync),
        ("UV Build", test_uv_build),
        ("UV Run", test_uv_run),
        ("Entry Points", test_entry_points),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Package is ready for:")
        print("   • uv run development")
        print("   • uvx global installation")
        print("   • PyPI publication")
        print("\n📋 Next steps:")
        print("   1. Test locally: python test_uv_veo3.py")
        print("   2. Build and publish: python scripts/publish.py")
        print("   3. Test global install: uvx mcp-veo3 --output-dir ~/Videos")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please fix issues before publishing.")
        
        if not any(name == "UV Installation" and result for name, result in results):
            print("\n💡 Install uv first:")
            print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
