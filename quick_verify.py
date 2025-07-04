#!/usr/bin/env python3
"""
Quick Post-Installation Verification Script for OWUI Adaptive Memory Plugin

A lightweight script that performs essential checks to verify the plugin
is ready to use. Perfect for CI/CD pipelines and quick manual checks.
"""

import sys
import os
import importlib.util
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version >= (3, 8):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False

def check_required_files():
    """Check for required files."""
    plugin_dir = Path(__file__).parent
    required_files = ["adaptive_memory_v4.0.py", "requirements.txt"]
    
    all_present = True
    for file in required_files:
        if (plugin_dir / file).exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            all_present = False
    
    return all_present

def check_filter_class():
    """Check if Filter class can be loaded."""
    try:
        plugin_dir = Path(__file__).parent
        spec = importlib.util.spec_from_file_location(
            "adaptive_memory_v4",
            plugin_dir / "adaptive_memory_v4.0.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, "Filter"):
            Filter = getattr(module, "Filter")
            
            # Check required methods
            required_methods = ["inlet", "outlet", "stream"]
            missing = [m for m in required_methods if not hasattr(Filter, m)]
            
            if not missing:
                print("✅ Filter class with all required methods")
                return True
            else:
                print(f"❌ Filter missing methods: {', '.join(missing)}")
                return False
        else:
            print("❌ Filter class not found")
            return False
            
    except Exception as e:
        print(f"❌ Error loading plugin: {str(e)}")
        return False

def check_basic_dependencies():
    """Check if basic dependencies are available."""
    required = ["pydantic", "numpy", "aiohttp", "pytz"]
    missing = []
    
    for package in required:
        try:
            importlib.import_module(package)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} missing")
            missing.append(package)
    
    return len(missing) == 0

def main():
    """Run quick verification."""
    print("=" * 50)
    print("OWUI Adaptive Memory Plugin - Quick Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Files", check_required_files),
        ("Filter Class", check_filter_class),
        ("Dependencies", check_basic_dependencies)
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        results.append(check_func())
    
    print("\n" + "=" * 50)
    
    if all(results):
        print("✅ VERIFICATION PASSED - Plugin is ready!")
        print("\nNext steps:")
        print("1. Upload adaptive_memory_v4.0.py to OpenWebUI")
        print("2. Enable the filter for your models")
        print("3. Start chatting with persistent memory!")
        return 0
    else:
        print("❌ VERIFICATION FAILED - Please fix the issues above")
        print("\nTroubleshooting:")
        print("• Install dependencies: pip install -r requirements.txt")
        print("• Run full verification: python post_install_verification.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())