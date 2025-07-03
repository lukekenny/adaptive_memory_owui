#!/usr/bin/env python3
"""
Basic framework validation script for OWUI Adaptive Memory Plugin.

This script validates the testing framework setup without requiring
external dependencies like pytest.
"""

import sys
import os
import importlib.util
from pathlib import Path


def test_import_main_module():
    """Test that the main module can be imported."""
    try:
        # Try to import the main module
        spec = importlib.util.spec_from_file_location(
            "adaptive_memory_v4_0", 
            "adaptive_memory_v4.0.py"
        )
        if spec is None:
            return False, "Could not create module spec"
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check for required classes
        if not hasattr(module, 'Filter'):
            return False, "Filter class not found"
        
        if not hasattr(module, 'Valves'):
            return False, "Valves class not found"
        
        return True, "Main module imported successfully"
        
    except Exception as e:
        return False, f"Import error: {str(e)}"


def test_filter_initialization():
    """Test Filter class initialization."""
    try:
        # Import module
        spec = importlib.util.spec_from_file_location(
            "adaptive_memory_v4_0", 
            "adaptive_memory_v4.0.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Create Filter instance
        filter_instance = module.Filter()
        
        # Check required methods
        required_methods = ['inlet', 'outlet', 'stream']
        for method in required_methods:
            if not hasattr(filter_instance, method):
                return False, f"Missing method: {method}"
            if not callable(getattr(filter_instance, method)):
                return False, f"Method {method} is not callable"
        
        return True, "Filter class initialized successfully"
        
    except Exception as e:
        return False, f"Filter initialization error: {str(e)}"


def test_basic_functionality():
    """Test basic filter functionality."""
    try:
        # Import module
        spec = importlib.util.spec_from_file_location(
            "adaptive_memory_v4_0", 
            "adaptive_memory_v4.0.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Create Filter instance
        filter_instance = module.Filter()
        
        # Test with empty body
        result = filter_instance.inlet({})
        if not isinstance(result, dict):
            return False, "inlet() should return dict"
        
        result = filter_instance.outlet({})
        if not isinstance(result, dict):
            return False, "outlet() should return dict"
        
        result = filter_instance.stream({})
        if not isinstance(result, dict):
            return False, "stream() should return dict"
        
        return True, "Basic functionality test passed"
        
    except Exception as e:
        return False, f"Basic functionality error: {str(e)}"


def test_directory_structure():
    """Test that the test directory structure is correct."""
    try:
        required_dirs = [
            "tests",
            "tests/unit",
            "tests/integration",
            "tests/functional",
            "tests/fixtures",
            "tests/mocks"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            return False, f"Missing directories: {', '.join(missing_dirs)}"
        
        return True, "Directory structure is correct"
        
    except Exception as e:
        return False, f"Directory structure error: {str(e)}"


def test_configuration_files():
    """Test that configuration files exist."""
    try:
        required_files = [
            "requirements.txt",
            "pytest.ini",
            "tests/conftest.py",
            "tests/__init__.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            return False, f"Missing files: {', '.join(missing_files)}"
        
        return True, "Configuration files are present"
        
    except Exception as e:
        return False, f"Configuration files error: {str(e)}"


def test_syntax_validation():
    """Test syntax validation of test files."""
    try:
        test_files = [
            "tests/conftest.py",
            "tests/unit/test_filter_basic.py",
            "tests/integration/test_openwebui_interface.py"
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Try to compile the file
                try:
                    compile(content, file_path, 'exec')
                except SyntaxError as e:
                    return False, f"Syntax error in {file_path}: {str(e)}"
        
        return True, "Test files have valid syntax"
        
    except Exception as e:
        return False, f"Syntax validation error: {str(e)}"


def main():
    """Run all framework validation tests."""
    print("OWUI Adaptive Memory Plugin - Testing Framework Validation")
    print("=" * 60)
    
    tests = [
        ("Import Main Module", test_import_main_module),
        ("Filter Initialization", test_filter_initialization),
        ("Basic Functionality", test_basic_functionality),
        ("Directory Structure", test_directory_structure),
        ("Configuration Files", test_configuration_files),
        ("Syntax Validation", test_syntax_validation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 40)
        
        try:
            success, message = test_func()
            if success:
                print(f"✅ PASSED: {message}")
                passed += 1
            else:
                print(f"❌ FAILED: {message}")
                failed += 1
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/(passed + failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! Testing framework is ready.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review and fix issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())