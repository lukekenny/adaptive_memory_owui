#!/usr/bin/env python3
"""
Validation script for configuration integration tests.

This script validates the test setup and runs basic checks to ensure
the configuration integration tests are properly set up and functional.

Usage:
    python validate_configuration_tests.py [--fix] [--verbose]
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if all required dependencies are available"""
    required_packages = [
        'pytest',
        'pytest_asyncio',
        'pytest_cov', 
        'pytest_html',
        'aiohttp',
        'psutil',
        'pydantic'
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    
    return len(missing) == 0, missing


def check_test_files() -> Tuple[bool, List[str]]:
    """Check if all required test files exist"""
    test_dir = Path(__file__).parent
    
    required_files = [
        'test_configuration_integration.py',
        'test_valves_configuration.py',
        'config_test_settings.py',
        'run_configuration_tests.py',
        'README_Configuration_Tests.md'
    ]
    
    missing = []
    
    for file in required_files:
        file_path = test_dir / file
        if not file_path.exists():
            missing.append(str(file_path))
    
    return len(missing) == 0, missing


def check_filter_import() -> Tuple[bool, str]:
    """Check if the main filter can be imported"""
    try:
        import importlib.util
        
        # Load the filter module directly
        project_root = Path(__file__).parent.parent.parent
        filter_path = project_root / "adaptive_memory_v4.0.py"
        
        if not filter_path.exists():
            return False, f"Filter file not found: {filter_path}"
        
        spec = importlib.util.spec_from_file_location("adaptive_memory_v4_0", filter_path)
        adaptive_memory_v4_0 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adaptive_memory_v4_0)
        
        Filter = adaptive_memory_v4_0.Filter
        filter_instance = Filter()
        
        # Check if Valves exists
        if not hasattr(filter_instance, 'valves'):
            return False, "Filter instance does not have 'valves' attribute"
        
        # Check if key methods exist
        required_methods = [
            'validate_configuration_before_ui_save',
            'inlet',
            'outlet'
        ]
        
        for method in required_methods:
            if not hasattr(filter_instance, method):
                return False, f"Filter missing required method: {method}"
        
        return True, "Filter import successful"
        
    except Exception as e:
        return False, f"Filter import failed: {e}"


def check_test_syntax() -> Tuple[bool, List[str]]:
    """Check Python syntax of test files"""
    test_dir = Path(__file__).parent
    
    test_files = [
        'test_configuration_integration.py',
        'test_valves_configuration.py', 
        'config_test_settings.py'
    ]
    
    errors = []
    
    for file in test_files:
        file_path = test_dir / file
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    compile(f.read(), str(file_path), 'exec')
            except SyntaxError as e:
                errors.append(f"{file}: {e}")
            except Exception as e:
                errors.append(f"{file}: Unexpected error: {e}")
    
    return len(errors) == 0, errors


def check_pytest_configuration() -> Tuple[bool, str]:
    """Check pytest configuration and basic functionality"""
    try:
        # Try to run pytest --collect-only to check test collection
        test_dir = Path(__file__).parent
        cmd = [
            sys.executable, '-m', 'pytest', 
            '--collect-only', 
            'test_configuration_integration.py',
            '-q'
        ]
        
        result = subprocess.run(
            cmd, 
            cwd=test_dir,
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return True, f"Pytest collection successful: {result.stdout.count('test')} tests found"
        else:
            return False, f"Pytest collection failed: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return False, "Pytest collection timed out"
    except Exception as e:
        return False, f"Pytest check failed: {e}"


def run_basic_test() -> Tuple[bool, str]:
    """Run a basic test to verify functionality"""
    try:
        from config_test_settings import ConfigurationFactory, TEST_CONFIG
        
        # Test configuration factory
        minimal_config = ConfigurationFactory.create_minimal_config()
        full_config = ConfigurationFactory.create_full_config()
        
        # Basic validation
        if not isinstance(minimal_config, dict):
            return False, "ConfigurationFactory.create_minimal_config() did not return dict"
        
        if not isinstance(full_config, dict):
            return False, "ConfigurationFactory.create_full_config() did not return dict"
        
        if 'llm_provider_type' not in minimal_config:
            return False, "Minimal config missing required field"
        
        # Test configuration constants
        if not isinstance(TEST_CONFIG, dict):
            return False, "TEST_CONFIG is not a dictionary"
        
        required_config_keys = [
            'DEFAULT_TIMEOUT',
            'MAX_CONCURRENT_OPERATIONS', 
            'MIN_SUCCESS_RATE'
        ]
        
        for key in required_config_keys:
            if key not in TEST_CONFIG:
                return False, f"TEST_CONFIG missing required key: {key}"
        
        return True, "Basic functionality test passed"
        
    except Exception as e:
        return False, f"Basic test failed: {e}"


def fix_issues(issues: Dict[str, Any], verbose: bool = False) -> bool:
    """Attempt to fix identified issues"""
    fixed = False
    
    # Install missing dependencies
    if 'dependencies' in issues and issues['dependencies']['missing']:
        if verbose:
            print("Attempting to install missing dependencies...")
        
        missing = issues['dependencies']['missing']
        pip_packages = {
            'pytest_asyncio': 'pytest-asyncio',
            'pytest_cov': 'pytest-cov',
            'pytest_html': 'pytest-html'
        }
        
        install_packages = []
        for package in missing:
            install_name = pip_packages.get(package, package)
            install_packages.append(install_name)
        
        try:
            cmd = [sys.executable, '-m', 'pip', 'install'] + install_packages
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                if verbose:
                    print(f"Successfully installed: {', '.join(install_packages)}")
                fixed = True
            else:
                if verbose:
                    print(f"Failed to install packages: {result.stderr}")
        except Exception as e:
            if verbose:
                print(f"Error installing packages: {e}")
    
    return fixed


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate configuration integration tests")
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🔍 Validating Configuration Integration Tests")
    print("=" * 50)
    
    issues = {}
    all_passed = True
    
    # Check dependencies
    print("1. Checking dependencies...")
    deps_ok, missing_deps = check_dependencies()
    if deps_ok:
        print("   ✅ All dependencies available")
    else:
        print(f"   ❌ Missing dependencies: {', '.join(missing_deps)}")
        issues['dependencies'] = {'missing': missing_deps}
        all_passed = False
    
    # Check test files
    print("2. Checking test files...")
    files_ok, missing_files = check_test_files()
    if files_ok:
        print("   ✅ All test files present")
    else:
        print(f"   ❌ Missing files: {', '.join(missing_files)}")
        issues['files'] = {'missing': missing_files}
        all_passed = False
    
    # Check filter import
    print("3. Checking filter import...")
    filter_ok, filter_msg = check_filter_import()
    if filter_ok:
        print(f"   ✅ {filter_msg}")
    else:
        print(f"   ❌ {filter_msg}")
        issues['filter'] = {'error': filter_msg}
        all_passed = False
    
    # Check test syntax
    print("4. Checking test syntax...")
    syntax_ok, syntax_errors = check_test_syntax()
    if syntax_ok:
        print("   ✅ All test files have valid syntax")
    else:
        print("   ❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"      - {error}")
        issues['syntax'] = {'errors': syntax_errors}
        all_passed = False
    
    # Check pytest configuration
    print("5. Checking pytest configuration...")
    pytest_ok, pytest_msg = check_pytest_configuration()
    if pytest_ok:
        print(f"   ✅ {pytest_msg}")
    else:
        print(f"   ❌ {pytest_msg}")
        issues['pytest'] = {'error': pytest_msg}
        all_passed = False
    
    # Run basic test
    print("6. Running basic functionality test...")
    basic_ok, basic_msg = run_basic_test()
    if basic_ok:
        print(f"   ✅ {basic_msg}")
    else:
        print(f"   ❌ {basic_msg}")
        issues['basic'] = {'error': basic_msg}
        all_passed = False
    
    print("\n" + "=" * 50)
    
    # Attempt to fix issues if requested
    if args.fix and issues:
        print("🔧 Attempting to fix issues...")
        fixed = fix_issues(issues, args.verbose)
        if fixed:
            print("   ✅ Some issues were fixed. Re-run validation to verify.")
        else:
            print("   ⚠️  Unable to automatically fix issues.")
        print()
    
    # Final status
    if all_passed:
        print("🎉 All validation checks passed!")
        print("\nYou can now run the configuration integration tests:")
        print("   python run_configuration_tests.py")
        print("   python run_configuration_tests.py --suite config")
        return 0
    else:
        print("❌ Some validation checks failed.")
        
        if not args.fix:
            print("\nTry running with --fix to automatically fix some issues:")
            print("   python validate_configuration_tests.py --fix")
        
        print("\nManual fixes may be required for:")
        for category, details in issues.items():
            print(f"   - {category}: {details}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())