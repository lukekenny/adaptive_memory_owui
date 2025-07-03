#!/usr/bin/env python3
"""
Complete testing framework validation for OWUI Adaptive Memory Plugin.

This script provides comprehensive validation of the testing framework
setup and readiness for development use.
"""

import os
import sys
from pathlib import Path


def validate_framework_structure():
    """Validate the complete framework structure."""
    print("🔍 Validating Testing Framework Structure...")
    
    required_structure = {
        'root_files': [
            'requirements.txt',
            'pytest.ini',
            'run_tests.py',
            'test_framework_basic.py',
            'TESTING_FRAMEWORK.md'
        ],
        'test_directories': [
            'tests',
            'tests/unit',
            'tests/integration',
            'tests/functional',
            'tests/fixtures',
            'tests/mocks'
        ],
        'test_files': [
            'tests/__init__.py',
            'tests/conftest.py',
            'tests/unit/__init__.py',
            'tests/unit/test_filter_basic.py',
            'tests/integration/__init__.py',
            'tests/integration/test_openwebui_interface.py',
            'tests/functional/__init__.py'
        ]
    }
    
    issues = []
    
    # Check root files
    for file_path in required_structure['root_files']:
        if not os.path.exists(file_path):
            issues.append(f"Missing file: {file_path}")
        else:
            print(f"  ✅ {file_path}")
    
    # Check directories
    for dir_path in required_structure['test_directories']:
        if not os.path.isdir(dir_path):
            issues.append(f"Missing directory: {dir_path}")
        else:
            print(f"  ✅ {dir_path}/")
    
    # Check test files
    for file_path in required_structure['test_files']:
        if not os.path.exists(file_path):
            issues.append(f"Missing test file: {file_path}")
        else:
            print(f"  ✅ {file_path}")
    
    if issues:
        print(f"\n❌ Structure validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✅ Framework structure is complete!")
        return True


def validate_configuration_files():
    """Validate configuration file contents."""
    print("\n🔍 Validating Configuration Files...")
    
    config_checks = []
    
    # Check requirements.txt
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            content = f.read()
            if 'pytest' in content:
                config_checks.append(('requirements.txt', 'pytest dependency', True))
            else:
                config_checks.append(('requirements.txt', 'pytest dependency', False))
            
            if 'sentence-transformers' in content:
                config_checks.append(('requirements.txt', 'sentence-transformers dependency', True))
            else:
                config_checks.append(('requirements.txt', 'sentence-transformers dependency', False))
    
    # Check pytest.ini
    if os.path.exists('pytest.ini'):
        with open('pytest.ini', 'r') as f:
            content = f.read()
            if 'testpaths = tests' in content:
                config_checks.append(('pytest.ini', 'testpaths configuration', True))
            else:
                config_checks.append(('pytest.ini', 'testpaths configuration', False))
            
            if 'markers' in content:
                config_checks.append(('pytest.ini', 'test markers', True))
            else:
                config_checks.append(('pytest.ini', 'test markers', False))
    
    # Check conftest.py
    if os.path.exists('tests/conftest.py'):
        with open('tests/conftest.py', 'r') as f:
            content = f.read()
            if '@pytest.fixture' in content:
                config_checks.append(('conftest.py', 'pytest fixtures', True))
            else:
                config_checks.append(('conftest.py', 'pytest fixtures', False))
    
    # Report results
    all_passed = True
    for file_name, check_name, passed in config_checks:
        if passed:
            print(f"  ✅ {file_name}: {check_name}")
        else:
            print(f"  ❌ {file_name}: {check_name}")
            all_passed = False
    
    if all_passed:
        print(f"\n✅ Configuration files are properly configured!")
    else:
        print(f"\n❌ Some configuration issues found!")
    
    return all_passed


def validate_test_files():
    """Validate test file syntax and structure."""
    print("\n🔍 Validating Test Files...")
    
    test_files = [
        'tests/conftest.py',
        'tests/unit/test_filter_basic.py',
        'tests/integration/test_openwebui_interface.py'
    ]
    
    syntax_issues = []
    
    for file_path in test_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                compile(content, file_path, 'exec')
                print(f"  ✅ {file_path}: Syntax OK")
            except SyntaxError as e:
                syntax_issues.append(f"{file_path}: {str(e)}")
                print(f"  ❌ {file_path}: Syntax Error")
        else:
            syntax_issues.append(f"{file_path}: File not found")
            print(f"  ❌ {file_path}: File not found")
    
    if syntax_issues:
        print(f"\n❌ Test file validation failed:")
        for issue in syntax_issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✅ All test files have valid syntax!")
        return True


def validate_executable_files():
    """Validate executable files."""
    print("\n🔍 Validating Executable Files...")
    
    executable_files = [
        'run_tests.py',
        'test_framework_basic.py',
        'validate_framework.py'
    ]
    
    execution_issues = []
    
    for file_path in executable_files:
        if os.path.exists(file_path):
            if os.access(file_path, os.X_OK):
                print(f"  ✅ {file_path}: Executable")
            else:
                print(f"  ⚠️  {file_path}: Not executable (can still run with python3)")
            
            # Check if it has a proper shebang
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('#!'):
                    print(f"  ✅ {file_path}: Has shebang")
                else:
                    print(f"  ⚠️  {file_path}: No shebang")
        else:
            execution_issues.append(f"{file_path}: File not found")
            print(f"  ❌ {file_path}: File not found")
    
    if execution_issues:
        print(f"\n❌ Executable file validation failed:")
        for issue in execution_issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✅ All executable files are ready!")
        return True


def validate_documentation():
    """Validate documentation completeness."""
    print("\n🔍 Validating Documentation...")
    
    doc_files = [
        'TESTING_FRAMEWORK.md',
        'README.md'
    ]
    
    doc_issues = []
    
    for file_path in doc_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                if len(content) > 100:  # Basic content check
                    print(f"  ✅ {file_path}: Has content")
                else:
                    doc_issues.append(f"{file_path}: Content too short")
                    print(f"  ❌ {file_path}: Content too short")
        else:
            if file_path == 'TESTING_FRAMEWORK.md':
                doc_issues.append(f"{file_path}: Missing documentation")
                print(f"  ❌ {file_path}: Missing documentation")
            else:
                print(f"  ⚠️  {file_path}: Not found (optional)")
    
    if doc_issues:
        print(f"\n❌ Documentation validation failed:")
        for issue in doc_issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\n✅ Documentation is complete!")
        return True


def generate_framework_report():
    """Generate a comprehensive framework report."""
    print("\n📊 Generating Framework Report...")
    
    report_data = {
        'framework_files': 0,
        'test_files': 0,
        'config_files': 0,
        'total_files': 0,
        'total_lines': 0
    }
    
    # Count files and lines
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and 'test' in file:
                report_data['test_files'] += 1
            elif file in ['requirements.txt', 'pytest.ini', '.gitignore']:
                report_data['config_files'] += 1
            elif file.endswith('.py') and any(name in file for name in ['run_tests', 'validate_framework', 'test_framework_basic']):
                report_data['framework_files'] += 1
            
            if file.endswith(('.py', '.txt', '.ini', '.md')):
                report_data['total_files'] += 1
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        report_data['total_lines'] += len(f.readlines())
                except:
                    pass
    
    print(f"  📁 Framework files: {report_data['framework_files']}")
    print(f"  🧪 Test files: {report_data['test_files']}")
    print(f"  ⚙️  Config files: {report_data['config_files']}")
    print(f"  📄 Total files: {report_data['total_files']}")
    print(f"  📏 Total lines: {report_data['total_lines']}")
    
    return report_data


def main():
    """Main validation function."""
    print("🚀 OWUI Adaptive Memory Plugin - Testing Framework Validation")
    print("=" * 70)
    
    # Run all validations
    validations = [
        ("Framework Structure", validate_framework_structure),
        ("Configuration Files", validate_configuration_files),
        ("Test Files", validate_test_files),
        ("Executable Files", validate_executable_files),
        ("Documentation", validate_documentation)
    ]
    
    passed = 0
    total = len(validations)
    
    for name, validation_func in validations:
        if validation_func():
            passed += 1
    
    # Generate report
    report_data = generate_framework_report()
    
    # Final summary
    print("\n" + "=" * 70)
    print("📋 VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Validations passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 Testing Framework Setup Complete!")
        print("✅ All validations passed")
        print("✅ Framework is ready for development")
        print("\nNext steps:")
        print("1. Install dependencies: python3 run_tests.py --install-deps")
        print("2. Run basic tests: python3 run_tests.py --unit")
        print("3. Run full test suite: python3 run_tests.py --full")
        return 0
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed")
        print("❌ Framework needs fixes before use")
        print("\nPlease review the issues above and fix them.")
        return 1


if __name__ == "__main__":
    sys.exit(main())