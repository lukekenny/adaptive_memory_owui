#!/usr/bin/env python3
"""
Unit Test Runner for OWUI Adaptive Memory Plugin

This script runs the comprehensive unit test suite for the monolithic
Adaptive Memory Filter, testing all core functionality while respecting
the OpenWebUI Filter Function architecture.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run the unit test suite and report results."""
    
    print("=" * 80)
    print("OWUI Adaptive Memory Plugin - Unit Test Suite")
    print("=" * 80)
    print()
    
    # Test categories to run
    test_categories = [
        {
            "name": "Basic Filter Tests",
            "path": "tests/unit/test_filter_basic.py",
            "description": "Tests basic filter functionality and structure"
        },
        {
            "name": "Memory Operations",
            "path": "tests/unit/test_memory_operations.py",
            "description": "Tests memory extraction, storage, and retrieval"
        },
        {
            "name": "API Compatibility",
            "path": "tests/unit/test_api_compatibility.py",
            "description": "Tests API compatibility and version handling"
        },
        {
            "name": "Filter Comprehensive - OpenWebUI Compliance",
            "path": "tests/unit/test_filter_comprehensive.py::TestFilterOpenWebUICompliance",
            "description": "Tests compliance with OpenWebUI Filter Function interface"
        },
        {
            "name": "Filter Error Handling",
            "path": "tests/unit/test_filter_comprehensive.py::TestFilterErrorHandling",
            "description": "Tests error handling and edge cases"
        },
        {
            "name": "Filter Performance",
            "path": "tests/unit/test_filter_comprehensive.py::TestFilterPerformance",
            "description": "Tests performance characteristics"
        },
        {
            "name": "Filter Thread Safety",
            "path": "tests/unit/test_filter_comprehensive.py::TestFilterThreadSafety",
            "description": "Tests thread safety implementation"
        },
        {
            "name": "Orchestration System",
            "path": "tests/unit/test_orchestration_system.py",
            "description": "Tests filter orchestration functionality"
        },
        {
            "name": "API Integration",
            "path": "tests/unit/test_api_integration.py",
            "description": "Tests API integration functionality"
        },
        {
            "name": "Monolithic Structure Compliance",
            "path": "tests/unit/test_filter_comprehensive.py::TestFilterMonolithicCompliance",
            "description": "Tests compliance with monolithic structure requirements"
        }
    ]
    
    total_categories = len(test_categories)
    passed_categories = 0
    
    results = []
    
    for i, category in enumerate(test_categories, 1):
        print(f"[{i}/{total_categories}] Running: {category['name']}")
        print(f"Description: {category['description']}")
        print("-" * 60)
        
        try:
            # Run the test category
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                category["path"],
                "-v", "--tb=short", 
                "--disable-warnings"
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                print("✓ PASSED")
                passed_categories += 1
                status = "PASSED"
            else:
                print("✗ FAILED")
                status = "FAILED"
                # Print failure details
                if result.stdout:
                    print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                if result.stderr:
                    print("STDERR:", result.stderr[-500:])  # Last 500 chars
            
            results.append({
                "name": category["name"],
                "status": status,
                "returncode": result.returncode
            })
            
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append({
                "name": category["name"],
                "status": "ERROR",
                "error": str(e)
            })
        
        print()
    
    # Print summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    
    for result in results:
        status_symbol = "✓" if result["status"] == "PASSED" else "✗"
        print(f"{status_symbol} {result['name']}: {result['status']}")
    
    print()
    print(f"Categories Passed: {passed_categories}/{total_categories}")
    print(f"Success Rate: {(passed_categories/total_categories)*100:.1f}%")
    print()
    
    if passed_categories == total_categories:
        print("🎉 ALL TESTS PASSED! The monolithic Filter Function implementation")
        print("   is compliant with OpenWebUI requirements and passes all unit tests.")
        return True
    else:
        print("⚠️  Some tests failed. Review the output above for details.")
        return False

def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print("Running Quick Smoke Test...")
    print("-" * 40)
    
    try:
        # Test basic imports and instantiation
        result = subprocess.run([
            sys.executable, "-c", """
import sys
sys.path.append('tests')
from conftest import Filter

# Test basic functionality
filter_instance = Filter()
print('✓ Filter instantiation successful')

# Test basic methods
test_body = {'messages': [], 'user': {'id': 'test'}}
inlet_result = filter_instance.inlet(test_body)
print('✓ Inlet method works')

outlet_result = filter_instance.outlet(test_body) 
print('✓ Outlet method works')

stream_result = filter_instance.stream({'type': 'test'})
print('✓ Stream method works')

print('✓ Smoke test PASSED')
"""
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print("✗ Smoke test FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Smoke test ERROR: {e}")
        return False

if __name__ == "__main__":
    print("Starting OWUI Adaptive Memory Plugin Unit Tests")
    print()
    
    # Check if we're in the right directory
    if not Path("adaptive_memory_v4.0.py").exists():
        print("Error: adaptive_memory_v4.0.py not found in current directory")
        print("Please run this script from the project root directory")
        sys.exit(1)
    
    # Run smoke test first
    if not run_quick_smoke_test():
        print("Smoke test failed. Skipping full test suite.")
        sys.exit(1)
    
    print()
    
    # Run full test suite
    success = run_tests()
    
    if success:
        print("Unit test implementation for Task 20.2 completed successfully!")
        sys.exit(0)
    else:
        print("Some tests failed. See output above for details.")
        sys.exit(1)