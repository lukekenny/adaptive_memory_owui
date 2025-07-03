#!/usr/bin/env python3
"""
Test runner for LLM integration tests.

This script runs the comprehensive LLM provider integration tests
and provides detailed reporting of test results.
"""

import sys
import os
import pytest
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_llm_integration_tests():
    """Run LLM integration tests with detailed reporting"""
    
    print("=" * 80)
    print("OWUI Adaptive Memory - LLM Integration Tests")
    print("=" * 80)
    print()
    
    # Test file path
    test_file = Path(__file__).parent / "test_llm_integration.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return 1
    
    print(f"📁 Running tests from: {test_file}")
    print()
    
    # Configure pytest arguments
    pytest_args = [
        str(test_file),
        "-v",                    # Verbose output
        "--tb=short",           # Short traceback format
        "--asyncio-mode=auto",  # Auto async mode
        "--capture=no",         # Don't capture output
        "-x",                   # Stop on first failure
        "--durations=10",       # Show 10 slowest tests
    ]
    
    # Add markers for specific test categories
    test_categories = {
        "provider_connections": "TestLLMProviderConnections",
        "memory_extraction": "TestMemoryExtractionWorkflows", 
        "error_handling": "TestErrorScenarios",
        "circuit_breaker": "TestCircuitBreakerFunctionality",
        "streaming": "TestStreamingAndFunctionCalling",
        "end_to_end": "TestEndToEndMemoryExtraction",
        "real_world": "TestRealWorldScenarios"
    }
    
    # Check if specific test category is requested
    if len(sys.argv) > 1:
        category = sys.argv[1]
        if category in test_categories:
            pytest_args.extend(["-k", test_categories[category]])
            print(f"🎯 Running {category} tests only")
        elif category == "help":
            print("Available test categories:")
            for cat, class_name in test_categories.items():
                print(f"  - {cat}: {class_name}")
            print()
            print("Usage:")
            print(f"  python {Path(__file__).name} [category]")
            print(f"  python {Path(__file__).name} help")
            return 0
        else:
            print(f"❌ Unknown test category: {category}")
            print(f"Available categories: {', '.join(test_categories.keys())}")
            return 1
    else:
        print("🚀 Running all LLM integration tests")
    
    print()
    print("Test Categories:")
    for category, description in [
        ("provider_connections", "LLM provider connection functionality"),
        ("memory_extraction", "Memory extraction and analysis workflows"),
        ("error_handling", "Error scenarios and edge cases"),
        ("circuit_breaker", "Circuit breaker functionality"),
        ("streaming", "Streaming responses and function calling"),
        ("end_to_end", "Complete memory extraction workflows"),
        ("real_world", "Realistic usage scenarios")
    ]:
        print(f"  • {category}: {description}")
    
    print()
    print("-" * 80)
    
    # Run the tests
    try:
        exit_code = pytest.main(pytest_args)
        
        print()
        print("-" * 80)
        
        if exit_code == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ Tests failed with exit code: {exit_code}")
            
        return exit_code
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def check_dependencies():
    """Check if required dependencies are available"""
    required_modules = [
        'pytest',
        'asyncio', 
        'aiohttp',
        'json'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Missing required modules: {', '.join(missing)}")
        print("Please install them with: pip install " + " ".join(missing))
        return False
    
    return True


def main():
    """Main entry point"""
    print("🔍 Checking dependencies...")
    
    if not check_dependencies():
        return 1
    
    print("✅ Dependencies OK")
    print()
    
    return run_llm_integration_tests()


if __name__ == "__main__":
    sys.exit(main())