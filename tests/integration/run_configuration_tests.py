#!/usr/bin/env python3
"""
Test runner for configuration integration tests.

This script runs the comprehensive configuration, error handling, and
system resilience tests for the OWUI Adaptive Memory Plugin.

Usage:
    python run_configuration_tests.py [--stress] [--verbose] [--report]
    
Options:
    --stress    Run stress and concurrency tests (takes longer)
    --verbose   Enable verbose test output
    --report    Generate HTML test report
    --coverage  Generate coverage report
    --suite     Run specific test suite (config|error|resilience|stress|valves)
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_test_suite(suite_name: str, args: argparse.Namespace) -> bool:
    """Run a specific test suite"""
    
    test_files = {
        'config': ['test_configuration_integration.py::TestConfigurationIntegration'],
        'error': ['test_configuration_integration.py::TestErrorScenarios'],
        'resilience': ['test_configuration_integration.py::TestSystemResilience'],
        'stress': ['test_configuration_integration.py::TestStressAndConcurrency'],
        'valves': ['test_valves_configuration.py'],
        'all': [
            'test_configuration_integration.py',
            'test_valves_configuration.py'
        ]
    }
    
    if suite_name not in test_files:
        print(f"Unknown test suite: {suite_name}")
        print(f"Available suites: {', '.join(test_files.keys())}")
        return False
    
    # Build pytest command
    cmd = ['python', '-m', 'pytest']
    
    # Add test files
    for test_file in test_files[suite_name]:
        cmd.append(test_file)
    
    # Add pytest options
    if args.verbose:
        cmd.extend(['-v', '-s'])
    else:
        cmd.append('-v')
    
    if args.coverage:
        cmd.extend([
            '--cov=adaptive_memory_v4',
            '--cov-report=html:htmlcov_config',
            '--cov-report=term-missing'
        ])
    
    if args.report:
        cmd.extend([
            '--html=reports/config_test_report.html',
            '--self-contained-html'
        ])
    
    # Add markers for stress tests
    if suite_name == 'stress' or (suite_name == 'all' and args.stress):
        cmd.append('-m')
        cmd.append('not slow or stress')
    elif not args.stress:
        cmd.append('-m')
        cmd.append('not stress')
    
    # Additional pytest options
    cmd.extend([
        '--tb=short',
        '--capture=no' if args.verbose else '--capture=sys',
        '--maxfail=10',
        '--durations=10'
    ])
    
    print(f"Running {suite_name} test suite...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Change to test directory
    os.chdir(Path(__file__).parent)
    
    # Run tests
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"\nTest suite '{suite_name}' completed in {duration:.2f} seconds")
    
    return result.returncode == 0


def setup_test_environment():
    """Setup test environment and dependencies"""
    print("Setting up test environment...")
    
    # Create reports directory
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Check required dependencies
    required_packages = [
        'pytest',
        'pytest-asyncio', 
        'pytest-cov',
        'pytest-html',
        'aiohttp',
        'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description="Run configuration integration tests for OWUI Adaptive Memory Plugin"
    )
    
    parser.add_argument(
        '--suite', 
        default='all',
        choices=['config', 'error', 'resilience', 'stress', 'valves', 'all'],
        help='Test suite to run'
    )
    parser.add_argument(
        '--stress', 
        action='store_true',
        help='Include stress and concurrency tests'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--report', 
        action='store_true',
        help='Generate HTML test report'
    )
    parser.add_argument(
        '--coverage', 
        action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (skip slow tests)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("OWUI Adaptive Memory Plugin - Configuration Integration Tests")
    print("=" * 60)
    
    # Setup environment
    if not setup_test_environment():
        sys.exit(1)
    
    # Override stress setting for quick mode
    if args.quick:
        args.stress = False
    
    # Run tests
    success = run_test_suite(args.suite, args)
    
    if success:
        print("\n✅ All tests passed!")
        
        if args.report:
            report_path = Path(__file__).parent / "reports" / "config_test_report.html"
            print(f"📊 Test report: {report_path}")
        
        if args.coverage:
            coverage_path = Path(__file__).parent / "htmlcov_config" / "index.html"
            print(f"📈 Coverage report: {coverage_path}")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()