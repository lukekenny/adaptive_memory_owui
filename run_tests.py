#!/usr/bin/env python3
"""
Test runner for OWUI Adaptive Memory Plugin.

This script provides convenient test execution with various options
for different testing scenarios.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, capture_output=False):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    else:
        return subprocess.run(cmd).returncode


def install_dependencies():
    """Install testing dependencies."""
    print("Installing testing dependencies...")
    return run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def run_linting():
    """Run linting checks."""
    print("Running linting checks...")
    return run_command(["ruff", "check", "adaptive_memory_v4.0.py"])


def run_type_checking():
    """Run type checking."""
    print("Running type checking...")
    return run_command(["mypy", "adaptive_memory_v4.0.py", "--ignore-missing-imports"])


def run_tests(test_type="all", verbose=False, coverage=False):
    """Run tests with specified options."""
    cmd = [sys.executable, "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov", "--cov-report=html", "--cov-report=term"])
    
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "functional":
        cmd.append("tests/functional/")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        cmd.append(test_type)  # Allow custom test path
    
    return run_command(cmd)


def run_performance_tests():
    """Run performance tests."""
    print("Running performance tests...")
    return run_command([
        sys.executable, "-m", "pytest", 
        "tests/", "-m", "performance", 
        "--timeout=60", "-v"
    ])


def run_security_scan():
    """Run security scanning."""
    print("Running security scan...")
    return run_command(["bandit", "-r", "adaptive_memory_v4.0.py"])


def run_full_validation():
    """Run complete validation suite."""
    print("Running full validation suite...")
    
    steps = [
        ("Installing dependencies", install_dependencies),
        ("Linting", run_linting),
        ("Type checking", run_type_checking),
        ("Unit tests", lambda: run_tests("unit", verbose=True, coverage=True)),
        ("Integration tests", lambda: run_tests("integration", verbose=True)),
        ("Performance tests", run_performance_tests),
    ]
    
    results = []
    for step_name, step_func in steps:
        print(f"\n{'='*60}")
        print(f"STEP: {step_name}")
        print(f"{'='*60}")
        
        result = step_func()
        results.append((step_name, result))
        
        if result != 0:
            print(f"❌ {step_name} FAILED")
        else:
            print(f"✅ {step_name} PASSED")
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result == 0)
    total = len(results)
    
    for step_name, result in results:
        status = "✅ PASSED" if result == 0 else "❌ FAILED"
        print(f"{step_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} steps passed")
    return 0 if passed == total else 1


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="OWUI Adaptive Memory Plugin Test Runner")
    parser.add_argument("--install-deps", action="store_true", help="Install dependencies")
    parser.add_argument("--lint", action="store_true", help="Run linting only")
    parser.add_argument("--type-check", action="store_true", help="Run type checking only")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--functional", action="store_true", help="Run functional tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--security", action="store_true", help="Run security scan only")
    parser.add_argument("--full", action="store_true", help="Run full validation suite")
    parser.add_argument("--coverage", action="store_true", help="Include coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--path", help="Custom test path")
    
    args = parser.parse_args()
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    if args.install_deps:
        return install_dependencies()
    elif args.lint:
        return run_linting()
    elif args.type_check:
        return run_type_checking()
    elif args.unit:
        return run_tests("unit", args.verbose, args.coverage)
    elif args.integration:
        return run_tests("integration", args.verbose, args.coverage)
    elif args.functional:
        return run_tests("functional", args.verbose, args.coverage)
    elif args.performance:
        return run_performance_tests()
    elif args.security:
        return run_security_scan()
    elif args.full:
        return run_full_validation()
    elif args.path:
        return run_tests(args.path, args.verbose, args.coverage)
    else:
        # Default: run all tests
        return run_tests("all", args.verbose, args.coverage)


if __name__ == "__main__":
    sys.exit(main())