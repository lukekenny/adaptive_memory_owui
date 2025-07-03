#!/usr/bin/env python3
"""
Integration test runner for OpenWebUI Adaptive Memory Plugin.

This script provides a comprehensive test runner that can execute
integration tests with various configurations and scenarios.
"""

import asyncio
import sys
import os
import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pytest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.integration.test_config import (
    IntegrationTestConfig,
    EnvironmentConfig,
    TestDataConfig,
    ValidationConfig
)


class IntegrationTestRunner:
    """Main test runner for integration tests"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = self._setup_logging()
        self.results: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "scenarios": {},
            "summary": {}
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration"""
        log_level = logging.DEBUG if self.args.debug else logging.INFO
        
        # Create logs directory
        log_dir = Path("tests/integration/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_file = log_dir / f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    def run(self) -> int:
        """Run integration tests based on configuration"""
        self.logger.info(f"Starting integration tests with args: {self.args}")
        
        try:
            # Validate environment
            if not self._validate_environment():
                return 1
            
            # Run tests for each scenario
            scenarios = self._get_scenarios()
            for scenario in scenarios:
                self.logger.info(f"Running scenario: {scenario.name}")
                result = self._run_scenario(scenario)
                self.results["scenarios"][scenario.name] = result
            
            # Generate summary
            self._generate_summary()
            
            # Save results
            if self.args.save_results:
                self._save_results()
            
            # Print summary
            self._print_summary()
            
            # Return exit code based on results
            return 0 if self._all_tests_passed() else 1
            
        except Exception as e:
            self.logger.error(f"Test runner failed: {e}", exc_info=True)
            return 2
    
    def _validate_environment(self) -> bool:
        """Validate test environment"""
        self.logger.info("Validating test environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error("Python 3.8+ is required")
            return False
        
        # Check required modules
        required_modules = ["pytest", "aiohttp", "sentence_transformers"]
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            self.logger.error(f"Missing required modules: {missing_modules}")
            return False
        
        # Check test data directory
        test_data_dir = Path(EnvironmentConfig.TEST_DATA_DIR)
        if not test_data_dir.exists():
            self.logger.info(f"Creating test data directory: {test_data_dir}")
            test_data_dir.mkdir(parents=True, exist_ok=True)
        
        return True
    
    def _get_scenarios(self) -> List:
        """Get test scenarios to run"""
        if self.args.scenario:
            # Run specific scenario
            return [IntegrationTestConfig.get_scenario(self.args.scenario)]
        elif self.args.all_scenarios:
            # Run all scenarios
            return IntegrationTestConfig.SCENARIOS
        else:
            # Default: run happy path scenario
            return [IntegrationTestConfig.get_scenario("happy_path")]
    
    def _run_scenario(self, scenario) -> Dict[str, Any]:
        """Run a specific test scenario"""
        scenario_result = {
            "name": scenario.name,
            "description": scenario.description,
            "start_time": datetime.now().isoformat(),
            "tests": {},
            "metrics": {}
        }
        
        # Set up environment for scenario
        self._setup_scenario_environment(scenario)
        
        # Determine which test modules to run
        test_modules = self._get_test_modules()
        
        # Run pytest for each module
        for module in test_modules:
            self.logger.info(f"Running tests in {module}")
            
            pytest_args = [
                module,
                "-v",
                f"--scenario={scenario.name}",
                "--tb=short"
            ]
            
            if self.args.debug:
                pytest_args.append("-s")
            
            if self.args.performance:
                pytest_args.append("--performance")
            
            if self.args.markers:
                pytest_args.extend(["-m", self.args.markers])
            
            # Run tests
            exit_code = pytest.main(pytest_args)
            
            scenario_result["tests"][module] = {
                "exit_code": exit_code,
                "passed": exit_code == 0
            }
        
        # Collect metrics
        scenario_result["metrics"] = self._collect_scenario_metrics(scenario)
        scenario_result["end_time"] = datetime.now().isoformat()
        
        return scenario_result
    
    def _setup_scenario_environment(self, scenario):
        """Set up environment variables for scenario"""
        # Set scenario-specific environment variables
        os.environ["TEST_SCENARIO"] = scenario.name
        os.environ["TEST_API_VERSION"] = scenario.api_configs.get("version", "v1")
        
        # Set error injection parameters
        for key, value in scenario.error_injection.items():
            env_key = f"TEST_ERROR_{key.upper()}"
            os.environ[env_key] = str(value)
    
    def _get_test_modules(self) -> List[str]:
        """Get test modules to run"""
        if self.args.test_module:
            return [self.args.test_module]
        
        # Default: run all integration test modules
        test_dir = Path("tests/integration")
        modules = []
        
        for test_file in test_dir.glob("test_*.py"):
            if test_file.name != "test_config.py":
                modules.append(str(test_file))
        
        return modules
    
    def _collect_scenario_metrics(self, scenario) -> Dict[str, Any]:
        """Collect metrics for a scenario"""
        metrics = {
            "total_api_calls": 0,
            "failed_api_calls": 0,
            "average_response_time_ms": 0,
            "memory_usage_mb": 0,
            "errors_encountered": []
        }
        
        # TODO: Implement metric collection from test results
        # This would typically read from test output files or monitoring systems
        
        return metrics
    
    def _generate_summary(self):
        """Generate test summary"""
        total_scenarios = len(self.results["scenarios"])
        passed_scenarios = sum(
            1 for s in self.results["scenarios"].values()
            if all(t["passed"] for t in s["tests"].values())
        )
        
        self.results["summary"] = {
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "failed_scenarios": total_scenarios - passed_scenarios,
            "end_time": datetime.now().isoformat()
        }
    
    def _save_results(self):
        """Save test results to file"""
        results_dir = Path("tests/integration/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"integration_results_{timestamp}.json"
        
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _print_summary(self):
        """Print test summary to console"""
        summary = self.results["summary"]
        
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Total Scenarios: {summary['total_scenarios']}")
        print(f"Passed: {summary['passed_scenarios']}")
        print(f"Failed: {summary['failed_scenarios']}")
        print("="*60)
        
        # Print details for each scenario
        for name, result in self.results["scenarios"].items():
            status = "PASSED" if all(t["passed"] for t in result["tests"].values()) else "FAILED"
            print(f"\nScenario: {name} - {status}")
            print(f"Description: {result['description']}")
            
            for module, test_result in result["tests"].items():
                test_status = "PASSED" if test_result["passed"] else "FAILED"
                print(f"  {os.path.basename(module)}: {test_status}")
    
    def _all_tests_passed(self) -> bool:
        """Check if all tests passed"""
        return self.results["summary"]["failed_scenarios"] == 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run integration tests for OpenWebUI Adaptive Memory Plugin"
    )
    
    # Test selection
    parser.add_argument(
        "--scenario",
        choices=[s.name for s in IntegrationTestConfig.SCENARIOS],
        help="Run a specific test scenario"
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run all test scenarios"
    )
    parser.add_argument(
        "--test-module",
        help="Run a specific test module"
    )
    
    # Test configuration
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance tests"
    )
    parser.add_argument(
        "--markers",
        help="Pytest markers to filter tests (e.g., 'slow', 'api')"
    )
    
    # Output options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save test results to file"
    )
    
    args = parser.parse_args()
    
    # Run tests
    runner = IntegrationTestRunner(args)
    exit_code = runner.run()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()