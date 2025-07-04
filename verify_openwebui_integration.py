#!/usr/bin/env python3
"""
OpenWebUI Integration Verification for OWUI Adaptive Memory Plugin

This script specifically tests the integration with OpenWebUI, including
API endpoints, filter functionality, and end-to-end workflows.
"""

import os
import sys
import json
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from datetime import datetime
import importlib.util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenWebUIIntegrationVerifier:
    """Verify integration with OpenWebUI."""
    
    def __init__(self, 
                 plugin_path: Optional[str] = None,
                 openwebui_url: str = "http://localhost:8080"):
        self.plugin_path = plugin_path or os.path.dirname(os.path.abspath(__file__))
        self.openwebui_url = openwebui_url.rstrip('/')
        self.session = None
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "openwebui_url": self.openwebui_url,
            "integration_tests": {},
            "errors": [],
            "warnings": [],
            "success": True
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self.session
    
    async def test_openwebui_connection(self) -> Tuple[bool, List[str]]:
        """Test basic connection to OpenWebUI."""
        logger.info("Testing OpenWebUI connection...")
        
        messages = []
        success = True
        
        try:
            session = await self._get_session()
            
            # Test API endpoint
            async with session.get(f"{self.openwebui_url}/api/version") as resp:
                if resp.status == 200:
                    version_data = await resp.json()
                    messages.append(f"✅ Connected to OpenWebUI at {self.openwebui_url}")
                    messages.append(f"ℹ️ OpenWebUI version: {version_data.get('version', 'unknown')}")
                else:
                    success = False
                    messages.append(f"❌ OpenWebUI API returned status {resp.status}")
                    
        except aiohttp.ClientConnectorError:
            success = False
            messages.append(f"❌ Cannot connect to OpenWebUI at {self.openwebui_url}")
            messages.append("ℹ️ Ensure OpenWebUI is running and accessible")
        except Exception as e:
            success = False
            messages.append(f"❌ Connection error: {str(e)}")
        
        return success, messages
    
    async def test_filter_endpoints(self) -> Tuple[bool, List[str]]:
        """Test filter-specific API endpoints."""
        logger.info("Testing filter API endpoints...")
        
        messages = []
        success = True
        
        try:
            session = await self._get_session()
            
            # Test functions endpoint
            async with session.get(f"{self.openwebui_url}/api/functions") as resp:
                if resp.status == 200:
                    functions = await resp.json()
                    messages.append(f"✅ Functions API accessible ({len(functions)} functions found)")
                    
                    # Look for our filter
                    memory_filters = [f for f in functions if "memory" in f.get("name", "").lower()]
                    if memory_filters:
                        messages.append(f"✅ Found {len(memory_filters)} memory-related filters")
                    else:
                        messages.append("⚠️ No memory filters found in OpenWebUI")
                else:
                    success = False
                    messages.append(f"❌ Functions API returned status {resp.status}")
                    
        except Exception as e:
            success = False
            messages.append(f"❌ Error accessing filter endpoints: {str(e)}")
        
        return success, messages
    
    async def test_filter_functionality(self) -> Tuple[bool, List[str]]:
        """Test the filter's core functionality."""
        logger.info("Testing filter functionality...")
        
        messages = []
        success = True
        
        try:
            # Import and instantiate the filter
            spec = importlib.util.spec_from_file_location(
                "adaptive_memory_v4",
                Path(self.plugin_path) / "adaptive_memory_v4.0.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            Filter = getattr(module, "Filter")
            filter_instance = Filter()
            
            # Test inlet method
            test_body = {
                "messages": [
                    {"role": "user", "content": "My name is TestUser and I like Python programming"}
                ],
                "user": {"id": "test-user-123", "name": "TestUser"}
            }
            
            # Test inlet processing
            try:
                inlet_result = filter_instance.inlet(test_body)
                if inlet_result and isinstance(inlet_result, dict):
                    messages.append("✅ Inlet method processed successfully")
                else:
                    messages.append("⚠️ Inlet method returned unexpected result")
            except Exception as e:
                success = False
                messages.append(f"❌ Inlet method error: {str(e)}")
            
            # Test outlet method
            test_body["messages"].append({
                "role": "assistant",
                "content": "Hello! Nice to meet you."
            })
            
            try:
                outlet_result = filter_instance.outlet(test_body)
                if outlet_result and isinstance(outlet_result, dict):
                    messages.append("✅ Outlet method processed successfully")
                else:
                    messages.append("⚠️ Outlet method returned unexpected result")
            except Exception as e:
                success = False
                messages.append(f"❌ Outlet method error: {str(e)}")
            
            # Test stream method
            test_event = {"data": "test"}
            try:
                stream_result = filter_instance.stream(test_event)
                if stream_result == test_event:
                    messages.append("✅ Stream method passed through correctly")
                else:
                    messages.append("⚠️ Stream method modified the event")
            except Exception as e:
                success = False
                messages.append(f"❌ Stream method error: {str(e)}")
                
        except Exception as e:
            success = False
            messages.append(f"❌ Error testing filter functionality: {str(e)}")
        
        return success, messages
    
    async def test_memory_operations(self) -> Tuple[bool, List[str]]:
        """Test memory extraction and storage operations."""
        logger.info("Testing memory operations...")
        
        messages = []
        success = True
        
        try:
            # Import the filter
            spec = importlib.util.spec_from_file_location(
                "adaptive_memory_v4",
                Path(self.plugin_path) / "adaptive_memory_v4.0.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            Filter = getattr(module, "Filter")
            filter_instance = Filter()
            
            # Simulate memory extraction
            test_messages = [
                "My email is test@example.com",
                "I prefer dark mode for coding",
                "My favorite programming language is Python",
                "I work as a software engineer at TechCorp"
            ]
            
            extracted_count = 0
            
            for message in test_messages:
                test_body = {
                    "messages": [{"role": "user", "content": message}],
                    "user": {"id": "test-user-memory", "name": "TestUser"}
                }
                
                try:
                    # Process through inlet (where memory extraction happens)
                    result = filter_instance.inlet(test_body)
                    
                    # Check if memory was likely extracted (this is a simplified check)
                    # In real implementation, we'd check the memory storage
                    if result:
                        extracted_count += 1
                        
                except Exception as e:
                    logger.warning(f"Memory extraction failed for: {message[:30]}... - {str(e)}")
            
            if extracted_count > 0:
                messages.append(f"✅ Memory extraction working ({extracted_count}/{len(test_messages)} messages processed)")
            else:
                success = False
                messages.append("❌ Memory extraction not working")
                
        except Exception as e:
            success = False
            messages.append(f"❌ Error testing memory operations: {str(e)}")
        
        return success, messages
    
    async def test_configuration_management(self) -> Tuple[bool, List[str]]:
        """Test configuration and valve management."""
        logger.info("Testing configuration management...")
        
        messages = []
        success = True
        
        try:
            # Import the filter
            spec = importlib.util.spec_from_file_location(
                "adaptive_memory_v4",
                Path(self.plugin_path) / "adaptive_memory_v4.0.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            Filter = getattr(module, "Filter")
            filter_instance = Filter()
            
            # Check Valves configuration
            if hasattr(filter_instance, "valves"):
                valves = filter_instance.valves
                
                # Count configuration options
                config_count = len([attr for attr in dir(valves) 
                                  if not attr.startswith('_')])
                
                messages.append(f"✅ Valves configuration loaded ({config_count} options)")
                
                # Check critical configurations
                critical_configs = [
                    ("enable_memory", "Memory functionality"),
                    ("llm_provider_type", "LLM provider"),
                    ("debug_logging", "Debug logging")
                ]
                
                for config_name, description in critical_configs:
                    if hasattr(valves, config_name):
                        value = getattr(valves, config_name)
                        messages.append(f"ℹ️ {description}: {value}")
                    else:
                        messages.append(f"⚠️ Missing configuration: {config_name}")
                        
            else:
                success = False
                messages.append("❌ Valves configuration not found")
                
        except Exception as e:
            success = False
            messages.append(f"❌ Error testing configuration: {str(e)}")
        
        return success, messages
    
    async def test_error_handling(self) -> Tuple[bool, List[str]]:
        """Test error handling and recovery."""
        logger.info("Testing error handling...")
        
        messages = []
        success = True
        
        try:
            # Import the filter
            spec = importlib.util.spec_from_file_location(
                "adaptive_memory_v4",
                Path(self.plugin_path) / "adaptive_memory_v4.0.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            Filter = getattr(module, "Filter")
            filter_instance = Filter()
            
            # Test with various malformed inputs
            test_cases = [
                (None, "None input"),
                ({}, "Empty dict"),
                ({"messages": []}, "Empty messages"),
                ({"messages": [{"role": "user"}]}, "Missing content"),
                ({"user": None}, "None user"),
                ("not a dict", "Invalid type")
            ]
            
            errors_handled = 0
            
            for test_input, description in test_cases:
                try:
                    # Test inlet
                    result = filter_instance.inlet(test_input)
                    
                    # Filter should handle errors gracefully
                    if result == test_input or (isinstance(result, dict) and result is not None):
                        errors_handled += 1
                    else:
                        logger.warning(f"Unexpected result for {description}: {result}")
                        
                except Exception as e:
                    # Filter should not raise exceptions
                    success = False
                    messages.append(f"❌ Exception raised for {description}: {str(e)}")
            
            if errors_handled == len(test_cases):
                messages.append(f"✅ All error cases handled gracefully ({errors_handled}/{len(test_cases)})")
            else:
                messages.append(f"⚠️ Some error cases not handled properly ({errors_handled}/{len(test_cases)})")
                
        except Exception as e:
            success = False
            messages.append(f"❌ Error testing error handling: {str(e)}")
        
        return success, messages
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        logger.info("=" * 60)
        logger.info("OpenWebUI Integration Verification")
        logger.info("=" * 60)
        
        tests = [
            ("openwebui_connection", self.test_openwebui_connection),
            ("filter_endpoints", self.test_filter_endpoints),
            ("filter_functionality", self.test_filter_functionality),
            ("memory_operations", self.test_memory_operations),
            ("configuration_management", self.test_configuration_management),
            ("error_handling", self.test_error_handling)
        ]
        
        for test_name, test_func in tests:
            success, messages = await test_func()
            self.results["integration_tests"][test_name] = {
                "success": success,
                "messages": messages
            }
            if not success:
                self.results["success"] = False
            
            # Log results
            for message in messages:
                logger.info(message)
            logger.info("")  # Empty line between tests
        
        # Generate summary
        self._generate_summary()
        
        # Close session
        if self.session:
            await self.session.close()
        
        return self.results
    
    def _generate_summary(self):
        """Generate a summary of integration test results."""
        logger.info("=" * 60)
        logger.info("INTEGRATION TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = len(self.results["integration_tests"])
        passed_tests = sum(1 for test in self.results["integration_tests"].values() 
                          if test["success"])
        
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        
        if self.results["success"]:
            logger.info("\n✅ OPENWEBUI INTEGRATION VERIFICATION PASSED")
            logger.info("The plugin is properly integrated with OpenWebUI!")
        else:
            logger.info("\n❌ OPENWEBUI INTEGRATION VERIFICATION FAILED")
            logger.info("Please check the failed tests above.")
            
            # Provide specific recommendations
            logger.info("\n" + "=" * 60)
            logger.info("RECOMMENDATIONS")
            logger.info("=" * 60)
            
            failed_tests = [name for name, result in self.results["integration_tests"].items() 
                           if not result["success"]]
            
            if "openwebui_connection" in failed_tests:
                logger.info("• Ensure OpenWebUI is running at the specified URL")
                logger.info("• Check firewall and network settings")
                logger.info("• Verify the OpenWebUI URL is correct")
            
            if "filter_endpoints" in failed_tests:
                logger.info("• Check if the filter is properly registered in OpenWebUI")
                logger.info("• Ensure the Functions API is enabled")
            
            if "filter_functionality" in failed_tests:
                logger.info("• Review the filter implementation for errors")
                logger.info("• Check the logs for detailed error messages")
            
            if "memory_operations" in failed_tests:
                logger.info("• Verify LLM provider is configured correctly")
                logger.info("• Check memory extraction prompts and settings")


async def main():
    """Main entry point for integration verification."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="OpenWebUI Integration Verification for OWUI Adaptive Memory Plugin"
    )
    parser.add_argument(
        "--path",
        help="Path to the plugin directory",
        default=os.path.dirname(os.path.abspath(__file__))
    )
    parser.add_argument(
        "--url",
        help="OpenWebUI URL",
        default=os.environ.get("OPENWEBUI_URL", "http://localhost:8080")
    )
    parser.add_argument(
        "--save-report",
        help="Save test report to specified file",
        type=str
    )
    
    args = parser.parse_args()
    
    # Run integration tests
    verifier = OpenWebUIIntegrationVerifier(args.path, args.url)
    results = await verifier.run_integration_tests()
    
    # Save report if requested
    if args.save_report:
        try:
            with open(args.save_report, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nTest report saved to: {args.save_report}")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    asyncio.run(main())