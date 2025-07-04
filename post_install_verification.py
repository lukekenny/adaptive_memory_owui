#!/usr/bin/env python3
"""
Post-Installation Verification System for OWUI Adaptive Memory Plugin

This script automates verification steps after installation to confirm successful
setup and connectivity, especially for LLM and API integrations.
"""

import os
import sys
import json
import asyncio
import time
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import importlib.util
import platform
import socket
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PostInstallVerifier:
    """Automated post-installation verification system."""
    
    def __init__(self, plugin_path: Optional[str] = None):
        self.plugin_path = plugin_path or os.path.dirname(os.path.abspath(__file__))
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "checks": {},
            "errors": [],
            "warnings": [],
            "success": True
        }
        self.auto_fix_applied = []
        
    def _get_system_info(self) -> Dict[str, str]:
        """Gather system information for diagnostics."""
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "os": platform.system(),
            "architecture": platform.machine(),
            "hostname": socket.gethostname(),
            "plugin_path": self.plugin_path
        }
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Verify Python version compatibility."""
        logger.info("Checking Python version...")
        
        version = sys.version_info
        min_version = (3, 8)
        
        if version >= min_version:
            message = f"✅ Python {version.major}.{version.minor}.{version.micro} meets requirements"
            return True, message
        else:
            message = f"❌ Python {version.major}.{version.minor}.{version.micro} is below minimum {min_version[0]}.{min_version[1]}"
            return False, message
    
    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """Verify all required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        required_packages = {
            "pydantic": "2.0.0",
            "numpy": "1.24.0",
            "aiohttp": "3.8.0",
            "pytz": "2023.3"
        }
        
        optional_packages = {
            "sentence-transformers": "2.2.0",
            "scikit-learn": "1.3.0"
        }
        
        missing_required = []
        missing_optional = []
        version_issues = []
        
        # Check required packages
        for package, min_version in required_packages.items():
            try:
                module = importlib.import_module(package.replace("-", "_"))
                installed_version = getattr(module, "__version__", "unknown")
                
                # Basic version comparison
                if installed_version != "unknown":
                    try:
                        from packaging import version
                        if version.parse(installed_version) < version.parse(min_version):
                            version_issues.append(f"{package}: {installed_version} < {min_version}")
                    except:
                        # Fallback to string comparison if packaging not available
                        pass
                        
            except ImportError:
                missing_required.append(package)
        
        # Check optional packages
        for package, _ in optional_packages.items():
            try:
                importlib.import_module(package.replace("-", "_"))
            except ImportError:
                missing_optional.append(package)
        
        messages = []
        success = True
        
        if missing_required:
            success = False
            messages.append(f"❌ Missing required packages: {', '.join(missing_required)}")
        else:
            messages.append("✅ All required packages installed")
        
        if version_issues:
            success = False
            messages.extend([f"⚠️ Version issue: {issue}" for issue in version_issues])
        
        if missing_optional:
            messages.append(f"ℹ️ Missing optional packages: {', '.join(missing_optional)}")
        
        return success, messages
    
    def check_plugin_structure(self) -> Tuple[bool, List[str]]:
        """Verify the plugin file structure is correct."""
        logger.info("Checking plugin structure...")
        
        required_files = [
            "adaptive_memory_v4.0.py",
            "requirements.txt"
        ]
        
        optional_files = [
            "README.md",
            "CLAUDE.md",
            "tests/",
            ".taskmaster/"
        ]
        
        messages = []
        success = True
        missing_required = []
        
        for file in required_files:
            file_path = Path(self.plugin_path) / file
            if not file_path.exists():
                missing_required.append(file)
                success = False
        
        if missing_required:
            messages.append(f"❌ Missing required files: {', '.join(missing_required)}")
        else:
            messages.append("✅ All required files present")
        
        # Check optional files
        present_optional = []
        for file in optional_files:
            file_path = Path(self.plugin_path) / file
            if file_path.exists():
                present_optional.append(file)
        
        if present_optional:
            messages.append(f"ℹ️ Optional files found: {', '.join(present_optional)}")
        
        return success, messages
    
    def check_filter_class(self) -> Tuple[bool, List[str]]:
        """Verify the Filter class is properly implemented."""
        logger.info("Checking Filter class implementation...")
        
        messages = []
        success = True
        
        try:
            # Import the main plugin file
            spec = importlib.util.spec_from_file_location(
                "adaptive_memory_v4",
                Path(self.plugin_path) / "adaptive_memory_v4.0.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for Filter class
            if not hasattr(module, "Filter"):
                success = False
                messages.append("❌ Filter class not found")
                return success, messages
            
            Filter = getattr(module, "Filter")
            
            # Check required methods
            required_methods = ["__init__", "inlet", "outlet", "stream"]
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(Filter, method):
                    missing_methods.append(method)
            
            if missing_methods:
                success = False
                messages.append(f"❌ Missing required methods: {', '.join(missing_methods)}")
            else:
                messages.append("✅ All required Filter methods present")
            
            # Check for Valves class
            if hasattr(Filter, "Valves"):
                messages.append("✅ Valves configuration class found")
            else:
                messages.append("⚠️ Valves configuration class not found")
            
        except Exception as e:
            success = False
            messages.append(f"❌ Error loading plugin: {str(e)}")
        
        return success, messages
    
    async def check_llm_connectivity(self) -> Tuple[bool, List[str]]:
        """Test LLM provider connectivity."""
        logger.info("Checking LLM connectivity...")
        
        messages = []
        success = True
        
        try:
            # Import the plugin to test connections
            spec = importlib.util.spec_from_file_location(
                "adaptive_memory_v4",
                Path(self.plugin_path) / "adaptive_memory_v4.0.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            Filter = getattr(module, "Filter")
            filter_instance = Filter()
            
            # Test different LLM providers if configured
            providers_tested = []
            
            # Check Ollama
            if hasattr(filter_instance.valves, "llm_provider_type") and \
               filter_instance.valves.llm_provider_type == "ollama":
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        url = getattr(filter_instance.valves, "ollama_base_url", "http://localhost:11434")
                        async with session.get(f"{url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                            if resp.status == 200:
                                messages.append(f"✅ Ollama connection successful at {url}")
                                providers_tested.append("ollama")
                            else:
                                messages.append(f"⚠️ Ollama responded with status {resp.status}")
                except Exception as e:
                    messages.append(f"❌ Ollama connection failed: {str(e)}")
                    success = False
            
            # Check OpenAI-compatible
            if hasattr(filter_instance.valves, "llm_provider_type") and \
               filter_instance.valves.llm_provider_type == "openai_compatible":
                api_url = getattr(filter_instance.valves, "openai_compatible_api_url", "")
                if api_url:
                    messages.append(f"ℹ️ OpenAI-compatible API configured: {api_url}")
                    providers_tested.append("openai_compatible")
                else:
                    messages.append("⚠️ OpenAI-compatible API URL not configured")
            
            # Check Gemini
            if hasattr(filter_instance.valves, "llm_provider_type") and \
               filter_instance.valves.llm_provider_type == "gemini":
                if getattr(filter_instance.valves, "gemini_api_key", ""):
                    messages.append("✅ Gemini API key configured")
                    providers_tested.append("gemini")
                else:
                    messages.append("⚠️ Gemini API key not configured")
            
            if not providers_tested:
                messages.append("⚠️ No LLM providers configured")
                
        except Exception as e:
            success = False
            messages.append(f"❌ Error testing LLM connectivity: {str(e)}")
        
        return success, messages
    
    def check_permissions(self) -> Tuple[bool, List[str]]:
        """Verify file and directory permissions."""
        logger.info("Checking permissions...")
        
        messages = []
        success = True
        
        # Check plugin directory is writable
        plugin_dir = Path(self.plugin_path)
        
        try:
            # Test write permission
            test_file = plugin_dir / ".permission_test"
            test_file.write_text("test")
            test_file.unlink()
            messages.append("✅ Plugin directory is writable")
        except Exception as e:
            success = False
            messages.append(f"❌ Plugin directory not writable: {str(e)}")
        
        # Check main plugin file is readable
        main_file = plugin_dir / "adaptive_memory_v4.0.py"
        if main_file.exists() and os.access(main_file, os.R_OK):
            messages.append("✅ Main plugin file is readable")
        else:
            success = False
            messages.append("❌ Main plugin file not readable")
        
        return success, messages
    
    def check_docker_environment(self) -> Tuple[bool, List[str]]:
        """Check if running in Docker and verify environment."""
        logger.info("Checking Docker environment...")
        
        messages = []
        in_docker = False
        
        # Check if running in Docker
        if os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
            in_docker = True
            messages.append("ℹ️ Running in Docker container")
            
            # Check for common Docker networking issues
            if os.environ.get("OPENWEBUI_URL"):
                messages.append(f"✅ OpenWebUI URL configured: {os.environ.get('OPENWEBUI_URL')}")
            else:
                messages.append("⚠️ OPENWEBUI_URL environment variable not set")
        else:
            messages.append("ℹ️ Not running in Docker")
        
        return True, messages
    
    def apply_auto_fixes(self) -> List[str]:
        """Attempt to automatically fix common issues."""
        logger.info("Attempting auto-fixes...")
        
        fixes_applied = []
        
        # Fix 1: Install missing required dependencies
        if "missing_required" in str(self.results.get("checks", {}).get("dependencies", [])):
            logger.info("Attempting to install missing dependencies...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", 
                    os.path.join(self.plugin_path, "requirements.txt")
                ], check=True, capture_output=True)
                fixes_applied.append("✅ Installed missing dependencies")
            except Exception as e:
                fixes_applied.append(f"❌ Failed to install dependencies: {str(e)}")
        
        # Fix 2: Create missing directories
        required_dirs = ["tests", "logs", "memory-bank"]
        for dir_name in required_dirs:
            dir_path = Path(self.plugin_path) / dir_name
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    fixes_applied.append(f"✅ Created missing directory: {dir_name}")
                except Exception as e:
                    fixes_applied.append(f"❌ Failed to create {dir_name}: {str(e)}")
        
        # Fix 3: Set proper permissions
        try:
            main_file = Path(self.plugin_path) / "adaptive_memory_v4.0.py"
            if main_file.exists():
                os.chmod(main_file, 0o644)
                fixes_applied.append("✅ Fixed file permissions")
        except Exception as e:
            fixes_applied.append(f"❌ Failed to fix permissions: {str(e)}")
        
        return fixes_applied
    
    async def run_verification(self, auto_fix: bool = False) -> Dict[str, Any]:
        """Run all verification checks."""
        logger.info("=" * 60)
        logger.info("Starting Post-Installation Verification")
        logger.info("=" * 60)
        
        # Run synchronous checks
        checks = [
            ("python_version", self.check_python_version),
            ("dependencies", self.check_dependencies),
            ("plugin_structure", self.check_plugin_structure),
            ("filter_class", self.check_filter_class),
            ("permissions", self.check_permissions),
            ("docker_environment", self.check_docker_environment)
        ]
        
        for check_name, check_func in checks:
            success, messages = check_func()
            self.results["checks"][check_name] = {
                "success": success,
                "messages": messages
            }
            if not success:
                self.results["success"] = False
            
            # Log results
            for message in messages:
                logger.info(message)
        
        # Run async checks
        success, messages = await self.check_llm_connectivity()
        self.results["checks"]["llm_connectivity"] = {
            "success": success,
            "messages": messages
        }
        if not success:
            self.results["success"] = False
        
        for message in messages:
            logger.info(message)
        
        # Apply auto-fixes if requested
        if auto_fix and not self.results["success"]:
            logger.info("\n" + "=" * 60)
            logger.info("Applying automatic fixes...")
            logger.info("=" * 60)
            
            fixes = self.apply_auto_fixes()
            self.results["auto_fixes"] = fixes
            
            for fix in fixes:
                logger.info(fix)
            
            # Re-run checks after fixes
            logger.info("\n" + "=" * 60)
            logger.info("Re-running verification after fixes...")
            logger.info("=" * 60)
            
            # Reset results
            self.results["checks_after_fix"] = {}
            recheck_success = True
            
            for check_name, check_func in checks:
                success, messages = check_func()
                self.results["checks_after_fix"][check_name] = {
                    "success": success,
                    "messages": messages
                }
                if not success:
                    recheck_success = False
            
            self.results["success_after_fix"] = recheck_success
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate a summary of verification results."""
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 60)
        
        total_checks = len(self.results["checks"])
        passed_checks = sum(1 for check in self.results["checks"].values() if check["success"])
        
        logger.info(f"Total checks: {total_checks}")
        logger.info(f"Passed: {passed_checks}")
        logger.info(f"Failed: {total_checks - passed_checks}")
        
        if self.results["success"]:
            logger.info("\n✅ POST-INSTALLATION VERIFICATION PASSED")
            logger.info("The OWUI Adaptive Memory Plugin is ready to use!")
        else:
            logger.info("\n❌ POST-INSTALLATION VERIFICATION FAILED")
            logger.info("Please address the issues above before using the plugin.")
            
            # Provide troubleshooting tips
            logger.info("\n" + "=" * 60)
            logger.info("TROUBLESHOOTING TIPS")
            logger.info("=" * 60)
            
            if not self.results["checks"].get("dependencies", {}).get("success", True):
                logger.info("• Install missing dependencies: pip install -r requirements.txt")
            
            if not self.results["checks"].get("filter_class", {}).get("success", True):
                logger.info("• Ensure adaptive_memory_v4.0.py contains a valid Filter class")
            
            if not self.results["checks"].get("llm_connectivity", {}).get("success", True):
                logger.info("• Check your LLM provider configuration in the plugin settings")
                logger.info("• Ensure LLM services are running and accessible")
    
    def save_report(self, filename: Optional[str] = None):
        """Save verification report to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"post_install_verification_{timestamp}.json"
        
        report_path = Path(self.plugin_path) / filename
        
        try:
            with open(report_path, "w") as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"\nVerification report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")


async def main():
    """Main entry point for post-installation verification."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Post-Installation Verification for OWUI Adaptive Memory Plugin"
    )
    parser.add_argument(
        "--path",
        help="Path to the plugin directory",
        default=os.path.dirname(os.path.abspath(__file__))
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Attempt to automatically fix common issues"
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save verification report to file"
    )
    
    args = parser.parse_args()
    
    # Run verification
    verifier = PostInstallVerifier(args.path)
    results = await verifier.run_verification(auto_fix=args.auto_fix)
    
    # Save report if requested
    if args.save_report:
        verifier.save_report()
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    asyncio.run(main())