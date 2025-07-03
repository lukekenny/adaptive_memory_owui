#!/usr/bin/env python3
"""
OpenWebUI Adaptive Memory Plugin - Installation Diagnostics Tool

This script helps diagnose and fix installation issues for the OWUI Adaptive Memory Plugin.
Run this script to identify and resolve common installation problems.
"""

import sys
import os
import json
import traceback
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import tempfile

class InstallationDiagnostics:
    """Comprehensive installation diagnostics for OWUI Adaptive Memory Plugin."""
    
    def __init__(self):
        self.issues_found = []
        self.fixes_applied = []
        self.warnings = []
        
    def log_issue(self, issue: str, severity: str = "ERROR"):
        """Log an installation issue."""
        self.issues_found.append(f"[{severity}] {issue}")
        print(f"🔴 [{severity}] {issue}")
        
    def log_fix(self, fix: str):
        """Log a fix that was applied."""
        self.fixes_applied.append(fix)
        print(f"✅ FIXED: {fix}")
        
    def log_warning(self, warning: str):
        """Log a warning."""
        self.warnings.append(warning)
        print(f"⚠️  WARNING: {warning}")
        
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        print("\n📋 Checking Python version...")
        version = sys.version_info
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.log_issue(f"Python {version.major}.{version.minor} is not supported. Requires Python 3.8+")
            return False
        
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies can be imported."""
        print("\n📋 Checking dependencies...")
        
        required_packages = [
            'pydantic',
            'typing',
            'json',
            'datetime',
            'logging',
            'asyncio',
            'aiohttp',
            'numpy',
            'sentence_transformers',
            'scikit-learn',
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} - Available")
            except ImportError as e:
                missing_packages.append(package)
                print(f"🔴 {package} - Missing: {e}")
        
        if missing_packages:
            self.log_issue(f"Missing required packages: {', '.join(missing_packages)}")
            return False
            
        return True
        
    def install_missing_dependencies(self) -> bool:
        """Install missing dependencies."""
        requirements_path = Path(__file__).parent / "requirements.txt"
        
        if not requirements_path.exists():
            self.log_issue("requirements.txt not found")
            return False
            
        print("\n📦 Installing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
            ])
            self.log_fix("Successfully installed dependencies from requirements.txt")
            return True
        except subprocess.CalledProcessError as e:
            self.log_issue(f"Failed to install dependencies: {e}")
            return False
            
    def validate_filter_structure(self, filter_path: Path) -> bool:
        """Validate the Filter class structure."""
        print(f"\n📋 Validating filter structure: {filter_path.name}")
        
        try:
            # Read the filter file
            content = filter_path.read_text(encoding='utf-8')
            
            # Check file size
            size_mb = len(content) / (1024 * 1024)
            if size_mb > 10:
                self.log_warning(f"Filter file is very large ({size_mb:.1f}MB). This may cause loading issues.")
            
            # Check for required class
            if "class Filter:" not in content:
                self.log_issue("Filter class not found in file")
                return False
            
            # Check for required methods (allowing both sync and async)
            required_patterns = [
                ("Valves class", "class Valves("),
                ("__init__ method", "def __init__("),
                ("inlet method", ("def inlet(", "async def inlet(")),
                ("outlet method", ("def outlet(", "async def outlet(")),
            ]
            
            for name, pattern in required_patterns:
                if isinstance(pattern, tuple):
                    # Multiple patterns (sync or async)
                    found = any(p in content for p in pattern)
                else:
                    found = pattern in content
                    
                if found:
                    print(f"✅ {name} - Found")
                else:
                    self.log_issue(f"{name} not found")
                    return False
            
            # Check for import issues
            problematic_imports = [
                "from open_webui.",  # May not be available during validation
            ]
            
            for imp in problematic_imports:
                if imp in content:
                    self.log_warning(f"Potentially problematic import found: {imp}")
            
            return True
            
        except Exception as e:
            self.log_issue(f"Failed to validate filter structure: {e}")
            return False
            
    def test_filter_loading(self, filter_path: Path) -> bool:
        """Test if the filter can be loaded without errors."""
        print(f"\n📋 Testing filter loading: {filter_path.name}")
        
        try:
            # Create a temporary test script
            test_script = f"""
import sys
import os
sys.path.insert(0, '{filter_path.parent}')

# Mock OpenWebUI imports that may not be available
class MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockField:
    def __init__(self, default=None, **kwargs):
        self.default = default
        
sys.modules['pydantic'] = type('MockPydantic', (), {{'BaseModel': MockBaseModel, 'Field': MockField}})()

# Try to import the filter
spec = importlib.util.spec_from_file_location("filter_module", "{filter_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Check if Filter class exists
if hasattr(module, 'Filter'):
    filter_instance = module.Filter()
    print("SUCCESS: Filter class loaded and instantiated")
else:
    print("ERROR: Filter class not found in module")
"""
            
            # Write and execute test script
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_path = f.name
            
            try:
                result = subprocess.run([
                    sys.executable, temp_path
                ], capture_output=True, text=True, timeout=30)
                
                if "SUCCESS" in result.stdout:
                    print("✅ Filter loads successfully")
                    return True
                else:
                    self.log_issue(f"Filter loading test failed: {result.stdout} {result.stderr}")
                    return False
                    
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            self.log_issue(f"Filter loading test error: {e}")
            return False
            
    def create_minimal_filter(self) -> Path:
        """Create a minimal working filter for testing."""
        print("\n🔧 Creating minimal filter for testing...")
        
        minimal_filter = '''"""
OpenWebUI Adaptive Memory Plugin - Minimal Test Version
This is a simplified version for testing installation and basic functionality.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Filter:
    """
    Minimal Adaptive Memory Filter for OpenWebUI
    
    This simplified version tests basic installation and functionality.
    """
    
    class Valves(BaseModel):
        """Configuration options for the filter."""
        
        enable_memory: bool = Field(
            default=True,
            description="Enable memory functionality"
        )
        test_mode: bool = Field(
            default=True,
            description="Run in test mode with minimal processing"
        )
        debug_logging: bool = Field(
            default=True,
            description="Enable debug logging"
        )
    
    def __init__(self):
        """Initialize the filter."""
        try:
            self.valves = self.Valves()
            logger.info("Minimal Adaptive Memory Filter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize filter: {e}")
            # Create default valves if initialization fails
            self.valves = type('DefaultValves', (), {
                'enable_memory': True,
                'test_mode': True,
                'debug_logging': True
            })()
    
    def inlet(self, body: dict) -> dict:
        """Process user input."""
        try:
            if self.valves.debug_logging:
                logger.info("Inlet called - processing user input")
            
            if not self.valves.enable_memory:
                return body
            
            # Simple test processing - just log the input
            if isinstance(body, dict) and "messages" in body:
                logger.info(f"Processing {len(body['messages'])} messages")
            
            return body
            
        except Exception as e:
            logger.error(f"Error in inlet: {e}")
            return body
    
    def outlet(self, body: dict) -> dict:
        """Process model output."""
        try:
            if self.valves.debug_logging:
                logger.info("Outlet called - processing model output")
            
            if not self.valves.enable_memory:
                return body
            
            # Simple test processing - just log the output
            if isinstance(body, dict) and "messages" in body:
                logger.info(f"Processing {len(body['messages'])} messages in outlet")
            
            return body
            
        except Exception as e:
            logger.error(f"Error in outlet: {e}")
            return body
    
    def stream(self, event: dict) -> dict:
        """Process streaming events."""
        try:
            if self.valves.debug_logging:
                logger.debug("Stream event processed")
            
            return event
            
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            return event
'''
        
        minimal_path = Path(__file__).parent / "adaptive_memory_minimal.py"
        
        try:
            minimal_path.write_text(minimal_filter, encoding='utf-8')
            self.log_fix(f"Created minimal filter: {minimal_path}")
            return minimal_path
        except Exception as e:
            self.log_issue(f"Failed to create minimal filter: {e}")
            raise
            
    def create_installation_guide(self) -> bool:
        """Create an installation guide with troubleshooting steps."""
        print("\n📖 Creating installation guide...")
        
        guide_content = """# OWUI Adaptive Memory Plugin - Installation Guide

## Quick Installation Steps

### 1. Check Requirements
- Python 3.8+
- OpenWebUI installed and running
- Admin access to OpenWebUI

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install the Filter
1. Open OpenWebUI admin panel
2. Go to "Workspace" → "Functions"  
3. Click "+" to add new function
4. Upload the filter file OR copy/paste the content
5. Click "Save" to install
6. Enable the filter for your desired models

### 4. Troubleshooting Common Issues

#### "No Tools class found" Error
This error is misleading. The real causes are usually:
- File too large (try the minimal version first)
- Missing dependencies 
- Import errors during loading
- OpenWebUI version compatibility

**Solutions:**
1. Try the minimal filter first (`adaptive_memory_minimal.py`)
2. Check OpenWebUI logs for actual error details
3. Ensure all dependencies are installed
4. Use the synchronous version if async fails

#### Filter Not Loading
1. Check OpenWebUI logs for detailed errors
2. Verify file is valid Python syntax
3. Test with minimal version first
4. Check file permissions

#### Memory Not Working
1. Enable debug logging in filter valves
2. Check OpenWebUI logs for filter execution
3. Verify filter is enabled for the correct model
4. Test with simple preference statements

#### Dependencies Missing
```bash
# Install core dependencies
pip install sentence-transformers pydantic aiohttp numpy scikit-learn

# For development/testing
pip install pytest pytest-asyncio pytest-mock
```

### 5. Testing Installation

After installation, test with these simple prompts:
- "My favorite color is blue"
- "I live in New York"  
- "I work as a software engineer"

Check the logs to see if the filter processes these inputs.

### 6. Advanced Configuration

Once basic installation works, you can:
- Configure LLM providers in filter valves
- Adjust memory thresholds
- Enable advanced features
- Switch to the full version

### 7. Getting Help

If you encounter issues:
1. Check OpenWebUI logs (detailed error information)
2. Run the diagnostic script: `python install_diagnostics.py`
3. Try the minimal version first
4. Report issues with full log details
"""
        
        guide_path = Path(__file__).parent / "INSTALLATION_GUIDE.md"
        
        try:
            guide_path.write_text(guide_content, encoding='utf-8')
            self.log_fix(f"Created installation guide: {guide_path}")
            return True
        except Exception as e:
            self.log_issue(f"Failed to create installation guide: {e}")
            return False
            
    def create_sync_version(self) -> bool:
        """Create a synchronous version of the filter."""
        print("\n🔧 Creating synchronous version of the filter...")
        
        v4_path = Path(__file__).parent / "adaptive_memory_v4.0.py"
        sync_path = Path(__file__).parent / "adaptive_memory_v4.0_sync.py"
        
        if not v4_path.exists():
            self.log_issue("adaptive_memory_v4.0.py not found")
            return False
        
        try:
            content = v4_path.read_text(encoding='utf-8')
            
            # Convert async methods to sync
            replacements = [
                ('async def inlet(', 'def inlet('),
                ('async def outlet(', 'def outlet('),
                ('async def stream(', 'def stream('),
                ('await ', ''),  # Remove await keywords (simplified conversion)
                ('async_inlet', 'inlet_async'),  # Rename internal async methods to avoid conflicts
                ('async_outlet', 'outlet_async'),
            ]
            
            for old, new in replacements:
                content = content.replace(old, new)
            
            # Add a note at the top
            sync_header = '''"""
SYNCHRONOUS VERSION of OpenWebUI Adaptive Memory Plugin v4.0

This version has been automatically converted to use synchronous methods
for compatibility with OpenWebUI installations that have issues with async filters.

If you're experiencing installation issues, try this version first.
"""

'''
            
            # Insert header after existing docstring
            if '"""' in content:
                first_docstring_end = content.find('"""', content.find('"""') + 3) + 3
                content = content[:first_docstring_end] + '\n\n' + sync_header + content[first_docstring_end:]
            else:
                content = sync_header + content
            
            sync_path.write_text(content, encoding='utf-8')
            self.log_fix(f"Created synchronous version: {sync_path}")
            return True
            
        except Exception as e:
            self.log_issue(f"Failed to create synchronous version: {e}")
            return False
            
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run complete installation diagnostics."""
        print("🔍 Running OpenWebUI Adaptive Memory Plugin Installation Diagnostics\n")
        
        results = {
            'python_ok': False,
            'dependencies_ok': False,
            'filter_structure_ok': False,
            'filter_loading_ok': False,
            'issues': [],
            'fixes': [],
            'warnings': []
        }
        
        # Check Python version
        results['python_ok'] = self.check_python_version()
        
        # Check dependencies
        results['dependencies_ok'] = self.check_dependencies()
        if not results['dependencies_ok']:
            if input("\n🤔 Install missing dependencies? (y/n): ").lower() == 'y':
                if self.install_missing_dependencies():
                    results['dependencies_ok'] = self.check_dependencies()
        
        # Validate filter files
        filter_files = [
            Path(__file__).parent / "adaptive_memory_v4.0.py",
        ]
        
        for filter_path in filter_files:
            if filter_path.exists():
                structure_ok = self.validate_filter_structure(filter_path)
                loading_ok = self.test_filter_loading(filter_path)
                
                if filter_path.name == "adaptive_memory_v4.0.py":
                    results['filter_structure_ok'] = structure_ok
                    results['filter_loading_ok'] = loading_ok
        
        # Create helpful files
        if not results['filter_loading_ok']:
            print("\n🔧 Creating helpful files for troubleshooting...")
            self.create_minimal_filter()
            self.create_sync_version()
        
        self.create_installation_guide()
        
        # Compile results
        results['issues'] = self.issues_found
        results['fixes'] = self.fixes_applied
        results['warnings'] = self.warnings
        
        return results
        
    def print_summary(self, results: Dict[str, Any]):
        """Print diagnostic summary."""
        print("\n" + "="*60)
        print("📋 DIAGNOSTIC SUMMARY")
        print("="*60)
        
        status_items = [
            ("Python Version", results['python_ok']),
            ("Dependencies", results['dependencies_ok']),
            ("Filter Structure", results['filter_structure_ok']),
            ("Filter Loading", results['filter_loading_ok']),
        ]
        
        for item, ok in status_items:
            status = "✅ PASS" if ok else "🔴 FAIL"
            print(f"{item:20} {status}")
        
        if results['issues']:
            print(f"\n🔴 Issues Found ({len(results['issues'])}):")
            for issue in results['issues']:
                print(f"  {issue}")
        
        if results['fixes']:
            print(f"\n✅ Fixes Applied ({len(results['fixes'])}):")
            for fix in results['fixes']:
                print(f"  {fix}")
                
        if results['warnings']:
            print(f"\n⚠️  Warnings ({len(results['warnings'])}):")
            for warning in results['warnings']:
                print(f"  {warning}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if not all([results['python_ok'], results['dependencies_ok']]):
            print("  1. Fix Python version and dependencies first")
        elif not results['filter_loading_ok']:
            print("  1. Try the minimal filter first (adaptive_memory_minimal.py)")
            print("  2. Check OpenWebUI logs for detailed error messages")
            print("  3. Try the synchronous version (adaptive_memory_v4.0_sync.py)")
        else:
            print("  1. Installation looks good! Try uploading the filter to OpenWebUI")
            print("  2. Enable debug logging to monitor functionality")
            print("  3. Test with simple preference statements")
        
        print(f"\n📖 See INSTALLATION_GUIDE.md for detailed instructions")
        print("="*60)

def main():
    """Main diagnostic function."""
    diagnostics = InstallationDiagnostics()
    results = diagnostics.run_full_diagnostics()
    diagnostics.print_summary(results)
    
    return 0 if all([
        results['python_ok'],
        results['dependencies_ok'],
        results['filter_structure_ok']
    ]) else 1

if __name__ == "__main__":
    import importlib.util
    sys.exit(main())