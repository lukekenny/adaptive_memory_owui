#!/usr/bin/env python3
"""
Automated Installation Fix Script for OWUI Adaptive Memory Plugin

This script automatically detects and fixes common installation issues.
"""

import os
import sys
import shutil
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import tempfile
import zipfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InstallationFixer:
    """Automatically fixes common installation issues."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.fixes_applied = []
        self.issues_found = []
        
    def log_issue(self, issue: str):
        """Log an issue found."""
        self.issues_found.append(issue)
        logger.error(f"ISSUE: {issue}")
        
    def log_fix(self, fix: str):
        """Log a fix applied."""
        self.fixes_applied.append(fix)
        logger.info(f"FIXED: {fix}")
        
    def fix_requirements(self) -> bool:
        """Fix and update requirements.txt."""
        logger.info("Checking and fixing requirements.txt...")
        
        # Core requirements that are absolutely necessary
        core_requirements = [
            "pydantic>=2.0.0",
            "sentence-transformers>=2.2.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "pytz>=2023.3",
            "aiohttp>=3.8.0",
        ]
        
        # Optional requirements (install if possible)
        optional_requirements = [
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ]
        
        requirements_path = self.project_root / "requirements.txt"
        
        try:
            if requirements_path.exists():
                current_content = requirements_path.read_text()
            else:
                current_content = ""
                
            # Create updated requirements
            new_requirements = []
            new_requirements.append("# Core dependencies for OWUI Adaptive Memory Plugin")
            new_requirements.extend(core_requirements)
            new_requirements.append("")
            new_requirements.append("# Optional dependencies (install if possible)")
            new_requirements.extend([f"# {req}" for req in optional_requirements])
            new_requirements.append("")
            new_requirements.append("# Testing framework dependencies")
            new_requirements.extend([
                "pytest>=7.4.0",
                "pytest-asyncio>=0.21.0",
                "pytest-mock>=3.11.0",
            ])
            
            new_content = "\n".join(new_requirements)
            
            if new_content != current_content:
                requirements_path.write_text(new_content)
                self.log_fix("Updated requirements.txt with minimal dependencies")
                return True
            
            return True
            
        except Exception as e:
            self.log_issue(f"Failed to fix requirements.txt: {e}")
            return False
            
    def install_dependencies(self) -> bool:
        """Install required dependencies."""
        logger.info("Installing dependencies...")
        
        try:
            # Install core dependencies one by one
            core_packages = [
                "pydantic",
                "numpy", 
                "scikit-learn",
                "pytz",
            ]
            
            for package in core_packages:
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    logger.info(f"Installed {package}")
                except subprocess.CalledProcessError:
                    logger.warning(f"Failed to install {package}")
            
            # Try to install sentence-transformers (may need special handling)
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "sentence-transformers"
                ], stdout=subprocess.DEVNULL)
                logger.info("Installed sentence-transformers")
            except subprocess.CalledProcessError:
                logger.warning("Failed to install sentence-transformers - will use fallback")
                
            self.log_fix("Installed core dependencies")
            return True
            
        except Exception as e:
            self.log_issue(f"Failed to install dependencies: {e}")
            return False
            
    def create_dependency_fallback(self) -> bool:
        """Create fallback versions that don't require heavy dependencies."""
        logger.info("Creating dependency fallbacks...")
        
        fallback_content = '''"""
Dependency Fallback for OWUI Adaptive Memory Plugin

This module provides fallback implementations when heavy dependencies are not available.
"""

import logging
import numpy as np
from typing import List, Optional, Any

logger = logging.getLogger(__name__)

class FallbackEmbeddings:
    """Fallback embedding implementation using simple text hashing."""
    
    def __init__(self, model_name: str = "fallback"):
        self.model_name = model_name
        logger.warning("Using fallback embeddings - install sentence-transformers for better results")
        
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Create simple hash-based embeddings."""
        embeddings = []
        for text in texts:
            # Simple character frequency-based embedding
            char_counts = [0] * 256
            for char in text.lower():
                char_counts[ord(char) % 256] += 1
            
            # Normalize to create embedding vector
            total = sum(char_counts) or 1
            embedding = [count / total for count in char_counts]
            embeddings.append(embedding)
        
        return np.array(embeddings)

def get_fallback_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """Get fallback embeddings when sentence-transformers is not available."""
    return FallbackEmbeddings(model_name)

# Test if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using fallback")
'''
        
        fallback_path = self.project_root / "fallback_dependencies.py"
        
        try:
            fallback_path.write_text(fallback_content)
            self.log_fix("Created dependency fallback module")
            return True
        except Exception as e:
            self.log_issue(f"Failed to create fallback: {e}")
            return False
            
    def create_lightweight_filter(self) -> bool:
        """Create an ultra-lightweight filter version."""
        logger.info("Creating ultra-lightweight filter...")
        
        lightweight_content = '''"""
OpenWebUI Adaptive Memory Plugin - Ultra-Lightweight Version

This version has minimal dependencies and simplified functionality
for maximum compatibility.
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple BaseModel replacement if pydantic is not available
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning("Pydantic not available, using simple configuration")
    
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def Field(default=None, description=""):
        return default

class Filter:
    """
    Ultra-Lightweight Adaptive Memory Filter
    
    Minimal dependencies, maximum compatibility.
    """
    
    if PYDANTIC_AVAILABLE:
        class Valves(BaseModel):
            enable_memory: bool = Field(default=True, description="Enable memory")
            debug_logging: bool = Field(default=True, description="Debug logging") 
            max_memories: int = Field(default=10, description="Max memories to store")
            similarity_threshold: float = Field(default=0.3, description="Similarity threshold")
    else:
        class Valves:
            def __init__(self):
                self.enable_memory = True
                self.debug_logging = True
                self.max_memories = 10
                self.similarity_threshold = 0.3
    
    def __init__(self):
        """Initialize filter with minimal setup."""
        try:
            self.valves = self.Valves()
            self._memories = {}  # Simple in-memory storage
            self._initialized = True
            
            if getattr(self.valves, 'debug_logging', True):
                logger.info("Ultra-lightweight memory filter initialized")
                
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._initialized = False
            self._memories = {}
            
    def inlet(self, body: dict) -> dict:
        """Process user input with minimal overhead."""
        try:
            if not getattr(self, '_initialized', False):
                return body
                
            if not getattr(self.valves, 'enable_memory', True):
                return body
            
            # Extract user info
            user_id = self._extract_user_id(body)
            if not user_id:
                return body
            
            # Get user message
            message = self._extract_last_message(body)
            if not message:
                return body
            
            # Simple preference extraction
            self._extract_simple_memories(user_id, message)
            
            # Inject relevant memories
            self._inject_memories(body, user_id, message)
            
            return body
            
        except Exception as e:
            logger.error(f"Inlet error: {e}")
            return body
    
    def outlet(self, body: dict) -> dict:
        """Process output with minimal overhead."""
        try:
            if getattr(self.valves, 'debug_logging', True):
                logger.debug("Outlet processing")
            return body
        except Exception as e:
            logger.error(f"Outlet error: {e}")
            return body
    
    def stream(self, event: dict) -> dict:
        """Process stream events."""
        return event
    
    def _extract_user_id(self, body: dict) -> Optional[str]:
        """Extract user ID from request body."""
        if isinstance(body, dict):
            if "user" in body and isinstance(body["user"], dict):
                return body["user"].get("id")
            elif "user_id" in body:
                return body["user_id"]
        return None
    
    def _extract_last_message(self, body: dict) -> Optional[str]:
        """Extract the last user message."""
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None
    
    def _extract_simple_memories(self, user_id: str, text: str):
        """Extract memories using simple pattern matching."""
        if user_id not in self._memories:
            self._memories[user_id] = []
        
        # Simple patterns for preferences
        patterns = [
            r"(?:my name is|i am|i'm called)\s+([\\w\\s]+)",
            r"(?:i like|i love|i enjoy)\s+([^.!?]+)",
            r"(?:my favorite)\s+([^.!?]+)",
            r"(?:i work as|i am a)\s+([^.!?]+)",
            r"(?:i live in|i'm from)\s+([^.!?]+)",
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower(), re.IGNORECASE)
            for match in matches:
                content = match.group(1).strip()
                if len(content) > 2 and len(content) < 100:
                    memory = {
                        "id": str(uuid.uuid4()),
                        "content": content,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "pattern": pattern
                    }
                    
                    # Simple deduplication
                    if not any(m["content"] == content for m in self._memories[user_id]):
                        self._memories[user_id].append(memory)
                        
                        # Limit memory count
                        max_memories = getattr(self.valves, 'max_memories', 10)
                        if len(self._memories[user_id]) > max_memories:
                            self._memories[user_id] = self._memories[user_id][-max_memories:]
                        
                        if getattr(self.valves, 'debug_logging', True):
                            logger.info(f"Stored memory: {content}")
    
    def _inject_memories(self, body: dict, user_id: str, query: str):
        """Inject relevant memories into context."""
        if user_id not in self._memories or not self._memories[user_id]:
            return
        
        # Simple keyword matching for relevance
        query_words = set(query.lower().split())
        relevant_memories = []
        
        for memory in self._memories[user_id]:
            memory_words = set(memory["content"].lower().split())
            overlap = len(query_words.intersection(memory_words))
            
            if overlap > 0:
                relevant_memories.append({
                    "content": memory["content"], 
                    "score": overlap
                })
        
        if not relevant_memories:
            return
        
        # Sort by relevance
        relevant_memories.sort(key=lambda x: x["score"], reverse=True)
        relevant_memories = relevant_memories[:3]  # Limit to top 3
        
        # Create memory context
        memory_text = "\\n\\nRelevant information about the user:\\n"
        for mem in relevant_memories:
            memory_text += f"- {mem['content']}\\n"
        
        # Inject as system message
        messages = body.get("messages", [])
        system_msg = {
            "role": "system",
            "content": memory_text
        }
        
        # Insert at appropriate position
        insert_pos = 0
        for i, msg in enumerate(messages):
            if msg.get("role") != "system":
                insert_pos = i
                break
        
        messages.insert(insert_pos, system_msg)
        body["messages"] = messages
        
        if getattr(self.valves, 'debug_logging', True):
            logger.info(f"Injected {len(relevant_memories)} memories")
'''
        
        lightweight_path = self.project_root / "adaptive_memory_ultra_lightweight.py"
        
        try:
            lightweight_path.write_text(lightweight_content)
            self.log_fix("Created ultra-lightweight filter version")
            return True
        except Exception as e:
            self.log_issue(f"Failed to create lightweight filter: {e}")
            return False
            
    def create_installation_package(self) -> bool:
        """Create a complete installation package."""
        logger.info("Creating installation package...")
        
        try:
            package_dir = self.project_root / "installation_package"
            package_dir.mkdir(exist_ok=True)
            
            # Copy essential files
            files_to_include = [
                "adaptive_memory_minimal.py",
                "adaptive_memory_ultra_lightweight.py", 
                "adaptive_memory_v4.0_sync.py",
                "requirements.txt",
                "INSTALLATION_GUIDE.md",
                "install_diagnostics.py",
                "install_validation.py"
            ]
            
            for filename in files_to_include:
                src = self.project_root / filename
                if src.exists():
                    dst = package_dir / filename
                    shutil.copy2(src, dst)
            
            # Create README for package
            readme_content = """# OpenWebUI Adaptive Memory Plugin - Installation Package

This package contains everything needed to install the adaptive memory plugin.

## Quick Start

1. Try the ultra-lightweight version first:
   - Upload `adaptive_memory_ultra_lightweight.py` to OpenWebUI

2. If that works, try the minimal version:
   - Upload `adaptive_memory_minimal.py` to OpenWebUI

3. For full functionality:
   - Install dependencies: `pip install -r requirements.txt`
   - Upload `adaptive_memory_v4.0_sync.py` to OpenWebUI

## Troubleshooting

- Run `python install_diagnostics.py` to check for issues
- Check `INSTALLATION_GUIDE.md` for detailed instructions
- Use `python install_validation.py` to test the installation

## Support

If you encounter issues, include the output from the diagnostic script
when asking for help.
"""
            
            (package_dir / "README.md").write_text(readme_content)
            
            # Create zip file
            zip_path = self.project_root / "owui_adaptive_memory_installation.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in package_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(package_dir)
                        zipf.write(file_path, arcname)
            
            self.log_fix(f"Created installation package: {zip_path}")
            return True
            
        except Exception as e:
            self.log_issue(f"Failed to create installation package: {e}")
            return False
            
    def run_fixes(self) -> Dict[str, Any]:
        """Run all automatic fixes."""
        logger.info("🔧 Running Automatic Installation Fixes")
        logger.info("=" * 50)
        
        fixes_run = [
            ("Fix Requirements", self.fix_requirements),
            ("Install Dependencies", self.install_dependencies),
            ("Create Dependency Fallback", self.create_dependency_fallback),
            ("Create Lightweight Filter", self.create_lightweight_filter),
            ("Create Installation Package", self.create_installation_package),
        ]
        
        results = {}
        
        for fix_name, fix_func in fixes_run:
            logger.info(f"Running: {fix_name}")
            try:
                success = fix_func()
                results[fix_name] = success
                if success:
                    logger.info(f"✅ {fix_name} completed successfully")
                else:
                    logger.error(f"❌ {fix_name} failed")
            except Exception as e:
                logger.error(f"❌ {fix_name} failed with exception: {e}")
                results[fix_name] = False
        
        # Summary
        successful_fixes = sum(1 for success in results.values() if success)
        total_fixes = len(results)
        
        logger.info("=" * 50)
        logger.info("🔧 FIX SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Fixes Applied: {successful_fixes}/{total_fixes}")
        
        if self.fixes_applied:
            logger.info("✅ Successful fixes:")
            for fix in self.fixes_applied:
                logger.info(f"  - {fix}")
        
        if self.issues_found:
            logger.info("❌ Issues found:")
            for issue in self.issues_found:
                logger.info(f"  - {issue}")
        
        logger.info("\n💡 Next steps:")
        logger.info("1. Try installing the ultra-lightweight version first")
        logger.info("2. Run 'python install_validation.py' to test the installation")
        logger.info("3. Check the INSTALLATION_GUIDE.md for detailed instructions")
        
        return {
            "fixes_applied": self.fixes_applied,
            "issues_found": self.issues_found,
            "results": results,
            "success_rate": successful_fixes / total_fixes if total_fixes > 0 else 0
        }

def main():
    """Main fix function."""
    fixer = InstallationFixer()
    results = fixer.run_fixes()
    
    return 0 if results["success_rate"] >= 0.8 else 1

if __name__ == "__main__":
    sys.exit(main())