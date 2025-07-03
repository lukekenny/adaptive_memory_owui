"""
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
            r"(?:my name is|i am|i'm called)\s+([\w\s]+)",
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
        memory_text = "\n\nRelevant information about the user:\n"
        for mem in relevant_memories:
            memory_text += f"- {mem['content']}\n"
        
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
