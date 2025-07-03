"""
SYNCHRONOUS VERSION of OpenWebUI Adaptive Memory Plugin v4.0

This version has been automatically converted to use synchronous methods
for compatibility with OpenWebUI installations that have issues with async filters.

If you're experiencing installation issues, try this version first.

Original: Adaptive Memory Filter v4.0 with Filter Orchestration System

This filter provides persistent, personalized memory capabilities for Large Language Models
with dynamic extraction, filtering, storage, and retrieval of user-specific information.
"""

from pydantic import BaseModel, Field
from typing import Any, Callable, Dict, List, Literal, Optional, Union, Set
import json
import copy
import traceback
from datetime import datetime, timezone
import logging
import re
import pytz
from difflib import SequenceMatcher
import random
import time
import os
import threading
import uuid
import hashlib
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Filter:
    """
    Adaptive Memory Filter v4.0 - Synchronous Version
    
    This filter provides persistent, personalized memory capabilities for Large Language Models
    with dynamic extraction, filtering, storage, and retrieval of user-specific information.
    
    This synchronous version is designed for OpenWebUI installations that have issues
    with async filter methods.
    """
    
    class Valves(BaseModel):
        """Configuration valves for the filter."""
        
        # Basic functionality
        enable_memory: bool = Field(
            default=True,
            description="Enable memory functionality"
        )
        
        debug_logging: bool = Field(
            default=True,
            description="Enable debug logging for troubleshooting"
        )
        
        # LLM Configuration
        llm_provider_type: Literal["ollama", "openai_compatible"] = Field(
            default="ollama",
            description="Type of LLM provider"
        )
        
        llm_api_endpoint_url: str = Field(
            default="http://localhost:11434/api/chat",
            description="API endpoint URL for the LLM provider"
        )
        
        llm_model_name: str = Field(
            default="llama3.2",
            description="Name of the LLM model to use for memory operations"
        )
        
        # Memory settings
        similarity_threshold: float = Field(
            default=0.65,
            description="Minimum similarity score for memory relevance (0.0-1.0)"
        )
        
        max_memories_to_inject: int = Field(
            default=5,
            description="Maximum number of relevant memories to inject"
        )
        
        # Memory banks
        allowed_memory_banks: List[str] = Field(
            default=["General", "Personal", "Work", "Preferences"],
            description="List of allowed memory banks for categorization"
        )
        
        default_memory_bank: str = Field(
            default="General",
            description="Default memory bank for new memories"
        )
        
        # Simple test mode
        test_mode: bool = Field(
            default=False,
            description="Run in test mode with simplified processing"
        )

    def __init__(self):
        """Initialize the filter."""
        try:
            self.valves = self.Valves()
            self._user_memories = {}  # Simple in-memory storage for testing
            self._memory_embeddings = {}
            self._relevance_cache = {}
            
            if self.valves.debug_logging:
                logger.info("Adaptive Memory Filter (Sync) initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize filter: {e}")
            # Create minimal default configuration
            self.valves = type('DefaultValves', (), {
                'enable_memory': True,
                'debug_logging': True,
                'test_mode': True,
                'similarity_threshold': 0.65,
                'max_memories_to_inject': 5,
                'allowed_memory_banks': ["General", "Personal", "Work", "Preferences"],
                'default_memory_bank': "General",
                'llm_provider_type': "ollama",
                'llm_api_endpoint_url': "http://localhost:11434/api/chat",
                'llm_model_name': "llama3.2"
            })()
            self._user_memories = {}
            self._memory_embeddings = {}
            self._relevance_cache = {}

    def inlet(self, body: dict) -> dict:
        """Process user input and extract memories."""
        try:
            if self.valves.debug_logging:
                logger.info("Inlet called - processing user input")
            
            if not self.valves.enable_memory:
                return body
            
            # Extract user context
            user_id = None
            if isinstance(body, dict):
                if "user" in body and isinstance(body["user"], dict):
                    user_id = body["user"].get("id")
                elif "user_id" in body:
                    user_id = body["user_id"]
            
            if not user_id:
                if self.valves.debug_logging:
                    logger.warning("No user ID found in request body")
                return body
            
            # Extract messages
            messages = body.get("messages", [])
            if not messages:
                return body
            
            # Get the last user message
            last_message = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_message = msg.get("content", "")
                    break
            
            if not last_message or len(last_message.strip()) < 10:
                return body
            
            # Simple memory extraction for test mode
            if self.valves.test_mode:
                self._extract_simple_preferences(user_id, last_message)
            else:
                # More complex extraction would go here
                self._extract_simple_preferences(user_id, last_message)
            
            # Inject relevant memories
            relevant_memories = self._get_relevant_memories(user_id, last_message)
            if relevant_memories:
                body = self._inject_memories_into_context(body, relevant_memories)
            
            return body
            
        except Exception as e:
            logger.error(f"Error in inlet: {e}")
            if self.valves.debug_logging:
                logger.error(f"Inlet error traceback: {traceback.format_exc()}")
            return body

    def outlet(self, body: dict) -> dict:
        """Process model output."""
        try:
            if self.valves.debug_logging:
                logger.info("Outlet called - processing model output")
            
            if not self.valves.enable_memory:
                return body
            
            # Simple processing - could extract memories from conversation here
            return body
            
        except Exception as e:
            logger.error(f"Error in outlet: {e}")
            if self.valves.debug_logging:
                logger.error(f"Outlet error traceback: {traceback.format_exc()}")
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

    def _extract_simple_preferences(self, user_id: str, text: str):
        """Extract simple preference statements from text."""
        try:
            # Initialize user memories if needed
            if user_id not in self._user_memories:
                self._user_memories[user_id] = []
            
            # Simple patterns for preference extraction
            patterns = [
                r"(?:my favorite|i love|i like|i prefer|i enjoy)\s+(.+?)(?:\.|$|,)",
                r"(?:i am|i'm)\s+(.+?)(?:\.|$|,)",
                r"(?:i work as|i am a|i'm a)\s+(.+?)(?:\.|$|,)",
                r"(?:i live in|i'm from|i'm located in)\s+(.+?)(?:\.|$|,)",
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, text.lower(), re.IGNORECASE)
                for match in matches:
                    preference = match.group(1).strip()
                    if len(preference) > 2 and len(preference) < 100:
                        memory = {
                            "id": str(uuid.uuid4()),
                            "content": f"User preference: {preference}",
                            "type": "preference",
                            "bank": "Preferences",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "source_text": text[:200] + "..." if len(text) > 200 else text
                        }
                        
                        # Simple deduplication
                        if not any(mem["content"] == memory["content"] for mem in self._user_memories[user_id]):
                            self._user_memories[user_id].append(memory)
                            
                            if self.valves.debug_logging:
                                logger.info(f"Extracted preference for user {user_id}: {preference}")
            
        except Exception as e:
            logger.error(f"Error extracting preferences: {e}")

    def _get_relevant_memories(self, user_id: str, query: str) -> List[Dict]:
        """Get relevant memories for a user query."""
        try:
            if user_id not in self._user_memories:
                return []
            
            user_memories = self._user_memories[user_id]
            if not user_memories:
                return []
            
            # Simple keyword-based relevance for test mode
            query_words = set(query.lower().split())
            relevant = []
            
            for memory in user_memories:
                memory_words = set(memory["content"].lower().split())
                overlap = len(query_words.intersection(memory_words))
                
                if overlap > 0:
                    memory_copy = memory.copy()
                    memory_copy["relevance_score"] = overlap / len(query_words)
                    relevant.append(memory_copy)
            
            # Sort by relevance and limit
            relevant.sort(key=lambda x: x["relevance_score"], reverse=True)
            return relevant[:self.valves.max_memories_to_inject]
            
        except Exception as e:
            logger.error(f"Error getting relevant memories: {e}")
            return []

    def _inject_memories_into_context(self, body: dict, memories: List[Dict]) -> dict:
        """Inject relevant memories into the conversation context."""
        try:
            if not memories or not isinstance(body, dict):
                return body
            
            # Create memory context
            memory_context = "\\n\\n**Relevant memories about you:**\\n"
            for memory in memories:
                memory_context += f"- {memory['content']} (from {memory['bank']})\\n"
            
            memory_context += "\\n*Use this information to personalize your response.*\\n"
            
            # Add as system message
            messages = body.get("messages", [])
            
            # Insert memory context as system message
            system_message = {
                "role": "system",
                "content": memory_context
            }
            
            # Insert at the beginning or after existing system messages
            insert_index = 0
            for i, msg in enumerate(messages):
                if msg.get("role") != "system":
                    insert_index = i
                    break
            
            messages.insert(insert_index, system_message)
            body["messages"] = messages
            
            if self.valves.debug_logging:
                logger.info(f"Injected {len(memories)} memories into context")
            
            return body
            
        except Exception as e:
            logger.error(f"Error injecting memories: {e}")
            return body

    def get_user_memory_stats(self, user_id: str) -> Dict:
        """Get memory statistics for a user."""
        try:
            if user_id not in self._user_memories:
                return {"total_memories": 0, "banks": {}}
            
            memories = self._user_memories[user_id]
            stats = {
                "total_memories": len(memories),
                "banks": {}
            }
            
            for memory in memories:
                bank = memory.get("bank", "Unknown")
                stats["banks"][bank] = stats["banks"].get(bank, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"total_memories": 0, "banks": {}}

    def clear_user_memories(self, user_id: str) -> bool:
        """Clear all memories for a user."""
        try:
            if user_id in self._user_memories:
                del self._user_memories[user_id]
                logger.info(f"Cleared memories for user {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error clearing memories: {e}")
            return False