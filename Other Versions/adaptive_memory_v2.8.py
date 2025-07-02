"""
Adaptive Memory - OpenWebUI Plugin
Author: AG

---

Adaptive Memory is an advanced, self-contained plugin that provides **personalized, persistent, and adaptive memory** capabilities for Large Language Models (LLMs) within OpenWebUI.

It **dynamically extracts, stores, retrieves, and injects** user-specific information to enable **context-aware, personalized conversations** that evolve over time.

---

## How It Works

1. **Memory Extraction**
   - Uses **LLM prompts** to extract **user-specific facts, preferences, goals, and implicit interests** from conversations.
   - Incorporates **recent conversation history** for better context.
   - Filters out **trivia, general knowledge, and meta-requests** using **regex, LLM classification, and keyword filters**.

2. **Multi-layer Filtering**
   - **Blacklist and whitelist filters** for topics and keywords.
   - **Regex-based trivia detection** to discard general knowledge.
   - **LLM-based meta-request classification** to discard transient queries.
   - **Regex-based meta-request phrase filtering**.
   - **Minimum length and relevance thresholds** to ensure quality.

3. **Memory Deduplication & Summarization**
   - Avoids storing **duplicate or highly similar memories**.
   - Periodically **summarizes older memories** into concise summaries to reduce clutter.

4. **Memory Injection**
   - Injects only the **most relevant, concise memories** into LLM prompts.
   - Limits total injected context length for efficiency.
   - Adds clear instructions to **avoid prompt leakage or hallucinations**.

5. **Output Filtering**
   - Removes any **meta-explanations or hallucinated summaries** from LLM responses before displaying to the user.

6. **Configurable Valves**
   - All thresholds, filters, and behaviors are **configurable via plugin valves**.
   - No external dependencies or servers required.

7. **Architecture Compliance**
   - Fully self-contained **OpenWebUI Filter plugin**.
   - Compatible with OpenWebUI's plugin architecture.
   - No external dependencies beyond OpenWebUI and Python standard libraries.

---

## Key Benefits

- **Highly accurate, privacy-respecting, adaptive memory** for LLMs.
- **Continuously evolves** with user interactions.
- **Minimizes irrelevant or transient data**.
- **Improves personalization and context-awareness**.
- **Easy to configure and maintain**.

---

## Summary

Adaptive Memory enables **dynamic, evolving, and accurate personalized memory** for LLMs in OpenWebUI, with **multi-layered filtering and adaptive context management**.

It is optimized for **accuracy, robustness, privacy, and compliance** with OpenWebUI's architecture and constraints.

"""

import json
import copy  # Add deepcopy import
import traceback
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, Set
import logging
import re
import asyncio
import pytz
import difflib
from difflib import SequenceMatcher
import random
import time

# Embedding model imports
from sentence_transformers import SentenceTransformer
import numpy as np

import aiohttp
from aiohttp import ClientError, ClientSession
from fastapi.requests import Request
from pydantic import BaseModel, Field, model_validator, field_validator, validator

# Updated imports for OpenWebUI 0.5+
from open_webui.routers.memories import (
    add_memory,
    AddMemoryForm,
    query_memory,
    QueryMemoryForm,
    delete_memory_by_id,
    Memories,
)
from open_webui.models.users import Users
from open_webui.main import app as webui_app

# Set up logging
logger = logging.getLogger("openwebui.plugins.adaptive_memory")
handler = logging.StreamHandler()


class JsonFormatter(logging.Formatter):
    def format(self, record):
        import json as _json

        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return _json.dumps(log_record)


formatter = JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False # Prevent duplicate logs if root logger has handlers
# Do not override root logger level; respect GLOBAL_LOG_LEVEL or root config


class MemoryOperation(BaseModel):
    """Model for memory operations"""

    operation: Literal["NEW", "UPDATE", "DELETE"]
    id: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = []


class Filter:
    # Class-level singleton attributes to avoid missing attribute errors
    _embedding_model = None
    _memory_embeddings = {}
    _relevance_cache = {}

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._embedding_model = None
        return self._embedding_model

    @property
    def memory_embeddings(self):
        if not hasattr(self, "_memory_embeddings") or self._memory_embeddings is None:
            self._memory_embeddings = {}
        return self._memory_embeddings

    @property
    def relevance_cache(self):
        if not hasattr(self, "_relevance_cache") or self._relevance_cache is None:
            self._relevance_cache = {}
        return self._relevance_cache

    class Valves(BaseModel):
        """Configuration valves for the filter"""

        # ------ Begin Background Task Management Configuration ------
        enable_summarization_task: bool = Field(
            default=True,
            description="Enable or disable the background memory summarization task"
        )
        summarization_interval: int = Field(
            default=3600,  # 1 hour in seconds
            description="Interval in seconds between memory summarization runs"
        )
        
        enable_error_logging_task: bool = Field(
            default=True,
            description="Enable or disable the background error counter logging task"
        )
        error_logging_interval: int = Field(
            default=300,  # 5 minutes in seconds
            description="Interval in seconds between error counter log entries"
        )
        
        enable_date_update_task: bool = Field(
            default=True,
            description="Enable or disable the background date update task"
        )
        date_update_interval: int = Field(
            default=900,  # 15 minutes in seconds
            description="Interval in seconds between date information updates"
        )
        
        enable_model_discovery_task: bool = Field(
            default=True,
            description="Enable or disable the background model discovery task"
        )
        model_discovery_interval: int = Field(
            default=3600,  # 1 hour in seconds
            description="Interval in seconds between model discovery runs"
        )
        # ------ End Background Task Management Configuration ------
        
        # ------ Begin Filtering & Saving Configuration ------
        enable_json_stripping: bool = Field(
            default=True,
            description="Attempt to strip non-JSON text before/after the main JSON object/array from LLM responses."
        )
        enable_fallback_regex: bool = Field(
            default=True,
            description="If primary JSON parsing fails, attempt a simple regex fallback to extract at least one memory."
        )
        enable_short_preference_shortcut: bool = Field(
            default=True,
            description="If JSON parsing fails for a short message containing preference keywords, directly save the message content."
        )
        # --- NEW: Deduplication bypass for short preference statements ---
        short_preference_no_dedupe_length: int = Field(
            default=60, # Increased from 40 to account for "[Tags: tag] " prefix
            description="If a NEW memory's content length is below this threshold and contains preference keywords, skip deduplication checks to avoid false positives."
        )
        preference_keywords_no_dedupe: str = Field(
            default="favorite,love,like,prefer,enjoy",
            description="Comma-separated keywords indicating user preferences that, when present in a short statement, trigger deduplication bypass."
        )

        # Blacklist topics (comma-separated substrings) - NOW OPTIONAL
        blacklist_topics: Optional[str] = Field(
            default=None,  # Default to None instead of empty string or default list
            description="Optional: Comma-separated list of topics to ignore during memory extraction",
        )

        # Enable trivia filtering
        filter_trivia: bool = Field(
            default=True,
            description="Enable filtering of trivia/general knowledge memories after extraction",
        )

        # Whitelist keywords (comma-separated substrings) - NOW OPTIONAL
        whitelist_keywords: Optional[str] = Field(
            default=None,  # Default to None
            description="Optional: Comma-separated keywords that force-save a memory even if blacklisted",
        )

        # Maximum total memories per user
        max_total_memories: int = Field(
            default=200,
            description="Maximum number of memories per user; prune oldest beyond this",
        )

        pruning_strategy: Literal["fifo", "least_relevant"] = Field(
            default="fifo",
            description="Strategy for pruning memories when max_total_memories is exceeded: 'fifo' (oldest first) or 'least_relevant' (lowest relevance to current message first).",
        )

        # Minimum memory length
        min_memory_length: int = Field(
            default=8, # Lowered default from 10
            description="Minimum length of memory content to be saved",
        )

        # Number of recent user messages to include in extraction context
        recent_messages_n: int = Field(
            default=5,
            description="Number of recent user messages to include in extraction prompt context",
        )

        # Relevance threshold for saving memories
        save_relevance_threshold: float = Field(
            default=0.8,
            description="Minimum relevance score (based on relevance calculation method) to save a memory",
        )

        # Max length of injected memory content (characters)
        max_injected_memory_length: int = Field(
            default=300,
            description="Maximum length of each injected memory snippet",
        )

        # --- Generic LLM Provider Configuration ---
        llm_provider_type: Literal["ollama", "openai_compatible"] = Field(
            default="ollama",
            description="Type of LLM provider ('ollama' or 'openai_compatible')",
        )
        llm_model_name: str = Field(
            default="llama3:latest",  # Default sensible for Ollama
            description="Name of the LLM model to use (e.g., 'llama3:latest', 'gpt-4o')",
        )
        llm_api_endpoint_url: str = Field(
            # Change default to use host.docker.internal for accessing Ollama on host
            default="http://host.docker.internal:11434/api/chat",
            description="API endpoint URL for the LLM provider (e.g., 'http://host.docker.internal:11434/api/chat', 'https://api.openai.com/v1/chat/completions')",
        )
        llm_api_key: Optional[str] = Field(
            default=None,
            description="API Key for the LLM provider (required if type is 'openai_compatible')",
        )
        # --- End Generic LLM Provider Configuration ---

        # Memory processing settings
        related_memories_n: int = Field(
            default=5,
            description="Number of related memories to consider",
        )
        relevance_threshold: float = Field(
            default=0.6,
            description="Minimum relevance score (0-1) for memories to be considered relevant",
        )
        memory_threshold: float = Field(
            default=0.6,
            description="Threshold for similarity when comparing memories (0-1)",
        )

        # Upgrade plan configs
        vector_similarity_threshold: float = Field(
            default=0.3,  # Lowered so short exact matches surface more often
            description="Minimum cosine similarity for vector filtering (0-1)",
        )
        top_n_memories: int = Field(
            default=5,
            description="Number of top similar memories to pass to LLM",
        )
        cache_ttl_seconds: int = Field(
            default=86400,
            description="Cache time-to-live in seconds (default 24 hours)",
        )

        # --- Relevance Calculation Configuration ---
        use_llm_for_relevance: bool = Field(
            default=True,
            description="Use LLM call for final relevance scoring (if False, relies solely on vector similarity + relevance_threshold)",
        )
        # --- End Relevance Calculation Configuration ---

        # Deduplicate identical memories
        deduplicate_memories: bool = Field(
            default=True,
            description="Prevent storing duplicate or very similar memories",
        )

        use_embeddings_for_deduplication: bool = Field(
            default=True,
            description="Use embedding-based similarity for more accurate semantic duplicate detection (if False, uses text-based similarity)",
        )

        similarity_threshold: float = Field(
            default=0.9, # Increased default from 0.8
            description="Threshold for detecting similar memories (0-1) using text or embeddings"
        )

        # Time settings
        timezone: str = Field(
            default="UTC",
            description="Timezone for date/time processing (e.g., 'America/New_York', 'Europe/London')",
        )

        # UI settings
        show_status: bool = Field(
            default=True, description="Show memory operations status in chat"
        )
        show_memories: bool = Field(
            default=True, description="Show relevant memories in context"
        )
        memory_format: Literal["bullet", "paragraph", "numbered"] = Field(
            default="bullet", description="Format for displaying memories in context"
        )

        # Memory categories
        enable_identity_memories: bool = Field(
            default=True,
            description="Enable collecting Basic Identity information (age, gender, location, etc.)",
        )
        enable_behavior_memories: bool = Field(
            default=True,
            description="Enable collecting Behavior information (interests, habits, etc.)",
        )
        enable_preference_memories: bool = Field(
            default=True,
            description="Enable collecting Preference information (likes, dislikes, etc.)",
        )
        enable_goal_memories: bool = Field(
            default=True,
            description="Enable collecting Goal information (aspirations, targets, etc.)",
        )
        enable_relationship_memories: bool = Field(
            default=True,
            description="Enable collecting Relationship information (friends, family, etc.)",
        )
        enable_possession_memories: bool = Field(
            default=True,
            description="Enable collecting Possession information (things owned or desired)",
        )

        # Error handling
        max_retries: int = Field(
            default=2, description="Maximum number of retries for API calls"
        )

        retry_delay: float = Field(
            default=1.0, description="Delay between retries (seconds)"
        )

        # System prompts
        memory_identification_prompt: str = Field(
            default="""You are an automated JSON data extraction system. Your ONLY function is to identify user-specific, persistent facts, preferences, goals, relationships, or interests from the user's messages and output them STRICTLY as a JSON array of operations.

**ABSOLUTE OUTPUT REQUIREMENT:**
+- Your ENTIRE response MUST be ONLY a valid JSON array starting with `[` and ending with `]`.
+- Each element MUST be a JSON object: `{\"operation\": \"NEW\", \"content\": \"...\", \"tags\": [\"...\"]}`
+- If NO relevant user-specific memories are found, output ONLY an empty JSON array: `[]`
+- A single memory MUST still be enclosed in an array: `[{"operation": ...}]`. DO NOT output a single JSON object `{...}`.
+- **DO NOT** include ANY text before or after the JSON array. No explanations, no greetings, no apologies, no notes, no summaries, no markdown formatting like ```json, no conversational text whatsoever. Failure to comply will break the system processing your output.

**INFORMATION TO EXTRACT (User-Specific ONLY):**
+- **Explicit Preferences/Statements:** User states "I love X", "My favorite is Y", "I enjoy Z". Extract these verbatim.
+- **Identity:** Name, location, age, profession, etc.
+- **Goals:** Aspirations, plans.
+- **Relationships:** Mentions of family, friends, colleagues.
+- **Possessions:** Things owned or desired.
+- **Behaviors/Interests:** Topics the user discusses or asks about (implying interest).

**STRICT RULES:**
+1.  **JSON ARRAY ONLY:** Output STARTS with `[` and ENDS with `]`. Nothing else.
+2.  **USER INFO ONLY:** Discard general knowledge, trivia, AI commands, or questions directed at the AI *unless* they reveal user interest (e.g., "Tell me about Rome" -> save "User is interested in Rome").
+3.  **DIRECT PREFERENCES ARE PRIORITY:** Extract all "I love/like/enjoy..." statements.
+4.  **SEPARATE ITEMS:** Each distinct piece of info is a separate JSON object in the array.
+5.  **ALLOWED TAGS ONLY:** Use ONLY `[\"identity\", \"behavior\", \"preference\", \"goal\", \"relationship\", \"possession\"]`.

**FAILURE EXAMPLES (DO NOT PRODUCE OUTPUT LIKE THIS):**
+- `{\"assistant\": \"Okay, here is the JSON: [...]"}` <-- INVALID (extra text)
+- `Okay, here you go: [{\"operation": ...}]` <-- INVALID (extra text)
+- ` ```json\n[{"operation": ...}]\n``` ` <-- INVALID (markdown)
+- `{\"memories\": [...]}` <-- INVALID (wrong structure, must be array)
+- `I found these memories: [...]` <-- INVALID (extra text)
+- `I couldn't find any memories.` <-- INVALID (Output `[]` instead)

Analyze the following user message(s) and provide ONLY the JSON array output. Adhere strictly to the format requirements.""",
            description="System prompt for memory identification (Very strict JSON focus)",
        )

        memory_relevance_prompt: str = Field(
            default="""You are a memory retrieval assistant. Your task is to determine which memories are relevant to the current context of a conversation.

IMPORTANT: **Do NOT mark general knowledge, trivia, or unrelated facts as relevant.** Only user-specific, persistent information should be rated highly.

Given the current user message and a set of memories, rate each memory's relevance on a scale from 0 to 1, where:
- 0 means completely irrelevant
- 1 means highly relevant and directly applicable

Consider:
- Explicit mentions in the user message
- Implicit connections to the user's personal info, preferences, goals, or relationships
- Potential usefulness for answering questions **about the user**
- Recency and importance of the memory

Examples:
- "User likes coffee" → likely relevant if coffee is mentioned
- "World War II started in 1939" → **irrelevant trivia, rate near 0**
- "User's friend is named Sarah" → relevant if friend is mentioned

Return your analysis as a JSON array with each memory's content, ID, and relevance score.
Example: [{"memory": "User likes coffee", "id": "123", "relevance": 0.8}]

Your output must be valid JSON only. No additional text.""",
            description="System prompt for memory relevance assessment",
        )

        memory_merge_prompt: str = Field(
            default="""You are a memory consolidation assistant. When given sets of memories, you merge similar or related memories while preserving all important information.

IMPORTANT: **Do NOT merge general knowledge, trivia, or unrelated facts.** Only merge user-specific, persistent information.

Rules for merging:
1. If two memories contradict, keep the newer information
2. Combine complementary information into a single comprehensive memory
3. Maintain the most specific details when merging
4. If two memories are distinct enough, keep them separate
5. Remove duplicate memories

Return your result as a JSON array of strings, with each string being a merged memory.
Your output must be valid JSON only. No additional text.""",
            description="System prompt for merging memories",
        )

        @field_validator(
            'summarization_interval', 'error_logging_interval', 'date_update_interval',
            'model_discovery_interval', 'max_total_memories', 'min_memory_length',
            'recent_messages_n', 'related_memories_n', 'top_n_memories',
            'cache_ttl_seconds', 'max_retries', 'max_injected_memory_length'
        )
        def check_non_negative_int(cls, v, info):
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{info.field_name} must be a non-negative integer")
            return v

        @field_validator(
            'save_relevance_threshold', 'relevance_threshold', 'memory_threshold',
            'vector_similarity_threshold', 'similarity_threshold'
        )
        def check_threshold_float(cls, v, info):
            """Ensure threshold values are between 0.0 and 1.0"""
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"{info.field_name} must be between 0.0 and 1.0. Received: {v}"
                )
            # Special documentation for similarity_threshold since it now has two usage contexts
            if info.field_name == 'similarity_threshold':
                logger.debug(
                    f"Set similarity_threshold to {v} - this threshold is used for both text-based and embedding-based deduplication based on the 'use_embeddings_for_deduplication' setting."
                )
            return v

        @field_validator('retry_delay')
        def check_non_negative_float(cls, v, info):
            if not isinstance(v, float) or v < 0.0:
                raise ValueError(f"{info.field_name} must be a non-negative float")
            return v
        
        @field_validator('timezone')
        def check_valid_timezone(cls, v):
            try:
                pytz.timezone(v)
            except pytz.exceptions.UnknownTimeZoneError:
                raise ValueError(f"Invalid timezone string: {v}")
            except Exception as e:
                 raise ValueError(f"Error validating timezone '{v}': {e}")
            return v

        # Keep existing model validator for LLM config
        @model_validator(mode="after")
        def check_llm_config(self):
            if self.llm_provider_type == "openai_compatible" and not self.llm_api_key:
                raise ValueError(
                    "API Key (llm_api_key) is required when llm_provider_type is 'openai_compatible'"
                )

            # Basic URL validation for Ollama default
            if self.llm_provider_type == "ollama":
                if not self.llm_api_endpoint_url.startswith(("http://", "https://")):
                    raise ValueError(
                        "Ollama API Endpoint URL (llm_api_endpoint_url) must be a valid URL starting with http:// or https://"
                    )
                # Could add more specific Ollama URL checks if needed

            # Basic URL validation for OpenAI compatible
            if self.llm_provider_type == "openai_compatible":
                if not self.llm_api_endpoint_url.startswith(("http://", "https://")):
                    raise ValueError(
                        "OpenAI Compatible API Endpoint URL (llm_api_endpoint_url) must be a valid URL starting with http:// or https://"
                    )

            return self

        # --- End Pydantic Validators for Valves ---

        # Control verbosity of error counter logging. When True, counters are logged at DEBUG level; when False, they are suppressed.
        debug_error_counter_logs: bool = Field(
            default=False,
            description="Emit detailed error counter logs at DEBUG level (set to True for troubleshooting).",
        )

        # ------ End Filtering & Saving Configuration ------

        # ------ Begin Error Handling & Guarding Configuration ------
        enable_error_counter_guard: bool = Field(
            default=True,
            description="Enable guard mechanism to temporarily disable LLM/embedding features if error rates spike."
        )
        error_guard_threshold: int = Field(
            default=5,
            description="Number of specific errors (e.g., JSON parsing) within the window to trigger the guard."
        )
        error_guard_window_seconds: int = Field(
            default=600, # 10 minutes
            description="Time window (seconds) to monitor errors for the guard mechanism."
        )

        # Error handling
        max_retries: int = Field(
            default=2, description="Maximum number of retries for API calls"
        )

        retry_delay: float = Field(
            default=1.0, description="Delay between retries (seconds)"
        )

        # System prompts
        memory_identification_prompt: str = Field(
            default="""You are an automated JSON data extraction system. Your ONLY function is to identify user-specific, persistent facts, preferences, goals, relationships, or interests from the user's messages and output them STRICTLY as a JSON array of operations.

**ABSOLUTE OUTPUT REQUIREMENT:**
+- Your ENTIRE response MUST be ONLY a valid JSON array starting with `[` and ending with `]`.
+- Each element MUST be a JSON object: `{\"operation\": \"NEW\", \"content\": \"...\", \"tags\": [\"...\"]}`
+- If NO relevant user-specific memories are found, output ONLY an empty JSON array: `[]`
+- A single memory MUST still be enclosed in an array: `[{"operation": ...}]`. DO NOT output a single JSON object `{...}`.
+- **DO NOT** include ANY text before or after the JSON array. No explanations, no greetings, no apologies, no notes, no summaries, no markdown formatting like ```json, no conversational text whatsoever. Failure to comply will break the system processing your output.

**INFORMATION TO EXTRACT (User-Specific ONLY):**
+- **Explicit Preferences/Statements:** User states "I love X", "My favorite is Y", "I enjoy Z". Extract these verbatim.
+- **Identity:** Name, location, age, profession, etc.
+- **Goals:** Aspirations, plans.
+- **Relationships:** Mentions of family, friends, colleagues.
+- **Possessions:** Things owned or desired.
+- **Behaviors/Interests:** Topics the user discusses or asks about (implying interest).

**STRICT RULES:**
+1.  **JSON ARRAY ONLY:** Output STARTS with `[` and ENDS with `]`. Nothing else.
+2.  **USER INFO ONLY:** Discard general knowledge, trivia, AI commands, or questions directed at the AI *unless* they reveal user interest (e.g., "Tell me about Rome" -> save "User is interested in Rome").
+3.  **DIRECT PREFERENCES ARE PRIORITY:** Extract all "I love/like/enjoy..." statements.
+4.  **SEPARATE ITEMS:** Each distinct piece of info is a separate JSON object in the array.
+5.  **ALLOWED TAGS ONLY:** Use ONLY `[\"identity\", \"behavior\", \"preference\", \"goal\", \"relationship\", \"possession\"]`.

**FAILURE EXAMPLES (DO NOT PRODUCE OUTPUT LIKE THIS):**
+- `{\"assistant\": \"Okay, here is the JSON: [...]"}` <-- INVALID (extra text)
+- `Okay, here you go: [{\"operation": ...}]` <-- INVALID (extra text)
+- ` ```json\n[{"operation": ...}]\n``` ` <-- INVALID (markdown)
+- `{\"memories\": [...]}` <-- INVALID (wrong structure, must be array)
+- `I found these memories: [...]` <-- INVALID (extra text)
+- `I couldn't find any memories.` <-- INVALID (Output `[]` instead)

Analyze the following user message(s) and provide ONLY the JSON array output. Adhere strictly to the format requirements.""",
            description="System prompt for memory identification (Very strict JSON focus)",
        )

        memory_relevance_prompt: str = Field(
            default="""You are a memory retrieval assistant. Your task is to determine which memories are relevant to the current context of a conversation.

IMPORTANT: **Do NOT mark general knowledge, trivia, or unrelated facts as relevant.** Only user-specific, persistent information should be rated highly.

Given the current user message and a set of memories, rate each memory's relevance on a scale from 0 to 1, where:
- 0 means completely irrelevant
- 1 means highly relevant and directly applicable

Consider:
- Explicit mentions in the user message
- Implicit connections to the user's personal info, preferences, goals, or relationships
- Potential usefulness for answering questions **about the user**
- Recency and importance of the memory

Examples:
- "User likes coffee" → likely relevant if coffee is mentioned
- "World War II started in 1939" → **irrelevant trivia, rate near 0**
- "User's friend is named Sarah" → relevant if friend is mentioned

Return your analysis as a JSON array with each memory's content, ID, and relevance score.
Example: [{"memory": "User likes coffee", "id": "123", "relevance": 0.8}]

Your output must be valid JSON only. No additional text.""",
            description="System prompt for memory relevance assessment",
        )

        memory_merge_prompt: str = Field(
            default="""You are a memory consolidation assistant. When given sets of memories, you merge similar or related memories while preserving all important information.

IMPORTANT: **Do NOT merge general knowledge, trivia, or unrelated facts.** Only merge user-specific, persistent information.

Rules for merging:
1. If two memories contradict, keep the newer information
2. Combine complementary information into a single comprehensive memory
3. Maintain the most specific details when merging
4. If two memories are distinct enough, keep them separate
5. Remove duplicate memories

Return your result as a JSON array of strings, with each string being a merged memory.
Your output must be valid JSON only. No additional text.""",
            description="System prompt for merging memories",
        )

        @field_validator(
            'summarization_interval', 'error_logging_interval', 'date_update_interval',
            'model_discovery_interval', 'max_total_memories', 'min_memory_length',
            'recent_messages_n', 'related_memories_n', 'top_n_memories',
            'cache_ttl_seconds', 'max_retries', 'max_injected_memory_length'
        )
        def check_non_negative_int(cls, v, info):
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{info.field_name} must be a non-negative integer")
            return v

        @field_validator(
            'save_relevance_threshold', 'relevance_threshold', 'memory_threshold',
            'vector_similarity_threshold', 'similarity_threshold'
        )
        def check_threshold_float(cls, v, info):
            """Ensure threshold values are between 0.0 and 1.0"""
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    f"{info.field_name} must be between 0.0 and 1.0. Received: {v}"
                )
            # Special documentation for similarity_threshold since it now has two usage contexts
            if info.field_name == 'similarity_threshold':
                logger.debug(
                    f"Set similarity_threshold to {v} - this threshold is used for both text-based and embedding-based deduplication based on the 'use_embeddings_for_deduplication' setting."
                )
            return v

        @field_validator('retry_delay')
        def check_non_negative_float(cls, v, info):
            if not isinstance(v, float) or v < 0.0:
                raise ValueError(f"{info.field_name} must be a non-negative float")
            return v
        
        @field_validator('timezone')
        def check_valid_timezone(cls, v):
            try:
                pytz.timezone(v)
            except pytz.exceptions.UnknownTimeZoneError:
                raise ValueError(f"Invalid timezone string: {v}")
            except Exception as e:
                 raise ValueError(f"Error validating timezone '{v}': {e}")
            return v

        # Keep existing model validator for LLM config
        @model_validator(mode="after")
        def check_llm_config(self):
            if self.llm_provider_type == "openai_compatible" and not self.llm_api_key:
                raise ValueError(
                    "API Key (llm_api_key) is required when llm_provider_type is 'openai_compatible'"
                )

            # Basic URL validation for Ollama default
            if self.llm_provider_type == "ollama":
                if not self.llm_api_endpoint_url.startswith(("http://", "https://")):
                    raise ValueError(
                        "Ollama API Endpoint URL (llm_api_endpoint_url) must be a valid URL starting with http:// or https://"
                    )
                # Could add more specific Ollama URL checks if needed

            # Basic URL validation for OpenAI compatible
            if self.llm_provider_type == "openai_compatible":
                if not self.llm_api_endpoint_url.startswith(("http://", "https://")):
                    raise ValueError(
                        "OpenAI Compatible API Endpoint URL (llm_api_endpoint_url) must be a valid URL starting with http:// or https://"
                    )

            return self

        # --- End Pydantic Validators for Valves ---

        # Control verbosity of error counter logging. When True, counters are logged at DEBUG level; when False, they are suppressed.
        debug_error_counter_logs: bool = Field(
            default=False,
            description="Emit detailed error counter logs at DEBUG level (set to True for troubleshooting).",
        )

        # ------ End Error Handling & Guarding Configuration ------

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True, description="Enable or disable the memory function"
        )
        show_status: bool = Field(
            default=True, description="Show memory processing status updates"
        )
        timezone: str = Field(
            default="",
            description="User's timezone (overrides global setting if provided)",
        )

    def __init__(self):
        """Initialize filter and schedule background tasks"""
        self.valves = self.Valves()
        self.stored_memories = None
        self._error_message = None # Stores the reason for the last failure (e.g., json_parse_error)
        self._aiohttp_session = None

        # --- Added initialisations to prevent AttributeError ---
        # Track already-processed user messages to avoid duplicate extraction
        self._processed_messages: Set[str] = set()
        # Simple metrics counter dictionary
        self.metrics: Dict[str, int] = {"llm_call_count": 0}
        # Hold last processed body for confirmation tagging
        self._last_body: Dict[str, Any] = {}

        # Background tasks tracking
        self._background_tasks = set()

        # Error counters
        self.error_counters = {
            "embedding_errors": 0,
            "llm_call_errors": 0,
            "json_parse_errors": 0,
            "memory_crud_errors": 0,
        }
        
        # Log configuration for deduplication, helpful for testing and validation
        logger.debug(f"Memory deduplication settings:")
        logger.debug(f"  - deduplicate_memories: {self.valves.deduplicate_memories}")
        logger.debug(f"  - use_embeddings_for_deduplication: {self.valves.use_embeddings_for_deduplication}")
        logger.debug(f"  - similarity_threshold: {self.valves.similarity_threshold}")

        # Schedule background tasks based on configuration valves
        if self.valves.enable_error_logging_task:
            self._error_log_task = asyncio.create_task(self._log_error_counters_loop())
            self._background_tasks.add(self._error_log_task)
            self._error_log_task.add_done_callback(self._background_tasks.discard)
            logger.debug("Started error logging background task")

        if self.valves.enable_summarization_task:
            self._summarization_task = asyncio.create_task(
                self._summarize_old_memories_loop()
            )
            self._background_tasks.add(self._summarization_task)
            self._summarization_task.add_done_callback(self._background_tasks.discard)
            logger.debug("Started memory summarization background task")

        # Model discovery results
        self.available_ollama_models = []
        self.available_openai_models = []

        # Add current date awareness for prompts
        self.current_date = datetime.now()
        self.date_info = self._update_date_info()

        # Schedule date update task if enabled
        if self.valves.enable_date_update_task:
            self._date_update_task = self._schedule_date_update()
            logger.debug("Scheduled date update background task")
        else:
            self._date_update_task = None

        # Schedule model discovery task if enabled
        if self.valves.enable_model_discovery_task:
            self._model_discovery_task = self._schedule_model_discovery()
            logger.debug("Scheduled model discovery background task")
        else:
            self._model_discovery_task = None

        # Initialize MiniLM embedding model (singleton)
        # self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2") # Removed: Property handles lazy init

        # In-memory store: memory_id -> embedding vector (np.array)
        self._memory_embeddings = {}

        # In-memory cache: (hash of user_emb + mem_emb) -> (score, timestamp)
        self._relevance_cache = {}

        # Error counter tracking for guard mechanism (Point 8)
        from collections import deque
        self.error_timestamps = {
            "json_parse_errors": deque(),
            # Add other error types here if needed for guarding
        }
        self._guard_active = False
        self._guard_activated_at = 0

        # Initialize duplicate counters (used in process_memories)
        self._duplicate_skipped = 0
        self._duplicate_refreshed = 0

    async def _summarize_old_memories_loop(self):
        """Periodically summarize old memories into concise summaries"""
        try:
            while True:
                # Use configurable interval with small random jitter to prevent thundering herd
                jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                interval = self.valves.summarization_interval * jitter
                await asyncio.sleep(interval)
                
                try:
                    # For each user, summarize old memories
                    # For simplicity, assume single user context
                    user_id = (
                        "default"  # Replace with actual user id logic if multi-user
                    )
                    user_obj = Users.get_user_by_id(user_id)
                    if not user_obj:
                        continue

                    all_mems = await self._get_formatted_memories(user_id)
                    if len(all_mems) < 5:
                        continue  # Not enough to summarize

                    # Sort by created_at or fallback
                    sorted_mems = sorted(
                        all_mems, key=lambda m: m.get("created_at", datetime.utcnow())
                    )
                    old_mems = sorted_mems[:5]  # Summarize oldest 5

                    mem_texts = [m.get("memory", "") for m in old_mems]
                    combined = "\n".join(mem_texts)

                    prompt = (
                        "Summarize the following user memories into a concise paragraph, preserving key facts:\n"
                        + combined
                    )

                    summary = await self.query_llm_with_retry(
                        "You are a memory summarizer.", prompt
                    )

                    if summary and not summary.startswith("Error:"):
                        # Save summary as new memory
                        await add_memory(
                            request=Request(scope={"type": "http", "app": webui_app}),
                            form_data=AddMemoryForm(content=summary),
                            user=user_obj,
                        )
                        # Delete originals
                        for m in old_mems:
                            try:
                                await delete_memory_by_id(m["id"], user=user_obj)
                            except Exception as del_err:
                                logger.warning(f"Failed to delete old memory {m.get('id')} during summarization: {del_err}")
                                pass
                        logger.info("Summarized and pruned old memories")
                except Exception as e:
                    logger.error(f"Error in summarization loop: {e}")
        except asyncio.CancelledError:
            logger.debug("Summarization task cancelled")

    def _update_date_info(self):
        """Update the date information dictionary with current time"""
        return {
            "iso_date": self.current_date.strftime("%Y-%m-%d"),
            "year": self.current_date.year,
            "month": self.current_date.strftime("%B"),
            "day": self.current_date.day,
            "weekday": self.current_date.strftime("%A"),
            "hour": self.current_date.hour,
            "minute": self.current_date.minute,
            "iso_time": self.current_date.strftime("%H:%M:%S"),
        }

    async def _log_error_counters_loop(self):
        """Periodically log error counters"""
        try:
            while True:
                # Use configurable interval with small random jitter
                jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                interval = self.valves.error_logging_interval * jitter
                await asyncio.sleep(interval)
                
                # Determine logging behaviour based on valve settings
                if self.valves.debug_error_counter_logs:
                    # Verbose debug logging – every interval
                    logger.debug(f"Error counters: {self.error_counters}")
                else:
                    # Only log when at least one counter is non-zero to reduce clutter
                    if any(count > 0 for count in self.error_counters.values()):
                        logger.info(f"Error counters (non-zero): {self.error_counters}")

                # Point 8: Error Counter Guard Logic
                if self.valves.enable_error_counter_guard:
                    now = time.time()
                    window = self.valves.error_guard_window_seconds
                    threshold = self.valves.error_guard_threshold

                    # Check JSON parse errors
                    error_type = "json_parse_errors"
                    # Record current count as a timestamp
                    current_count = self.error_counters[error_type]
                    # --- NOTE: This simple approach assumes the counter *increases* to track new errors.
                    # If the counter could be reset externally, a more robust timestamp queue is needed.
                    # For simplicity, assuming monotonically increasing count for now.
                    # A better approach: Store timestamp of each error occurrence.
                    # Let's refine this: Add timestamp whenever the error counter increments.
                    # We need to modify where the counter is incremented.

                    # --- Revised approach: Use a deque to store timestamps of recent errors --- 
                    timestamps = self.error_timestamps[error_type]
                    
                    # Remove old timestamps outside the window
                    while timestamps and timestamps[0] < now - window:
                        timestamps.popleft()

                    # Check if the count within the window exceeds the threshold
                    if len(timestamps) >= threshold:
                        if not self._guard_active:
                            logger.warning(f"Guard Activated: {error_type} count ({len(timestamps)}) reached threshold ({threshold}) in window ({window}s). Temporarily disabling LLM relevance and embedding dedupe.")
                            self._guard_active = True
                            self._guard_activated_at = now
                            # Temporarily disable features
                            self._original_use_llm_relevance = self.valves.use_llm_for_relevance
                            self._original_use_embedding_dedupe = self.valves.use_embeddings_for_deduplication
                            self.valves.use_llm_for_relevance = False
                            self.valves.use_embeddings_for_deduplication = False
                        elif self._guard_active:
                            # Deactivate guard if error rate drops below threshold (with hysteresis?)
                            # For simplicity, deactivate immediately when below threshold.
                            logger.info(f"Guard Deactivated: {error_type} count ({len(timestamps)}) below threshold ({threshold}). Re-enabling LLM relevance and embedding dedupe.")
                            self._guard_active = False
                            # Restore original settings
                            if hasattr(self, '_original_use_llm_relevance'):
                                self.valves.use_llm_for_relevance = self._original_use_llm_relevance
                            if hasattr(self, '_original_use_embedding_dedupe'):
                                self.valves.use_embeddings_for_deduplication = self._original_use_embedding_dedupe
        except asyncio.CancelledError:
            logger.debug("Error counter logging task cancelled")
        except Exception as e:
            logger.error(
                f"Error in error counter logging task: {e}\n{traceback.format_exc()}"
            )

    def _schedule_date_update(self):
        """Schedule a regular update of the date information"""

        async def update_date_loop():
            try:
                while True:
                    # Use configurable interval with small random jitter
                    jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                    interval = self.valves.date_update_interval * jitter
                    await asyncio.sleep(interval)
                    
                    self.current_date = self.get_formatted_datetime()
                    self.date_info = self._update_date_info()
                    logger.debug(f"Updated date information: {self.date_info}")
            except asyncio.CancelledError:
                logger.debug("Date update task cancelled")
            except Exception as e:
                logger.error(f"Error in date update task: {e}")

        # Start the update loop in the background
        task = asyncio.create_task(update_date_loop())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _schedule_model_discovery(self):
        """Schedule a regular update of available models"""

        async def discover_models_loop():
            try:
                while True:
                    try:
                        # Discover models
                        await self._discover_models()
                        
                        # Use configurable interval with small random jitter
                        jitter = random.uniform(0.9, 1.1)  # ±10% randomization
                        interval = self.valves.model_discovery_interval * jitter
                        await asyncio.sleep(interval)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Error in model discovery: {e}")
                        # On error, retry sooner (1/6 of normal interval)
                        await asyncio.sleep(self.valves.model_discovery_interval / 6)
            except asyncio.CancelledError:
                logger.debug("Model discovery task cancelled")

        # Start the discovery loop in the background
        task = asyncio.create_task(discover_models_loop())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _discover_models(self):
        """Discover available models from open_webui.configured providers"""
        logger.debug("Starting model discovery")

        # Create a session if needed
        session = await self._get_aiohttp_session()

        # Discover Ollama models
        try:
            ollama_url = "http://host.docker.internal:11434/api/tags"
            async with session.get(ollama_url) as response:
                if response.status == 200:
                    data = await response.json()
                    if "models" in data:
                        self.available_ollama_models = [
                            model["name"] for model in data["models"]
                        ]
                    logger.debug(
                        f"Discovered {len(self.available_ollama_models)} Ollama models"
                    )
        except Exception as e:
            logger.warning(f"Error discovering Ollama models: {e}")
            self.available_ollama_models = []

    def get_formatted_datetime(self, user_timezone=None):
        """
        Get properly formatted datetime with timezone awareness

        Args:
            user_timezone: Optional timezone string to override the default

        Returns:
            Timezone-aware datetime object
        """
        timezone_str = user_timezone or self.valves.timezone or "UTC"

        # Normalize common aliases
        alias_map = {
            "UAE/Dubai": "Asia/Dubai",
            "GMT+4": "Asia/Dubai",
            "GMT +4": "Asia/Dubai",
            "Dubai": "Asia/Dubai",
            "EST": "America/New_York",
            "PST": "America/Los_Angeles",
            "CST": "America/Chicago",
            "IST": "Asia/Kolkata",
            "BST": "Europe/London",
            "GMT": "Etc/GMT",
            "UTC": "UTC",
        }
        tz_key = timezone_str.strip()
        timezone_str = alias_map.get(tz_key, timezone_str)

        try:
            utc_now = datetime.utcnow()
            local_tz = pytz.timezone(timezone_str)
            local_now = utc_now.replace(tzinfo=pytz.utc).astimezone(local_tz)
            return local_now
        except pytz.exceptions.UnknownTimeZoneError:
            logger.warning(
                f"Invalid timezone: {timezone_str}, falling back to default 'Asia/Dubai'"
            )
            try:
                local_tz = pytz.timezone("Asia/Dubai")
                local_now = (
                    datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(local_tz)
                )
                return local_now
            except Exception:
                logger.warning("Fallback timezone also invalid, using UTC")
                return datetime.utcnow().replace(tzinfo=pytz.utc)

    async def _get_aiohttp_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session"""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            self._aiohttp_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)  # 30 second timeout
            )
        return self._aiohttp_session

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process incoming messages and inject relevant memories into the context"""
        self.stored_memories = None
        self._error_message = None

        if not body or not isinstance(body, dict) or not __user__:
            logger.debug("Missing body or user information in inlet")
        # Filter LLM output to remove meta-explanations
        try:
            if "messages" in body and isinstance(body["messages"], list):
                for msg in body["messages"]:
                    if msg.get("role") == "assistant" and isinstance(
                        msg.get("content"), str
                    ):
                        content = msg["content"]
                        meta_patterns = [
                            r"(?i)here'?s what i (currently )?remember",
                            r"(?i)i have not remembered",
                            r"(?i)based on our interaction",
                            r"(?i)preferred topics",
                            r"(?i)interaction history",
                            r"(?i)username",
                            r"(?i)language preference",
                            r"(?i)tags associated",
                            r"(?i)to remember more",
                            r"(?i)how would you like to proceed",
                            r"(?i)since our interaction just began",
                            r"(?i)breakdown of what i have",
                            r"(?i)information remembered about you",
                            r"(?i)information stored about you",
                            r"(?i)if you'd like to share more",
                            r"(?i)optional",
                        ]
                        for pat in meta_patterns:
                            match = re.search(pat, content)
                            if match:
                                logger.info(
                                    f"Filtered meta-explanation from LLM output: {content[match.start():match.end()]}"
                                )
                                msg["content"] = content[: match.start()].strip()
                                break
        except Exception:
            pass

        # Check if the function is enabled for this user
        user_valves = self._get_user_valves(__user__)
        if not user_valves.enabled:
            logger.debug(
                f"Memory manager disabled for user {__user__.get('id', 'unknown')}"
            )
            return body

        try:
            if "messages" in body and body["messages"]:
                user_messages = [m for m in body["messages"] if m["role"] == "user"]
                if user_messages:
                    # Determine if status updates should be shown
                    show_status = user_valves.show_status

                    # Safely emit status update
                    if show_status:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": "🔍 Scanning your personal memory bank for relevant information…",
                                    "done": False,
                                },
                            },
                        )

                    # Get user timezone if specified
                    user_timezone = (
                        user_valves.timezone
                        if user_valves.timezone
                        else self.valves.timezone
                    )

                    # Get relevant memories for the current context
                    relevant_memories = await self.get_relevant_memories(
                        user_messages[-1]["content"], __user__["id"], user_timezone
                    )

                    # Safely emit completion status (different message if none found)
                    if show_status:
                        if relevant_memories:
                            count = len(relevant_memories)
                            plural = "memory" if count == 1 else "memories"
                            retrieval_desc = f"✅ Retrieved {count} relevant {plural} from memory bank"
                        else:
                            retrieval_desc = "ℹ️ No stored memories are relevant to the current conversation."
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": retrieval_desc,
                                    "done": True,
                                },
                            },
                        )

                    # Inject relevant memories into the context if show_memories is enabled
                    if self.valves.show_memories and relevant_memories:
                        self._inject_memories_into_context(body, relevant_memories)

        except Exception as e:
            logger.error(f"Error in inlet: {e}\n{traceback.format_exc()}")
            await self._safe_emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "description": f"❌ Memory retrieval error – see logs for details: {str(e)}",
                        "done": True,
                    },
                },
            )

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Process LLM response, extract memories, and update the response"""

        # Log function entry
        logger.debug("Outlet called - making deep copy of body dictionary")

        # DEFENSIVE: Make a deep copy of the body to avoid dictionary changed size during iteration
        # This was a source of many subtle bugs
        body_copy = copy.deepcopy(body)

        # Skip processing if user is not authenticated
        if not __user__:
            logger.warning("No user information available - skipping memory processing")
            return body_copy

        # Get user's ID for memory storage
        user_id = __user__.get("id")
        if not user_id:
            logger.warning("User object contains no ID - skipping memory processing")
            return body_copy

        # Check if user has enabled memory function
        user_valves = self._get_user_valves(__user__)
        if not user_valves.enabled:
            logger.info(f"Memory function is disabled for user {user_id}")
            return body_copy

        # Get user's timezone if set
        user_timezone = user_valves.timezone or self.valves.timezone

        # Extract user message and chat history
        user_message = None
        recent_chat_history = None
        try:
            # Make a deep copy of messages to prevent modifications affecting iteration
            messages_copy = copy.deepcopy(body_copy.get("messages", []))

            # Skip if no messages
            if not messages_copy:
                logger.debug("No messages found in body")
                return body_copy

            # Log message count
            logger.debug(f"Processing {len(messages_copy)} messages for memories")

            # Get recent message history (up to n messages) for context
            recent_n = min(len(messages_copy), self.valves.recent_messages_n)
            recent_chat_history = messages_copy[-recent_n:]

            # Find the last user message
            for msg in reversed(messages_copy):
                if msg.get("role") == "user" and msg.get("content"):
                    user_message = msg.get("content")
                    logger.info(
                        f"Found user message for memory processing: '{user_message[:50]}...'"
                    )
                    break
        except Exception as e:
            logger.error(f"Error extracting message content: {e}")
            return body_copy  # Return unmodified on error

        # If we have a user message, process it for memories in the background
        if user_message:
            try:
                # Debug logging before starting the background task
                logger.debug(
                    f"Creating background task for memory processing (user_id: {user_id})"
                )

                # Create and start background task
                memory_task = asyncio.create_task(
                    self._process_user_memories(
                        user_message=user_message,
                        user_id=user_id,
                        event_emitter=__event_emitter__,
                        show_status=user_valves.show_status,
                        user_timezone=user_timezone,
                        recent_chat_history=recent_chat_history,
                    )
                )

                # Log the task creation
                logger.debug(f"Background memory processing task created: {memory_task}")

                # Optional: Wait for task to complete - can be disabled for truly async behavior
                try:
                    # Wait for the memory processing to complete, up to a timeout
                    await asyncio.wait_for(memory_task, timeout=90.0)
                    logger.info("Memory processing task completed")
                except asyncio.TimeoutError:
                    logger.warning(
                        "Memory processing task timed out - continuing in background"
                    )
                    # Continue without waiting for completion
                except Exception as e:
                    logger.error(f"Error in memory processing task: {e}")
                    # Continue even if there was an error
            except Exception as e:
                logger.error(
                    f"Error creating memory processing task: {e}\n{traceback.format_exc()}"
                )

        # Process the response content for injecting memories
        try:
            # Get relevant memories for context injection on next interaction
            memories = await self.get_relevant_memories(
                current_message=user_message or "",
                user_id=user_id,
                user_timezone=user_timezone,
            )

            # If we found relevant memories and the user wants to see them
            if memories and self.valves.show_memories:
                # Inject memories into the context for the next interaction
                self._inject_memories_into_context(body_copy, memories)
                logger.debug(f"Injected {len(memories)} memories into context")
        except Exception as e:
            logger.error(
                f"Error processing memories for context: {e}\n{traceback.format_exc()}"
            )

        # Add confirmation message if memories were processed
        try:
            if user_valves.show_status:
                await self._add_confirmation_message(body_copy)
        except Exception as e:
            logger.error(f"Error adding confirmation message: {e}")

        # Return the modified response
        return body_copy

    async def _safe_emit(
        self,
        event_emitter: Optional[Callable[[Any], Awaitable[None]]],
        data: Dict[str, Any],
    ) -> None:
        """Safely emit an event, handling missing emitter"""
        if not event_emitter:
            logger.debug("Event emitter not available")
            return

        try:
            await event_emitter(data)
        except Exception as e:
            logger.error(f"Error in event emitter: {e}")

    def _get_user_valves(self, __user__: dict) -> UserValves:
        """Extract and validate user valves settings"""
        if not __user__:
            logger.warning("No user information provided")
            return self.UserValves()

        # Access the valves attribute directly from the UserModel object
        user_valves_data = getattr(
            __user__, "valves", {}
        )  # Use getattr for safe access

        # Ensure we have a dictionary to work with
        if not isinstance(user_valves_data, dict):
            logger.warning(
                f"User valves attribute is not a dictionary (type: {type(user_valves_data)}), using defaults."
            )
            user_valves_data = {}

        try:
            # Validate and return the UserValves model
            return self.UserValves(**user_valves_data)
        except Exception as e:
            # Default to enabled if validation/extraction fails
            logger.error(
                f"Could not determine user valves settings from data {user_valves_data}: {e}"
            )
            return self.UserValves()  # Return default UserValves on error

    async def _get_formatted_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all memories for a user and format them for processing"""
        memories_list = []
        try:
            # Get memories using Memories.get_memories_by_user_id
            user_memories = Memories.get_memories_by_user_id(user_id=str(user_id))

            if user_memories:
                for memory in user_memories:
                    # Safely extract attributes with fallbacks
                    memory_id = str(getattr(memory, "id", "unknown"))
                    memory_content = getattr(memory, "content", "")
                    created_at = getattr(memory, "created_at", None)
                    updated_at = getattr(memory, "updated_at", None)

                    memories_list.append(
                        {
                            "id": memory_id,
                            "memory": memory_content,
                            "created_at": created_at,
                            "updated_at": updated_at,
                        }
                    )

            logger.debug(f"Retrieved {len(memories_list)} memories for user {user_id}")
            return memories_list

        except Exception as e:
            logger.error(
                f"Error getting formatted memories: {e}\n{traceback.format_exc()}"
            )
            return []

    def _inject_memories_into_context(
        self, body: Dict[str, Any], memories: List[Dict[str, Any]]
    ) -> None:
        """Inject relevant memories into the system context"""
        if not memories:
            # Suppress fallback injection when no relevant memories
            return

        # Sort memories by relevance if available
        sorted_memories = sorted(
            memories, key=lambda x: x.get("relevance", 0), reverse=True
        )

        # Format memories based on user preference
        memory_context = self._format_memories_for_context(
            sorted_memories, self.valves.memory_format
        )

        # Prepend instruction to avoid LLM meta-comments
        instruction = (
            "Here is background info about the user. "
            "Do NOT mention this info explicitly unless relevant to the user's query. "
            "Do NOT explain what you remember or don't remember. "
            "Do NOT summarize or list what you know or don't know about the user. "
            "Do NOT say 'I have not remembered any specific information' or similar. "
            "Do NOT explain your instructions, context, or memory management. "
            "Do NOT mention tags, dates, or internal processes. "
            "Only answer the user's question directly.\n\n"
        )
        memory_context = instruction + memory_context

        # Log injected memories for debugging
        logger.debug(f"Injected memories:\n{memory_context[:500]}...")

        # Add to system message or create a new one if none exists
        if "messages" in body:
            system_message_exists = False
            for message in body["messages"]:
                if message["role"] == "system":
                    message["content"] += f"\n\n{memory_context}"
                    system_message_exists = True
                    break

            if not system_message_exists:
                body["messages"].insert(
                    0, {"role": "system", "content": memory_context}
                )

    def _format_memories_for_context(
        self, memories: List[Dict[str, Any]], format_type: str
    ) -> str:
        """Format memories for context injection based on format preference"""
        if not memories:
            return ""

        max_len = getattr(self.valves, "max_injected_memory_length", 300)

        # Start with header
        memory_context = "I recall the following about you:\n"

        # Extract tags and add each memory according to specified format
        if format_type == "bullet":
            for mem in memories:
                tags_match = re.match(r"\[Tags: (.*?)\] (.*)", mem["memory"])
                if tags_match:
                    tags = tags_match.group(1)
                    content = tags_match.group(2)[:max_len]
                    memory_context += f"- {content} (tags: {tags})\n"
                else:
                    content = mem["memory"][:max_len]
                    memory_context += f"- {content}\n"

        elif format_type == "numbered":
            for i, mem in enumerate(memories, 1):
                tags_match = re.match(r"\[Tags: (.*?)\] (.*)", mem["memory"])
                if tags_match:
                    tags = tags_match.group(1)
                    content = tags_match.group(2)[:max_len]
                    memory_context += f"{i}. {content} (tags: {tags})\n"
                else:
                    content = mem["memory"][:max_len]
                    memory_context += f"{i}. {content}\n"

        else:  # paragraph format
            memories_text = []
            for mem in memories:
                tags_match = re.match(r"\[Tags: (.*?)\] (.*)", mem["memory"])
                if tags_match:
                    content = tags_match.group(2)[:max_len]
                    memories_text.append(content)
                else:
                    content = mem["memory"][:max_len]
                    memories_text.append(content)

            memory_context += f"{'. '.join(memories_text)}.\n"

        return memory_context

    async def _process_user_memories(
        self,
        user_message: str,
        user_id: str,
        event_emitter: Optional[
            Callable[[Any], Awaitable[None]]
        ] = None,  # Renamed for clarity
        show_status: bool = True,
        user_timezone: str = None,
        recent_chat_history: Optional[
            List[Dict[str, Any]]
        ] = None,  # Added this argument
    ) -> List[Dict[str, Any]]:
        """Process user message to extract and store memories

        Returns:
            List of stored memory operations
        """
        # Start timer
        start_time = time.perf_counter()

        # Reset stored memories and error message
        # This variable held identified memories, not saved ones. We'll get saved count from process_memories return.
        # self.stored_memories = [] # Remove or repurpose if needed elsewhere, currently unused after this point.
        self._error_message = None

        # Emit "processing memories" status if enabled
        if show_status:
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": "📝 Extracting potential new memories from your message…",
                        "done": False,
                    },
                },
            )

        # Debug logging for function entry
        logger.debug(
            f"Starting _process_user_memories for user {user_id} with message: {user_message[:50]}..."
        )

        # Get user valves
        user_valves = None
        try:
            user = Users.get_user_by_id(user_id)
            user_valves = self._get_user_valves(user)

            # Debug logging for user valves
            logger.debug(
                f"Retrieved user valves with memory enabled: {user_valves.enabled}"
            )

            if not user_valves.enabled:
                logger.info(f"Memory function disabled for user: {user_id}")
                if show_status:
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": "⏸️ Adaptive Memory is disabled in your settings – skipping memory save.",
                                "done": True,
                            },
                        },
                    )
                return []
        except Exception as e:
            logger.error(f"Error getting user valves: {e}")
            if show_status:
                await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ Unable to access memory settings – aborting memory save process.",
                            "done": True,
                        },
                    },
                )
            return []

        # Debug logging for memory identification start
        logger.debug(f"Starting memory identification for message: {user_message[:60]}...")

        # Step 1: Use LLM to identify memories in the message
        memories = []
        parse_error_occurred = False # Track if parsing failed
        try:
            # Get user's existing memories for context (optional - can also be None)
            existing_memories = None
            # If the LLM needs context of existing memories:
            try:
                existing_memories = await self._get_formatted_memories(user_id)
                logger.debug(
                    f"Retrieved {len(existing_memories)} existing memories for context"
                )
            except Exception as e:
                logger.warning(f"Could not get existing memories (continuing): {e}")

            # Process message to extract memory operations
            memories = await self.identify_memories(
                user_message,
                existing_memories=existing_memories,
                user_timezone=user_timezone,
            )

            # Debug logging after memory identification
            logger.debug(
                f"Memory identification complete. Found {len(memories)} potential memories"
            )

        except Exception as e:
            self.error_counters["llm_call_errors"] += 1
            logger.error(f"Error identifying memories: {e}\n{traceback.format_exc()}")
            self._error_message = f"llm_error: {str(e)[:50]}..." # Point 6: More specific error
            parse_error_occurred = True # Indicate identification failed
            if show_status:
                await self._safe_emit(
                    event_emitter,
                    {
                        "type": "status",
                        "data": {
                            "description": f"⚠️ Memory error: {str(e)}",
                            "done": True,
                        },
                    },
                )
            return []

        # Debug logging for filtering
        logger.debug("Starting memory filtering step...")

        # Step 2: Filter memories (apply blacklist/whitelist/trivia filtering)
        filtered_memories = []
        if memories:
            # Apply filters based on valves
            try:
                # Get filter configuration valves
                min_length = self.valves.min_memory_length
                blacklist = self.valves.blacklist_topics
                whitelist = self.valves.whitelist_keywords
                filter_trivia = self.valves.filter_trivia

                logger.debug(
                    f"Using filters: min_length={min_length}, blacklist={blacklist}, whitelist={whitelist}, filter_trivia={filter_trivia}"
                )

                # Default trivia patterns (common knowledge patterns)
                trivia_patterns = [
                    r"\b(when|what|who|where|how)\s+(is|was|were|are|do|does|did)\b",  # Common knowledge questions
                    r"\b(fact|facts)\b",  # Explicit facts
                    r"\b(in the year|in \d{4})\b",  # Historical dates
                    r"\b(country|countries|capital|continent|ocean|sea|river|mountain|planet)\b",  # Geographic/scientific
                    r"\b(population|inventor|invented|discovered|founder|founded|created|author|written|directed)\b",  # Attribution/creation
                ]

                # Known meta-request phrases
                meta_request_phrases = [
                    "remember this",
                    "make a note",
                    "don't forget",
                    "keep in mind",
                    "save this",
                    "add this to",
                    "log this",
                    "put this in",
                ]

                # Process each memory with filtering
                for memory in memories:
                    # Validate operation
                    if not self._validate_memory_operation(memory):
                        logger.debug(f"Invalid memory operation: {str(memory)}")
                        continue

                    # Extract content for filtering
                    content = memory.get("content", "").strip()

                    # Apply minimum length filter
                    if len(content) < min_length:
                        logger.debug(
                            f"Memory too short ({len(content)} < {min_length}): {content}"
                        )
                        continue

                    # Check if it's a meta-request
                    is_meta_request = False
                    for phrase in meta_request_phrases:
                        if phrase.lower() in content.lower():
                            is_meta_request = True
                            logger.debug(f"Meta-request detected: {content}")
                            break

                    if is_meta_request:
                        continue

                    # Check blacklist (if configured)
                    if blacklist:
                        is_blacklisted = False
                        for topic in blacklist.split(","):
                            topic = topic.strip().lower()
                            if topic and topic in content.lower():
                                # Check whitelist override
                                is_whitelisted = False
                                if whitelist:
                                    for keyword in whitelist.split(","):
                                        keyword = keyword.strip().lower()
                                        if keyword and keyword in content.lower():
                                            is_whitelisted = True
                                            logger.debug(
                                                f"Whitelisted term '{keyword}' found in blacklisted content"
                                            )
                                            break

                                if not is_whitelisted:
                                    is_blacklisted = True
                                    logger.debug(
                                        f"Blacklisted topic '{topic}' found: {content}"
                                    )
                                    break

                        if is_blacklisted:
                            continue

                    # Check trivia patterns (if enabled)
                    if filter_trivia:
                        is_trivia = False
                        for pattern in trivia_patterns:
                            if re.search(pattern, content.lower()):
                                logger.debug(
                                    f"Trivia pattern '{pattern}' matched: {content}"
                                )
                                is_trivia = True
                                break

                        if is_trivia:
                            # COMMENTED OUT: Secondary LLM classification to confirm if it's meta/trivia
                            # This was disabled due to Issue #9: Overly Aggressive Post-Extraction Filtering
                            """
                            try:
                                memory_classification_prompt = "Classify if this statement is META (about the conversation or a request to the AI) or FACT (actual information about the user). Respond with exactly ONE word - either META or FACT:\n\n"
                                classification = await self.query_llm_with_retry(memory_classification_prompt, content)
                                classification = classification.strip().upper()

                                logger.debug(f"LLM classification for potential trivia: '{classification}'")

                                # If it's actually a fact about the user despite matching trivia patterns, keep it
                                if "FACT" in classification:
                                    is_trivia = False
                                    logger.debug(f"LLM classified as FACT, keeping despite trivia pattern: {content}")
                            except Exception as e:
                                logger.warning(f"Error during memory classification (keeping memory): {e}")
                                is_trivia = False  # On error, don't filter
                            """

                        if is_trivia:
                            continue

                    # Memory passed all filters
                    filtered_memories.append(memory)
                    logger.debug(f"Memory passed all filters: {content}")

                logger.info(
                    f"Filtered memories: {len(filtered_memories)}/{len(memories)} passed"
                )
            except Exception as e:
                logger.error(f"Error filtering memories: {e}\n{traceback.format_exc()}")
                filtered_memories = (
                    memories  # On error, attempt to process all memories
                )

        # Debug logging after filtering
        logger.debug(f"After filtering: {len(filtered_memories)} memories remain")

        # If no memories to process after filtering, log and return
        if not filtered_memories: # Check if the list is empty
            # Point 5: Immediate-Save Shortcut for short preferences on parse error
            if (
                self.valves.enable_short_preference_shortcut
                and parse_error_occurred
                and len(user_message) <= 60
                and any(keyword in user_message.lower() for keyword in ["favorite", "love", "like", "enjoy"])
            ):
                logger.info("JSON parse failed, but applying short preference shortcut.")
                try:
                    shortcut_op = MemoryOperation(
                        operation="NEW",
                        content=user_message.strip(), # Save the raw message content
                        tags=["preference"] # Assume preference tag
                    )
                    await self._execute_memory_operation(shortcut_op, user) # Directly execute
                    logger.info(f"Successfully saved memory via shortcut: {user_message[:50]}...")
                    # Set a specific status message for this case
                    self._error_message = None # Clear parse error flag
                    # Since we bypassed normal processing, we need a result list for status reporting
                    saved_operations_list = [shortcut_op.model_dump()] # Use model_dump() for Pydantic v2+
                    # Skip the rest of the processing steps as we forced a save
                except Exception as shortcut_err:
                    logger.error(f"Error during short preference shortcut save: {shortcut_err}")
                    self._error_message = "shortcut_save_error"
                    saved_operations_list = [] # Indicate save failed
            else:
                # Normal case: No memories identified or filtered out, and no shortcut applied
                logger.info("No valid memories to process after filtering/identification.")
                if show_status and not self._error_message:
                    # Determine reason for no save
                    final_status_reason = self._error_message or "filtered_or_duplicate"
                    status_desc = f"ℹ️ Memory save skipped – {final_status_reason.replace('_', ' ')}."
                    await self._safe_emit(
                        event_emitter,
                        {
                            "type": "status",
                            "data": {
                                "description": status_desc,
                                "done": True,
                            },
                        },
                    )
                return [] # Return empty list as nothing was saved through normal path
        else:
           # We have filtered_memories, proceed with normal processing
           pass # Continue to Step 3

        # Step 3: Get current memories and handle max_total_memories limit
        try:
            current_memories_data = await self._get_formatted_memories(user_id)
            logger.debug(
                f"Retrieved {len(current_memories_data)} existing memories from database"
            )

            # If we'd exceed the maximum memories per user, apply pruning
            max_memories = self.valves.max_total_memories
            current_count = len(current_memories_data)
            new_count = len(filtered_memories) # Only count NEW operations towards limit for pruning decision
            
            if current_count + new_count > max_memories:
                to_remove = current_count + new_count - max_memories
                logger.info(
                    f"Memory limit ({max_memories}) would be exceeded. Need to prune {to_remove} memories."
                )
                
                memories_to_prune_ids = []
                
                # Choose pruning strategy based on valve
                strategy = self.valves.pruning_strategy
                logger.info(f"Applying pruning strategy: {strategy}")
                
                if strategy == "least_relevant":
                    try:
                        # Calculate relevance for all existing memories against the current user message
                        memories_with_relevance = []
                        # Re-use logic similar to get_relevant_memories but for *all* memories
                        
                        user_embedding = None
                        if self.embedding_model:
                            try:
                                user_embedding = self.embedding_model.encode(user_message, normalize_embeddings=True)
                            except Exception as e:
                                logger.warning(f"Could not encode user message for relevance pruning: {e}")

                        # Determine if we can use vectors or need LLM fallback (respecting valve)
                        can_use_vectors = user_embedding is not None
                        needs_llm = self.valves.use_llm_for_relevance

                        # --- Calculate Scores --- 
                        if not needs_llm and can_use_vectors:
                             # Vector-only relevance calculation
                            for mem_data in current_memories_data:
                                mem_id = mem_data.get("id")
                                mem_emb = self.memory_embeddings.get(mem_id)
                                # Ensure embedding exists or try to compute it
                                if mem_emb is None and self.embedding_model is not None:
                                    try:
                                        mem_text = mem_data.get("memory") or ""
                                        if mem_text:
                                            mem_emb = self.embedding_model.encode(mem_text, normalize_embeddings=True)
                                            self.memory_embeddings[mem_id] = mem_emb # Cache it
                                    except Exception as e:
                                        logger.warning(f"Failed to compute embedding for existing memory {mem_id}: {e}")
                                        mem_emb = None # Mark as failed
                                
                                if mem_emb is not None:
                                    sim_score = float(np.dot(user_embedding, mem_emb))
                                    memories_with_relevance.append({"id": mem_id, "relevance": sim_score})
                                else:
                                    # Assign low relevance if embedding fails
                                    memories_with_relevance.append({"id": mem_id, "relevance": 0.0})
                        elif needs_llm:
                            # LLM-based relevance calculation (simplified, no caching needed here)
                            # Prepare memories for LLM prompt
                            memory_strings_for_llm = [
                                f"ID: {mem['id']}, CONTENT: {mem['memory']}" 
                                for mem in current_memories_data
                            ]
                            system_prompt = self.valves.memory_relevance_prompt
                            llm_user_prompt = f"""Current user message: "{user_message}"

Available memories:
{json.dumps(memory_strings_for_llm)}

Rate the relevance of EACH memory to the current user message."""
                            
                            try:
                                llm_response_text = await self.query_llm_with_retry(system_prompt, llm_user_prompt)
                                llm_relevance_results = self._extract_and_parse_json(llm_response_text)
                                
                                if isinstance(llm_relevance_results, list):
                                    # Map results back to IDs
                                    llm_scores = {item.get("id"): item.get("relevance", 0.0) for item in llm_relevance_results if isinstance(item, dict)}
                                    for mem_data in current_memories_data:
                                        mem_id = mem_data.get("id")
                                        score = llm_scores.get(mem_id, 0.0) # Default to 0 if LLM missed it
                                        memories_with_relevance.append({"id": mem_id, "relevance": score})
                                else:
                                    logger.warning("LLM relevance check for pruning failed to return valid list. Pruning might default to FIFO.")
                                    # Fallback: assign 0 relevance to all, effectively making it FIFO-like for this run
                                    memories_with_relevance = [{"id": m["id"], "relevance": 0.0} for m in current_memories_data]
                            except Exception as llm_err:
                                logger.error(f"Error during LLM relevance check for pruning: {llm_err}")
                                memories_with_relevance = [{"id": m["id"], "relevance": 0.0} for m in current_memories_data]
                        else: # Cannot use vectors and LLM not enabled - default to FIFO-like
                             logger.warning("Cannot determine relevance for pruning (no embeddings/LLM). Pruning will be FIFO-like.")
                             memories_with_relevance = [{"id": m["id"], "relevance": 0.0} for m in current_memories_data]

                        # --- Sort and Select for Pruning ---                     
                        # Sort by relevance ascending (lowest first)
                        memories_with_relevance.sort(key=lambda x: x.get("relevance", 0.0))
                        
                        # Select the IDs of the least relevant memories to remove
                        memories_to_prune_ids = [mem["id"] for mem in memories_with_relevance[:to_remove]]
                        logger.info(f"Identified {len(memories_to_prune_ids)} least relevant memories for pruning.")
                        
                    except Exception as relevance_err:
                        logger.error(f"Error calculating relevance for pruning, falling back to FIFO: {relevance_err}")
                        # Fallback to FIFO on any error during relevance calculation
                        strategy = "fifo"
                        
                # Default or fallback FIFO strategy
                if strategy == "fifo":
                    # Sort by timestamp ascending (oldest first)
                    # Make sure timestamp exists, fallback to a very old date if not
                    default_date = datetime.min.replace(tzinfo=timezone.utc)
                    sorted_memories = sorted(
                        current_memories_data, 
                        key=lambda x: x.get("created_at", default_date)
                    )
                    memories_to_prune_ids = [mem["id"] for mem in sorted_memories[:to_remove]]
                    logger.info(f"Identified {len(memories_to_prune_ids)} oldest memories (FIFO) for pruning.")

                # Execute pruning if IDs were identified
                if memories_to_prune_ids:
                    pruned_count = 0
                    for memory_id_to_delete in memories_to_prune_ids:
                        try:
                            delete_op = MemoryOperation(operation="DELETE", id=memory_id_to_delete)
                            await self._execute_memory_operation(delete_op, user)
                            pruned_count += 1
                        except Exception as e:
                            logger.error(f"Error pruning memory {memory_id_to_delete}: {e}")
                    logger.info(f"Successfully pruned {pruned_count} memories.")
                else:
                    logger.warning("Pruning needed but no memory IDs identified for deletion.")
                    
        except Exception as e:
            logger.error(
                f"Error handling max_total_memories: {e}\n{traceback.format_exc()}"
            )
            # Continue processing the new memories even if pruning failed

        # Debug logging before processing operations
        logger.debug("Beginning to process memory operations...")

        # Step 4: Process the filtered memories
        processing_error: Optional[Exception] = None
        try:
            # process_memories now returns the list of successfully executed operations
            logger.debug(f"Calling process_memories with {len(filtered_memories)} items: {str(filtered_memories)}") # Log the exact list being passed
            saved_operations_list = await self.process_memories(
                filtered_memories, user_id
            )
            logger.debug(
                f"Memory saving attempt complete, returned {len(saved_operations_list)} successfully saved operations."
            )
        except Exception as e:
            processing_error = e
            logger.error(f"Error processing memories: {e}\n{traceback.format_exc()}")
            self._error_message = f"processing_error: {str(e)[:50]}..." # Point 6: More specific error

        # Debug confirmation logs
        if saved_operations_list:
            logger.info(
                f"Successfully processed and saved {len(saved_operations_list)} memories"
            )
        elif processing_error:
            logger.warning(
                f"Memory processing failed due to an error: {processing_error}"
            )
        else:
            logger.warning(
                "Memory processing finished, but no memories were saved (potentially due to duplicates or errors during save).)"
            )

        # Emit completion status
        if show_status:
            elapsed_time = time.perf_counter() - start_time
            # Base the status on the actual saved operations list
            saved_count = len(saved_operations_list)  # Directly use length of result
            if saved_count > 0:
                # Check if it was the shortcut save
                if any(op.get("content") == user_message.strip() for op in saved_operations_list):
                     status_desc = f"✅ Saved 1 memory via shortcut ({elapsed_time:.2f}s)"
                else:
                    plural = "memory" if saved_count == 1 else "memories"
                    status_desc = f"✅ Added {saved_count} new {plural} to your memory bank ({elapsed_time:.2f}s)"
            else:
                # Build smarter status based on duplicate counters
                if getattr(self, "_duplicate_refreshed", 0):
                    status_desc = f"✅ Memory refreshed (duplicate confirmed) ({elapsed_time:.2f}s)"
                elif getattr(self, "_duplicate_skipped", 0):
                    status_desc = f"✅ Preference already saved – duplicate ignored ({elapsed_time:.2f}s)"
                else:
                    final_status_reason = self._error_message or "filtered_or_duplicate"
                    status_desc = f"⚠️ Memory save skipped – {final_status_reason.replace('_', ' ')} ({elapsed_time:.2f}s)"
            await self._safe_emit(
                event_emitter,
                {
                    "type": "status",
                    "data": {
                        "description": status_desc,
                        "done": True,
                    },
                },
            )

        # Return the list of operations that were actually saved
        return saved_operations_list

    async def identify_memories(
        self,
        input_text: str,
        existing_memories: Optional[List[Dict[str, Any]]] = None,
        user_timezone: str = None,
    ) -> List[Dict[str, Any]]:
        """Identify potential memories from text using LLM"""
        logger.debug(
            f"Starting memory identification from input text: {input_text[:50]}..."
        )

        # Remove <details> blocks that may interfere with processing
        input_text = re.sub(r"<details>.*?</details>", "", input_text, flags=re.DOTALL)

        # Clean up and prepare the input
        clean_input = input_text.strip()
        logger.debug(f"Cleaned input text length: {len(clean_input)}")

        # Prepare the system prompt
        try:
            # Get the base prompt template
            memory_prompt = self.valves.memory_identification_prompt

            # Add datetime context
            now_str = self.get_formatted_datetime(user_timezone)
            datetime_context = f"Current datetime: {now_str}"

            # Add memory categories context based on enabled flags
            categories = []
            if self.valves.enable_identity_memories:
                categories.append("identity")
            if self.valves.enable_behavior_memories:
                categories.append("behavior")
            if self.valves.enable_preference_memories:
                categories.append("preference")
            if self.valves.enable_goal_memories:
                categories.append("goal")
            if self.valves.enable_relationship_memories:
                categories.append("relationship")
            if self.valves.enable_possession_memories:
                categories.append("possession")

            categories_str = ", ".join(categories)

            # Add existing memories context if provided
            existing_memories_str = ""
            if existing_memories and len(existing_memories) > 0:
                existing_memories_str = "Existing memories:\n"
                for i, mem in enumerate(
                    existing_memories[:5]
                ):  # Limit to 5 recent memories
                    existing_memories_str += f"- {mem.get('content', 'Unknown')}\n"

            # Combine all context
            context = f"{datetime_context}\nEnabled categories: {categories_str}\n{existing_memories_str}"

            # Log the components of the prompt
            logger.debug(f"Memory identification context: {context}")

            # Create the final system prompt with context
            system_prompt = f"{memory_prompt}\n\nCONTEXT:\n{context}"

            logger.debug(
                f"Final memory identification system prompt length: {len(system_prompt)}"
            )
        except Exception as e:
            logger.error(f"Error building memory identification prompt: {e}")
            system_prompt = self.valves.memory_identification_prompt

        # Call LLM to identify memories
        start_time = time.time()
        logger.debug(
            f"Calling LLM for memory identification with provider: {self.valves.llm_provider_type}, model: {self.valves.llm_model_name}"
        )

        try:
            # Construct the user prompt with few-shot examples
            user_prompt = f"""Analyze the following user message and extract relevant memories:
>>> USER MESSAGE START <<<
+{clean_input}
>>> USER MESSAGE END <<<

--- EXAMPLES OF DESIRED OUTPUT FORMAT ---
Example 1 Input: "I really love pizza, especially pepperoni."
Example 1 Output: [{{"operation": "NEW", "content": "User loves pizza, especially pepperoni", "tags": ["preference"]}}]

Example 2 Input: "What's the weather like today?"
Example 2 Output: []

Example 3 Input: "My sister Jane is visiting next week. I should buy her flowers."
Example 3 Output: [{{"operation": "NEW", "content": "User has a sister named Jane", "tags": ["relationship"]}}, {{"operation": "NEW", "content": "User's sister Jane is visiting next week", "tags": ["relationship"]}}]
--- END EXAMPLES ---

Produce ONLY the JSON array output for the user message above, adhering strictly to the format requirements outlined in the system prompt.
"""
            # Note: Doubled curly braces {{ }} are used to escape them within the f-string for the JSON examples.

            # Log the user prompt structure for debugging
            logger.debug(
                f"User prompt structure with few-shot examples:\n{user_prompt[:500]}..."
            )  # Log first 500 chars

            # Call LLM with the modified prompts
            llm_response = await self.query_llm_with_retry(
                system_prompt, user_prompt
            )  # Pass the new user_prompt
            elapsed = time.time() - start_time
            logger.debug(
                f"LLM memory identification completed in {elapsed:.2f}s, response length: {len(llm_response)}"
            )
            logger.debug(f"LLM raw response for memory identification: {llm_response}")

            # Parse the response (assumes JSON format)
            result = self._extract_and_parse_json(llm_response)
            logger.debug(
                f"Parsed result type: {type(result)}, content: {str(result)[:500]}"
            )

            # Check if we got a dict instead of a list (common LLM error)
            if isinstance(result, dict):
                logger.warning(
                    "LLM returned a JSON object instead of an array. Attempting conversion."
                )
                result = self._convert_dict_to_memory_operations(result)
                logger.debug(f"Converted dict to {len(result)} memory operations")

            # Check for empty result
            if not result:
                logger.warning("No memory operations identified by LLM")
                return []

            # Validate operations format
            valid_operations = []
            invalid_count = 0

            if isinstance(result, list):
                for op in result:
                    if self._validate_memory_operation(op):
                        valid_operations.append(op)
                    else:
                        invalid_count += 1

                logger.debug(
                    f"Identified {len(valid_operations)} valid memory operations, {invalid_count} invalid"
                )
                return valid_operations
            else:
                logger.error(
                    f"LLM returned invalid format (neither list nor dict): {type(result)}"
                )
                self._error_message = (
                    "LLM returned invalid format. Expected JSON array."
                )
                return []

        except Exception as e:
            logger.error(
                f"Error in memory identification: {e}\n{traceback.format_exc()}"
            )
            self.error_counters["llm_call_errors"] += 1
            self._error_message = f"Memory identification error: {str(e)}"
            return []

    def _validate_memory_operation(self, op: Dict[str, Any]) -> bool:
        """Validate memory operation format and required fields"""
        if not isinstance(op, dict):
            logger.warning(f"Invalid memory operation format (not a dict): {op}")
            return False

        # Check if operation field exists, if not try to infer it
        if "operation" not in op:
            # Look for typical patterns to guess the operation type
            if any(k.lower() == "operation" for k in op.keys()):
                # Operation may be under a different case
                for k, v in op.items():
                    if k.lower() == "operation" and isinstance(v, str):
                        op["operation"] = v
                        break

            # Look for operation in original format but in wrong place
            elif isinstance(op, dict) and any(
                v in ["NEW", "UPDATE", "DELETE"] for v in op.values()
            ):
                for k, v in op.items():
                    if v in ["NEW", "UPDATE", "DELETE"]:
                        op["operation"] = v
                        # Remove the old key if it's not "operation"
                        if k != "operation":
                            op.pop(k, None)
                        break

            # Default based on presence of fields
            elif "id" in op and "content" in op:
                # Default to UPDATE if we have both id and content
                op["operation"] = "UPDATE"
            elif "content" in op:
                # Default to NEW if we only have content
                op["operation"] = "NEW"
            else:
                logger.warning(f"Cannot determine operation type for: {op}")
                return False

        # Normalize operation to uppercase
        if isinstance(op["operation"], str):
            op["operation"] = op["operation"].upper()

        if op["operation"] not in ["NEW", "UPDATE", "DELETE"]:
            logger.warning(f"Invalid operation type: {op['operation']}")
            return False

        if op["operation"] in ["UPDATE", "DELETE"] and "id" not in op:
            logger.warning(f"Missing ID for {op['operation']} operation: {op}")
            return False

        if op["operation"] in ["NEW", "UPDATE"] and "content" not in op:
            logger.warning(f"Missing content for {op['operation']} operation: {op}")
            return False

        # Tags are optional but should be a list if present
        if "tags" in op and not isinstance(op["tags"], list):
            # Try to fix if it's a string
            if isinstance(op["tags"], str):
                try:
                    # See if it's a JSON string
                    parsed_tags = json.loads(op["tags"])
                    if isinstance(parsed_tags, list):
                        op["tags"] = parsed_tags
                    else:
                        # If it parsed but isn't a list, handle that case
                        op["tags"] = [str(parsed_tags)]
                except json.JSONDecodeError:
                    # Split by comma if it looks like a comma-separated list
                    if "," in op["tags"]:
                        op["tags"] = [tag.strip() for tag in op["tags"].split(",")]
                    else:
                        # Just make it a single-item list
                        op["tags"] = [op["tags"]]
            else:
                logger.warning(
                    f"Invalid tags format, not a list or string: {op['tags']}"
                )
                op["tags"] = []  # Default to empty list

        return True

    def _extract_and_parse_json(self, text: str) -> Union[List, Dict, None]:
        """Extract and parse JSON from text, handling common LLM response issues"""
        skip_reason = None # For granular status updates
        if not text:
            logger.warning("Empty text provided to JSON parser")
            return None

        # Clean the text
        text = text.strip()

        # Optional: Strip leading/trailing non-JSON text (Point 1)
        if self.valves.enable_json_stripping:
            start_brace = text.find('{')
            start_bracket = text.find('[')

            if start_bracket != -1 and (start_brace == -1 or start_bracket < start_brace):
                # Looks like an array first
                start_index = start_bracket
                end_index = text.rfind(']')
            elif start_brace != -1:
                # Looks like an object first
                start_index = start_brace
                end_index = text.rfind('}')
            else:
                start_index = -1
                end_index = -1

            if start_index != -1 and end_index != -1 and end_index > start_index:
                original_length = len(text)
                text = text[start_index : end_index + 1]
                if len(text) < original_length:
                    logger.debug(f"Stripped non-JSON wrappers. Original length: {original_length}, New length: {len(text)}")
            else:
                 logger.debug("JSON stripping enabled, but no clear start/end brackets/braces found.")

        # Log first 100 characters of the text for debugging
        logger.debug(f"Attempting to parse JSON from: {text[:100]}...")

        # Try direct parsing first (most efficient if valid)
        try:
            parsed = json.loads(text)
            logger.debug("Successfully parsed JSON directly")

            # Handle empty object case immediately after successful parsing
            if parsed == {} or parsed == []:
                logger.info(
                    "LLM returned empty object/array, treating as empty memory list"
                )
                return []

            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Direct JSON parsing failed: {e}")
            # Continue with extraction methods

        # If direct parsing failed, try to extract JSON from markdown code blocks
        # Common LLM formatting: ```json [...] ```
        code_block_pattern = r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```"
        matches = re.findall(code_block_pattern, text)

        if matches:
            logger.debug(f"Found {len(matches)} JSON code blocks to try parsing")
            for i, match in enumerate(matches):
                try:
                    parsed = json.loads(match)
                    logger.debug(f"Successfully parsed JSON from code block {i+1}")

                    # Handle empty results
                    if parsed == {} or parsed == []:
                        continue

                    return parsed
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from code block {i+1}: {e}")

        # Try to find JSON directly in the text (without code block markers)
        # Common pattern: clear JSON object/array in the response
        direct_json_patterns = [
            r"\[\s*\{[\s\S]*?\}\s*\]",  # Array of objects (non-greedy match)
            r"\[\s*\"[\s\S]*?\"\s*\]",  # Array of strings (non-greedy match)
            r"\[\s*\]",  # Empty array
            r"\{[\s\S]*?\}",  # Single object (non-greedy match)
        ]

        for pattern in direct_json_patterns:
            matches = re.findall(pattern, text)
            if matches:
                logger.debug(
                    f"Found {len(matches)} potential direct JSON matches to try parsing"
                )
                for i, match in enumerate(matches):
                    try:
                        parsed = json.loads(match)
                        logger.debug(f"Successfully parsed direct JSON match {i+1}")

                        # Handle empty results
                        if parsed == {} or parsed == []:
                            continue

                        return parsed
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse direct JSON match {i+1}: {e}")

        # Handle Ollama's response format - sometimes it adds quotes around the JSON
        if text.startswith('"') and text.endswith('"'):
            # Try to unescape the quoted JSON
            try:
                unescaped = json.loads(text)  # This will interpret as a JSON string
                if isinstance(unescaped, str):
                    try:
                        parsed = json.loads(
                            unescaped
                        )  # Then parse the content of that string
                        logger.debug("Successfully parsed quoted JSON from Ollama")

                        # Handle empty results
                        if parsed == {} or parsed == []:
                            logger.info("Ollama returned empty quoted object/array")
                            return []

                        return parsed
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse unescaped JSON: {e}")
            except json.JSONDecodeError:
                pass  # Not a valid JSON string

        # Last resort: Try to extract any JSON-like structure with regex
        # This is risky but might recover some results
        json_pattern = r"(\[.*?\]|\{.*?\})"
        matches = re.findall(json_pattern, text)

        if matches:
            logger.debug(f"Found {len(matches)} potential JSON-like structures")
            for i, match in enumerate(matches):
                # Only try to parse if it looks promising (starts/ends with proper brackets)
                if (match.startswith("[") and match.endswith("]")) or (
                    match.startswith("{") and match.endswith("}")
                ):
                    try:
                        parsed = json.loads(match)
                        logger.debug(f"Successfully parsed JSON-like structure {i+1}")

                        # Handle empty results
                        if parsed == {} or parsed == []:
                            continue

                        return parsed
                    except json.JSONDecodeError:
                        pass  # Silently fail for this last-resort attempt

        # If we reach here, we couldn't extract any valid JSON
        self.error_counters["json_parse_errors"] += 1
        self._error_message = "json_parse_error" # Point 6: Set specific error reason
        logger.error("Failed to extract valid JSON from LLM response")

        # Log the entire text for thorough debugging when JSON parsing fails
        logger.debug(f"Full text that failed JSON parsing: {text}")

        # One last simple check: if the cleaned text *contains* an empty array token
        # we interpret that as an explicit signal of "no memories".
        if "[]" in text.replace(" ", ""):
            logger.info("Detected '[]' token in LLM response after exhaustive parsing attempts. Treating as empty list.")
            return []

        return None

    def _calculate_memory_similarity(self, memory1: str, memory2: str) -> float:
        """
        Calculate similarity between two memory contents using a more robust method.
        Returns a score between 0.0 (completely different) and 1.0 (identical).
        """
        if not memory1 or not memory2:
            return 0.0

        # Clean the memories - remove tags and normalize
        memory1_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory1).lower().strip()
        memory2_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory2).lower().strip()

        # Handle exact matches quickly
        if memory1_clean == memory2_clean:
            return 1.0

        # Handle near-duplicates with same meaning but minor differences
        # Split into words and compare overlap
        words1 = set(re.findall(r"\b\w+\b", memory1_clean))
        words2 = set(re.findall(r"\b\w+\b", memory2_clean))

        if not words1 or not words2:
            return 0.0

        # Calculate Jaccard similarity for word overlap
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0.0

        # Use sequence matcher for more precise comparison
        seq_similarity = SequenceMatcher(None, memory1_clean, memory2_clean).ratio()

        # Combine both metrics, weighting sequence similarity higher
        combined_similarity = (0.4 * jaccard) + (0.6 * seq_similarity)

        return combined_similarity
        
    async def _calculate_embedding_similarity(self, memory1: str, memory2: str) -> float:
        """
        Calculate semantic similarity between two memory contents using embeddings.
        Returns a score between 0.0 (completely different) and 1.0 (identical).
        
        This method uses the sentence transformer model to generate embeddings
        and calculates cosine similarity for more accurate semantic matching.
        """
        if not memory1 or not memory2:
            return 0.0
            
        # Clean the memories - remove tags and normalize
        memory1_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory1).lower().strip()
        memory2_clean = re.sub(r"\[Tags:.*?\]\s*", "", memory2).lower().strip()
        
        # Handle exact matches quickly
        if memory1_clean == memory2_clean:
            return 1.0
            
        try:
            # Check if embedding model is available
            if self.embedding_model is None:
                logger.warning("Embedding model not available for similarity calculation. Falling back to text-based similarity.")
                return self._calculate_memory_similarity(memory1, memory2)
                
            # Generate embeddings for both memories
            mem1_embedding = self.embedding_model.encode(memory1_clean, normalize_embeddings=True)
            mem2_embedding = self.embedding_model.encode(memory2_clean, normalize_embeddings=True)
            
            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = float(np.dot(mem1_embedding, mem2_embedding))
            
            return similarity
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}\n{traceback.format_exc()}")
            # Fall back to text-based similarity on error
            logger.info("Falling back to text-based similarity due to error.")
            return self._calculate_memory_similarity(memory1, memory2)

    async def get_relevant_memories(
        self, current_message: str, user_id: str, user_timezone: str = None
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to the current context"""
        import time

        start = time.perf_counter()
        try:
            # Get all memories for the user
            existing_memories = await self._get_formatted_memories(user_id)

            if not existing_memories:
                logger.debug("No existing memories found for relevance assessment")
                return []

            # --- Local vector similarity filtering ---
            vector_similarities = []
            user_embedding = None # Initialize to handle potential errors
            try:
                if self.embedding_model:
                    user_embedding = self.embedding_model.encode(
                        current_message, normalize_embeddings=True
                    )
                else:
                    logger.warning("Embedding model not available for user message encoding.")
                    # If no embedding model, cannot use vector similarity, fallback depends on config
                    if not self.valves.use_llm_for_relevance:
                         logger.warning("Cannot calculate relevance without embedding model or LLM fallback.")
                         return [] # Cannot proceed without either method

            except Exception as e:
                self.error_counters["embedding_errors"] += 1
                logger.error(
                    f"Error computing embedding for user message: {e}\n{traceback.format_exc()}" # Removed extra backslash
                )
                # Decide fallback based on config
                if not self.valves.use_llm_for_relevance:
                    logger.warning("Cannot calculate relevance due to embedding error and no LLM fallback.")
                    return [] # Cannot proceed

            if user_embedding is not None:
                # Calculate vector similarities only if user embedding was successful
                for mem in existing_memories:
                    mem_id = mem.get("id")
                    # Ensure embedding exists in our cache for this memory
                    mem_emb = self.memory_embeddings.get(mem_id)
                    # Lazily compute and cache the memory embedding if not present
                    if mem_emb is None and self.embedding_model is not None:
                        try:
                            mem_text = mem.get("memory") or ""
                            if mem_text:
                                mem_emb = self.embedding_model.encode(
                                    mem_text, normalize_embeddings=True
                                )
                                # Cache for future similarity checks
                                self.memory_embeddings[mem_id] = mem_emb
                        except Exception as e:
                            logger.warning(
                                f"Error computing embedding for memory {mem_id}: {e}"
                            )

                    if mem_emb is not None:
                        try:
                            # Cosine similarity (since embeddings are normalized)
                            sim = float(np.dot(user_embedding, mem_emb))
                            vector_similarities.append((sim, mem))
                        except Exception as e:
                            logger.warning(
                                f"Error calculating similarity for memory {mem_id}: {e}"
                            )
                            continue  # Skip this memory if calculation fails
                        else:
                            logger.debug(
                                f"No embedding available for memory {mem_id} even after attempted computation."
                            )
                    else:
                        logger.debug(
                            f"No embedding available for memory {mem_id} even after attempted computation."
                        )

                # Sort by similarity descending
                vector_similarities.sort(reverse=True, key=lambda x: x[0])

                # Filter by threshold
                sim_threshold = self.valves.vector_similarity_threshold
                top_n = self.valves.top_n_memories # Note: This top_n is applied BEFORE deciding on LLM/Vector scoring.
                filtered_by_vector = [mem for sim, mem in vector_similarities if sim >= sim_threshold][:top_n]
                logger.info(
                    f"Vector filter selected {len(filtered_by_vector)} of {len(existing_memories)} memories (Threshold: {sim_threshold}, Top N: {top_n})"
                )
            else:
                 # If user_embedding failed and LLM fallback is disabled, we already returned.
                 # If LLM fallback is enabled, proceed with all existing memories for LLM relevance check.
                 logger.warning("User embedding failed, proceeding with all memories for potential LLM check.")
                 filtered_by_vector = existing_memories # Pass all memories to LLM check if enabled


            # --- Decide Relevance Method ---
            if not self.valves.use_llm_for_relevance:
                # --- Use Vector Similarity Scores Directly ---
                logger.info("Using vector similarity directly for relevance scoring (LLM call skipped).")
                relevant_memories = []
                final_relevance_threshold = self.valves.vector_similarity_threshold # Use the same threshold we applied during the initial vector filter so we don't discard everything.

                # Use the already calculated and sorted vector similarities
                for sim_score, mem in vector_similarities: # Iterate through the originally sorted list
                    if sim_score >= final_relevance_threshold:
                         # Check if this memory was part of the top_n initially filtered by vector
                         # This ensures we respect the vector_similarity_threshold AND top_n_memories filter first
                         if any(filtered_mem['id'] == mem['id'] for filtered_mem in filtered_by_vector):
                            relevant_memories.append(
                                {"id": mem["id"], "memory": mem["memory"], "relevance": sim_score} # Use vector score as relevance
                            )

                # Sort again just to be sure (though vector_similarities was already sorted)
                relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)

                # Limit to configured number
                final_top_n = self.valves.related_memories_n
                logger.info(
                    f"Found {len(relevant_memories)} relevant memories using vector similarity >= {final_relevance_threshold}"
                )
                logger.info(f"Memory retrieval (vector only) took {time.perf_counter() - start:.2f}s")
                return relevant_memories[:final_top_n]

            else:
                # --- Use LLM for Relevance Scoring (Existing Logic) ---
                logger.info("Proceeding with LLM call for relevance scoring.")
                memories_for_llm = filtered_by_vector # Use the vector-filtered list

                if not memories_for_llm:
                     logger.debug("No memories passed vector filter for LLM relevance check.")
                     return []

                # Build the prompt for LLM
                memory_strings = []
                for mem in memories_for_llm:
                    memory_strings.append(f"ID: {mem['id']}, CONTENT: {mem['memory']}")

                system_prompt = self.valves.memory_relevance_prompt
                user_prompt = f"""Current user message: "{current_message}"

Available memories (pre-filtered by vector similarity):
{json.dumps(memory_strings)}

Rate the relevance of EACH memory to the current user message based *only* on the provided content and message context.""" # Removed escaping backslashes

                # Add current datetime for context
                current_datetime = self.get_formatted_datetime(user_timezone)
                user_prompt += f"""

Current datetime: {current_datetime.strftime('%A, %B %d, %Y %H:%M:%S')} ({current_datetime.tzinfo})""" # Removed escaping backslashes

                # Check cache or call LLM for relevance score
                import time as time_module

                now = time_module.time()
                ttl_seconds = self.valves.cache_ttl_seconds

                relevance_data = []
                uncached_memories = [] # Memories needing LLM call
                uncached_ids = set() # Track IDs needing LLM call

                # Check cache first
                if user_embedding is not None: # Can only use cache if we have user embedding
                    for mem in memories_for_llm:
                        mem_id = mem.get("id")
                        mem_emb = self.memory_embeddings.get(mem_id)
                        if mem_emb is None:
                             # If memory embedding is missing, cannot use cache, must call LLM
                             if mem_id not in uncached_ids:
                                 uncached_memories.append(mem)
                                 uncached_ids.add(mem_id)
                             continue

                        key = hash((user_embedding.tobytes(), mem_emb.tobytes()))
                        cached = self.relevance_cache.get(key)
                        if cached:
                            score, ts = cached
                            if now - ts < ttl_seconds:
                                logger.info(f"Cache hit for memory {mem_id} (LLM relevance)")
                                relevance_data.append(
                                    {"memory": mem["memory"], "id": mem_id, "relevance": score}
                                )
                                continue  # use cached score

                        # Cache miss or expired, add to uncached list if not already there
                        if mem_id not in uncached_ids:
                             uncached_memories.append(mem)
                             uncached_ids.add(mem_id)
                else:
                     # No user embedding, cannot use cache, all need LLM call
                     logger.warning("Cannot use relevance cache as user embedding failed.")
                     uncached_memories = memories_for_llm # Send all vector-filtered memories to LLM


                # If any uncached memories, call LLM
                if uncached_memories:
                    logger.info(f"Calling LLM for relevance on {len(uncached_memories)} uncached memories.")
                    # Build prompt with only uncached memories
                    uncached_memory_strings = [
                        f"ID: {mem['id']}, CONTENT: {mem['memory']}"
                        for mem in uncached_memories
                    ]
                    # Reuse system_prompt, construct user_prompt specifically for uncached items
                    uncached_user_prompt = f"""Current user message: "{current_message}"

Available memories (evaluate relevance for these specific IDs):
{json.dumps(uncached_memory_strings)}

Rate the relevance of EACH listed memory to the current user message based *only* on the provided content and message context.""" # Removed escaping backslashes
                    current_datetime = self.get_formatted_datetime(user_timezone)
                    uncached_user_prompt += f"""

Current datetime: {current_datetime.strftime('%A, %B %d, %Y %H:%M:%S')} ({current_datetime.tzinfo})""" # Removed escaping backslashes

                    llm_response_text = await self.query_llm_with_retry(
                        system_prompt, uncached_user_prompt # Use the specific uncached prompt
                    )

                    if not llm_response_text or llm_response_text.startswith("Error:"):
                        if llm_response_text:
                            logger.error(
                                f"Error from LLM during memory relevance: {llm_response_text}"
                            )
                        # If LLM fails, we might return empty or potentially fall back
                        # For now, return empty to indicate failure
                        return []

                    # Parse the LLM response for the uncached items
                    llm_relevance_results = self._extract_and_parse_json(
                        llm_response_text
                    )

                    if not llm_relevance_results or not isinstance(
                        llm_relevance_results, list
                    ):
                        logger.warning("Failed to parse relevance data from LLM response for uncached items.")
                        # Decide how to handle partial failure - return only cached? or empty?
                        # Returning only cached for now
                    else:
                         # Process successful LLM results
                         for item in llm_relevance_results:
                            mem_id = item.get("id")
                            score = item.get("relevance")
                            mem_text = item.get("memory") # Use memory text from LLM response if available
                            if mem_id and isinstance(score, (int, float)):
                                relevance_data.append(
                                    {"memory": mem_text or f"Content for {mem_id}", # Fallback if memory text missing
                                     "id": mem_id,
                                     "relevance": score}
                                )
                                # Save to cache if possible
                                if user_embedding is not None:
                                    mem_emb = self.memory_embeddings.get(mem_id)
                                    if mem_emb is not None:
                                        key = hash((user_embedding.tobytes(), mem_emb.tobytes()))
                                        self.relevance_cache[key] = (score, now)
                                    else:
                                         logger.debug(f"Cannot cache relevance for {mem_id}, embedding missing.")
                            else:
                                logger.warning(f"Invalid item format in LLM relevance response: {item}")


                # Combine cached and newly fetched results, filter by relevance threshold
                final_relevant_memories = []
                final_relevance_threshold = self.valves.vector_similarity_threshold # Use the same threshold we applied during the initial vector filter so we don't discard everything.

                seen_ids = set() # Ensure unique IDs in final list
                for item in relevance_data:
                    if not isinstance(item, dict): continue # Skip invalid entries

                    memory_content = item.get("memory")
                    relevance_score = item.get("relevance")
                    mem_id = item.get("id")

                    if memory_content and isinstance(relevance_score, (int, float)) and mem_id:
                        if relevance_score >= final_relevance_threshold and mem_id not in seen_ids:
                            final_relevant_memories.append(
                                {"id": mem_id, "memory": memory_content, "relevance": relevance_score}
                            )
                            seen_ids.add(mem_id)

                # Sort final list by relevance (descending)
                final_relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)

                # Limit to configured number
                final_top_n = self.valves.related_memories_n
                logger.info(
                    f"Found {len(final_relevant_memories)} relevant memories using LLM score >= {final_relevance_threshold}"
                )
                logger.info(f"Memory retrieval (LLM scoring) took {time.perf_counter() - start:.2f}s")
                return final_relevant_memories[:final_top_n]

        except Exception as e:
            logger.error(
                f"Error getting relevant memories: {e}\n{traceback.format_exc()}" # Removed extra backslash
            )
            return []

    async def process_memories(
        self, memories: List[Dict[str, Any]], user_id: str
    ) -> List[Dict[str, Any]]:  # Return list of successfully processed operations
        """Process memory operations"""
        successfully_saved_ops = []
        try:
            user = Users.get_user_by_id(user_id)
            if not user:
                logger.error(f"User not found: {user_id}")
                return []

            # Get existing memories for deduplication
            existing_memories = []
            if self.valves.deduplicate_memories:
                existing_memories = await self._get_formatted_memories(user_id)

            logger.debug(f"Processing {len(memories)} memory operations")

            # First filter for duplicates if enabled
            processed_memories = []
            if self.valves.deduplicate_memories and existing_memories:
                # Store all existing contents for quick lookup
                existing_contents = []
                for mem in existing_memories:
                    existing_contents.append(mem["memory"])

                logger.debug(f"[DEDUPE] Existing memories being checked against: {existing_contents}")

                # Check if we should use embedding-based similarity for deduplication
                use_embeddings = self.valves.use_embeddings_for_deduplication
                logger.debug(f"Using {'embedding-based' if use_embeddings else 'text-based'} similarity for deduplication.")

                # Check each new memory against existing ones
                for new_memory_idx, memory_dict in enumerate(memories):
                    if memory_dict["operation"] == "NEW":
                        logger.debug(f"[DEDUPE CHECK {new_memory_idx+1}/{len(memories)}] Processing NEW memory: {memory_dict}") # LOG START
                        # Format the memory content
                        operation = MemoryOperation(**memory_dict)
                        formatted_content = self._format_memory_content(operation)

                        # --- BYPASS: Skip dedup for short preference statements ---
                        if (
                            self.valves.enable_short_preference_shortcut
                            and len(formatted_content) <= self.valves.short_preference_no_dedupe_length
                        ):
                            pref_kwds = [kw.strip() for kw in self.valves.preference_keywords_no_dedupe.split(',') if kw.strip()]
                            if any(kw in formatted_content.lower() for kw in pref_kwds):
                                logger.debug("Bypassing deduplication for short preference statement: '%s'", formatted_content)
                                processed_memories.append(memory_dict)
                                continue  # Skip duplicate checking entirely for this memory

                        is_duplicate = False
                        similarity_score = 0.0 # Track similarity score for logging
                        similarity_method = 'none' # Track method used

                        if use_embeddings:
                            # Precompute embedding for the new memory once
                            try:
                                if self.embedding_model is None:
                                    raise ValueError("Embedding model not available")
                                new_embedding = self.embedding_model.encode(
                                    formatted_content.lower().strip(), normalize_embeddings=True
                                )
                            except Exception as e:
                                logger.warning(f"Failed to encode new memory for deduplication; falling back to text sim. Error: {e}")
                                use_embeddings = False  # fall back

                        for existing_idx, existing_content in enumerate(existing_contents):
                            if use_embeddings:
                                # Retrieve or compute embedding for the existing memory content
                                existing_mem_dict = existing_memories[existing_idx]
                                existing_id = existing_mem_dict.get("id")
                                existing_emb = self.memory_embeddings.get(existing_id)
                                if existing_emb is None and self.embedding_model is not None:
                                    try:
                                        existing_emb = self.embedding_model.encode(
                                            existing_content.lower().strip(), normalize_embeddings=True
                                        )
                                        self.memory_embeddings[existing_id] = existing_emb
                                    except Exception:
                                        # On failure, mark duplicate check using text sim for this item
                                        existing_emb = None
                                if existing_emb is not None:
                                    similarity = float(np.dot(new_embedding, existing_emb))
                                    similarity_score = similarity # Store score
                                    similarity_method = 'embedding'
                                else:
                                    similarity = self._calculate_memory_similarity(
                                        formatted_content, existing_content
                                    )
                                    similarity_score = similarity # Store score
                                    similarity_method = 'text'
                            else:
                                # Choose the appropriate similarity calculation method
                                similarity = self._calculate_memory_similarity(
                                    formatted_content, existing_content
                                )
                                
                            if similarity >= self.valves.similarity_threshold:
                                logger.debug(
                                    f"  -> Duplicate found vs existing mem {existing_idx} (Similarity: {similarity_score:.3f}, Method: {similarity_method}, Threshold: {self.valves.similarity_threshold})"
                                )
                                logger.debug(
                                    f"Skipping duplicate NEW memory (similarity: {similarity_score:.2f}, method: {similarity_method}): {formatted_content[:50]}..."
                                )
                                is_duplicate = True
                                # Increment duplicate skipped counter for status reporting
                                self._duplicate_skipped += 1
                                break # Stop checking against other existing memories for this new one

                        if not is_duplicate:
                            logger.debug(f"  -> No duplicate found. Adding to processed list: {formatted_content[:50]}...")
                            processed_memories.append(memory_dict)
                        else:
                             logger.debug(f"NEW memory was identified as duplicate and skipped: {formatted_content[:50]}...")
                    else:
                        # Keep all UPDATE and DELETE operations
                        logger.debug(f"Keeping non-NEW operation: {memory_dict['operation']} ID: {memory_dict.get('id', 'N/A')}")
                        processed_memories.append(memory_dict)
            else:
                logger.debug("Deduplication skipped (valve disabled or no existing memories). Processing all operations.")
                processed_memories = memories

            # Process the filtered memories
            logger.debug(f"Executing {len(processed_memories)} filtered memory operations.")
            for idx, memory_dict in enumerate(processed_memories):
                logger.debug(f"Executing operation {idx + 1}/{len(processed_memories)}: {memory_dict}")
                try:
                    # Validate memory operation
                    operation = MemoryOperation(**memory_dict)
                    # Execute the memory operation
                    await self._execute_memory_operation(operation, user)
                    # If successful, add to our list
                    logger.debug(f"Successfully executed operation: {operation.operation} ID: {operation.id}")
                    successfully_saved_ops.append(memory_dict)
                except ValueError as e:
                    logger.error(f"Invalid memory operation during execution phase: {e} {memory_dict}")
                    self.error_counters["memory_crud_errors"] += 1 # Increment error counter
                    continue
                except Exception as e:
                    logger.error(f"Error executing memory operation in process_memories: {e} {memory_dict}")
                    self.error_counters["memory_crud_errors"] += 1 # Increment error counter
                    continue

            logger.debug(
                f"Successfully executed {len(successfully_saved_ops)} memory operations out of {len(processed_memories)} processed.")
            # Add confirmation message if any memory was added or updated
            if successfully_saved_ops:
                # Check if any operation was NEW or UPDATE
                if any(op.get("operation") in ["NEW", "UPDATE"] for op in successfully_saved_ops):
                    logger.debug("Attempting to add confirmation message.") # Log confirmation attempt
                    try:
                        from fastapi.requests import Request  # ensure import

                        # Find the last assistant message and append confirmation
                        # This is a safe operation, no error if no assistant message
                        for i in reversed(range(len(self._last_body.get("messages", [])))):
                            msg = self._last_body["messages"][i]
                            if msg.get("role") == "assistant":
                                # Do nothing here
                                break
                    except Exception:
                        pass
            return successfully_saved_ops
        except Exception as e:
            logger.error(f"Error processing memories: {e}\n{traceback.format_exc()}")
            return []  # Return empty list on major error

    async def _execute_memory_operation(
        self, operation: MemoryOperation, user: Any
    ) -> None:
        """Execute a memory operation (NEW, UPDATE, DELETE)"""
        formatted_content = self._format_memory_content(operation)

        if operation.operation == "NEW":
            try:
                result = await add_memory(
                    request=Request(scope={"type": "http", "app": webui_app}),
                    form_data=AddMemoryForm(content=formatted_content),
                    user=user,
                )
                logger.info(f"NEW memory created: {formatted_content[:50]}...")

                # Generate and cache embedding for new memory if embedding model is available
                # This helps with future deduplication checks when using embedding-based similarity
                if self.embedding_model is not None:
                    # Handle both Pydantic model and dict response forms
                    mem_id = getattr(result, "id", None)
                    if mem_id is None and isinstance(result, dict):
                        mem_id = result.get("id")
                    if mem_id is not None:
                        try:
                            memory_clean = re.sub(r"\[Tags:.*?\]\s*", "", formatted_content).lower().strip()
                            memory_embedding = self.embedding_model.encode(
                                memory_clean, normalize_embeddings=True
                            )
                            self.memory_embeddings[mem_id] = memory_embedding
                            logger.debug(f"Generated and cached embedding for new memory ID: {mem_id}")
                        except Exception as e:
                            logger.warning(f"Failed to generate embedding for new memory: {e}")
                            # Non-critical error, don't raise

            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                logger.error(
                    f"Error creating memory (operation=NEW, user_id={getattr(user, 'id', 'unknown')}): {e}\n{traceback.format_exc()}"
                )
                raise

        elif operation.operation == "UPDATE" and operation.id:
            try:
                # Delete existing memory
                deleted = await delete_memory_by_id(operation.id, user=user)
                if deleted:
                    # Create new memory with updated content
                    result = await add_memory(
                        request=Request(scope={"type": "http", "app": webui_app}),
                        form_data=AddMemoryForm(content=formatted_content),
                        user=user,
                    )
                    logger.info(
                        f"UPDATE memory {operation.id}: {formatted_content[:50]}..."
                    )

                    # Update embedding for modified memory
                    if self.embedding_model is not None:
                        # Handle both Pydantic model and dict response forms
                        new_mem_id = getattr(result, "id", None)
                        if new_mem_id is None and isinstance(result, dict):
                            new_mem_id = result.get("id")

                        if new_mem_id is not None:
                            try:
                                memory_clean = re.sub(r"\[Tags:.*?\]\s*", "", formatted_content).lower().strip()
                                memory_embedding = self.embedding_model.encode(
                                    memory_clean, normalize_embeddings=True
                                )
                                # Store with the new ID from the result
                                self.memory_embeddings[new_mem_id] = memory_embedding
                                logger.debug(
                                    f"Updated embedding for memory ID: {new_mem_id} (was: {operation.id})"
                                )

                                # Remove old embedding if ID changed
                                if operation.id != new_mem_id and operation.id in self.memory_embeddings:
                                    del self.memory_embeddings[operation.id]
                            except Exception as e:
                                logger.warning(
                                    f"Failed to update embedding for memory ID {new_mem_id}: {e}"
                                )
                                # Non-critical error, don't raise

                else:
                    logger.warning(f"Memory {operation.id} not found for UPDATE")
            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                logger.error(
                    f"Error updating memory (operation=UPDATE, memory_id={operation.id}, user_id={getattr(user, 'id', 'unknown')}): {e}\n{traceback.format_exc()}"
                )
                raise

            # Invalidate cache entries involving this memory
            mem_emb = self.memory_embeddings.get(operation.id)
            if mem_emb is not None:
                keys_to_delete = []
                for key, (score, ts) in self.relevance_cache.items():
                    # key is hash of (user_emb, mem_emb)
                    # We can't extract mem_emb from key, so approximate by deleting all keys with this mem_emb
                    # Since we can't reverse hash, we skip this for now
                    # Future: store reverse index or use tuple keys
                    pass  # Placeholder for future precise invalidation

        elif operation.operation == "DELETE" and operation.id:
            try:
                deleted = await delete_memory_by_id(operation.id, user=user)
                logger.info(f"DELETE memory {operation.id}: {deleted}")

                # Invalidate cache entries involving this memory
                mem_emb = self.memory_embeddings.get(operation.id)
                if mem_emb is not None:
                    keys_to_delete = []
                    for key, (score, ts) in self.relevance_cache.items():
                        # Same as above, placeholder
                        pass

                # Remove embedding
                if operation.id in self.memory_embeddings:
                    del self.memory_embeddings[operation.id]
                    logger.debug(f"Removed embedding for deleted memory ID: {operation.id}")

            except Exception as e:
                self.error_counters["memory_crud_errors"] += 1
                logger.error(
                    f"Error deleting memory (operation=DELETE, memory_id={operation.id}, user_id={getattr(user, 'id', 'unknown')}): {e}\n{traceback.format_exc()}"
                )
                raise

    def _format_memory_content(self, operation: MemoryOperation) -> str:
        """Format memory content with tags"""
        if not operation.tags:
            return operation.content or ""

        return f"[Tags: {', '.join(operation.tags)}] {operation.content}"

    async def query_llm_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """Query LLM with retry logic, supporting multiple provider types.

        Args:
            system_prompt: System prompt for context/instructions
            user_prompt: User prompt/query

        Returns:
            String response from LLM or error message
        """
        # Get configuration from valves
        provider_type = self.valves.llm_provider_type
        model = self.valves.llm_model_name
        api_url = self.valves.llm_api_endpoint_url
        api_key = self.valves.llm_api_key
        max_retries = self.valves.max_retries
        retry_delay = self.valves.retry_delay

        logger.info(
            f"LLM Query: Provider={provider_type}, Model={model}, URL={api_url}"
        )
        logger.debug(
            f"System prompt length: {len(system_prompt)}, User prompt length: {len(user_prompt)}"
        )

        # Ensure we have a valid aiohttp session
        session = await self._get_aiohttp_session()

        # Add the current datetime to system prompt for time awareness
        system_prompt_with_date = system_prompt
        try:
            now = self.get_formatted_datetime()
            tzname = now.tzname() or "UTC"
            system_prompt_with_date = f"{system_prompt}\n\nCurrent date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} {tzname}"
        except Exception as e:
            logger.warning(f"Could not add date to system prompt: {e}")

        headers = {"Content-Type": "application/json"}

        # Add API key if provided (required for OpenAI-compatible APIs)
        if provider_type == "openai_compatible" and api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        for attempt in range(
            1, max_retries + 2
        ):  # +2 because we start at 1 and want max_retries+1 attempts
            logger.debug(f"LLM query attempt {attempt}/{max_retries+1}")
            try:
                if provider_type == "ollama":
                    # Prepare the request body for Ollama
                    data = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt_with_date},
                            {"role": "user", "content": user_prompt},
                        ],
                        # Set some parameters to encourage consistent outputs
                        "options": {
                            "temperature": 0.1,  # Lower temperature for more deterministic responses
                            "top_p": 0.95,  # Slightly constrain token selection
                            "top_k": 80,  # Reasonable top_k value
                            "num_predict": 2048,  # Reasonable length limit
                            "format": "json",  # Request JSON format
                        },
                        # Disable streaming so we get a single JSON response; newer Ollama respects this flag.
                        "stream": False,
                    }
                    logger.debug(f"Ollama request data: {json.dumps(data)[:500]}...")
                elif provider_type == "openai_compatible":
                    # Prepare the request body for OpenAI-compatible API
                    data = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt_with_date},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 1024,
                        "response_format": {
                            "type": "json_object"
                        },  # Request JSON format when supported
                    }
                    logger.debug(
                        f"OpenAI-compatible request data: {json.dumps(data)[:500]}..."
                    )
                else:
                    error_msg = f"Unsupported provider type: {provider_type}"
                    logger.error(error_msg)
                    return error_msg

                # Log the API call attempt
                logger.info(
                    f"Making API request to {api_url} (attempt {attempt}/{max_retries+1})"
                )

                # Make the API call with timeout
                async with session.post(
                    api_url, json=data, headers=headers, timeout=60
                ) as response:
                    # Log the response status
                    logger.info(f"API response status: {response.status}")

                    if response.status == 200:
                        # Success - parse the response, handling both JSON and NDJSON
                        content_type = response.headers.get("content-type", "")
                        if "application/x-ndjson" in content_type:
                            # Ollama may still return NDJSON even with stream=False; aggregate lines
                            raw_text = await response.text()
                            logger.debug(
                                f"Received NDJSON response length: {len(raw_text)}"
                            )
                            last_json = None
                            for line in raw_text.strip().splitlines():
                                try:
                                    last_json = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                            if last_json is None:
                                error_msg = "Could not decode NDJSON response from LLM"
                                logger.error(error_msg)
                                if attempt > max_retries:
                                    return error_msg
                                else:
                                    continue
                            data = last_json
                        else:
                            # Regular JSON
                            data = await response.json()

                        # Extract content based on provider type
                        content = None

                        # Log the raw response for debugging
                        logger.debug(f"Raw API response: {json.dumps(data)[:500]}...")

                        if provider_type == "openai_compatible":
                            if (
                                data.get("choices")
                                and data["choices"][0].get("message")
                                and data["choices"][0]["message"].get("content")
                            ):
                                content = data["choices"][0]["message"]["content"]
                                logger.info(
                                    f"Retrieved content from OpenAI-compatible response (length: {len(content)})"
                                )
                        elif provider_type == "ollama":
                            if data.get("message") and data["message"].get("content"):
                                content = data["message"]["content"]
                                logger.info(
                                    f"Retrieved content from Ollama response (length: {len(content)})"
                                )

                        if content:
                            return content
                        else:
                            error_msg = f"Could not extract content from {provider_type} response format"
                            logger.error(f"{error_msg}: {data}")

                            # If we're on the last attempt, return the error message
                            if attempt > max_retries:
                                return error_msg
                    else:
                        # Handle error response
                        error_text = await response.text()
                        error_msg = f"Error: LLM API ({provider_type}) returned {response.status}: {error_text}"
                        logger.warning(f"API error: {error_msg}")

                        # Determine if we should retry based on status code
                        is_retryable = response.status in [429, 500, 502, 503, 504]

                        if is_retryable and attempt <= max_retries:
                            sleep_time = retry_delay * (
                                2 ** (attempt - 1)
                            ) + random.uniform(
                                0, 1.0
                            )  # Longer backoff for rate limits/server errors
                            logger.warning(f"Retrying in {sleep_time:.2f} seconds...")
                            await asyncio.sleep(sleep_time)
                            continue  # Retry
                        else:
                            return error_msg  # Final failure

            except asyncio.TimeoutError:
                logger.warning(f"Attempt {attempt} failed: LLM API request timed out")
                if attempt <= max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    await asyncio.sleep(sleep_time)
                    continue  # Retry on timeout
                else:
                    return "Error: LLM API request timed out after multiple retries."
            except ClientError as e:
                logger.warning(
                    f"Attempt {attempt} failed: LLM API connection error: {str(e)}"
                )
                if attempt <= max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    await asyncio.sleep(sleep_time)
                    continue  # Retry on connection error
                else:
                    return f"Error: LLM API connection error after multiple retries: {str(e)}"
            except Exception as e:
                logger.error(
                    f"Attempt {attempt} failed: Unexpected error during LLM query: {e}\n{traceback.format_exc()}"
                )
                if attempt <= max_retries:
                    # Generic retry for unexpected errors
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    await asyncio.sleep(sleep_time)
                    continue
                else:
                    return f"Error: Failed after {max_retries} attempts due to unexpected error: {str(e)}"

        return f"Error: LLM query failed after {max_retries} attempts."

    async def _add_confirmation_message(self, body: Dict[str, Any]) -> None:
        """Add a confirmation message about memory operations"""
        if (
            not body
            or "messages" not in body
            or not body["messages"]
            or not self.valves.show_status
        ):
            return

        # Prepare the confirmation message
        confirmation = ""

        if self._error_message:
            confirmation = f"(Memory error: {self._error_message})"
        elif self.stored_memories:
            # Count operations by type
            new_count = 0
            update_count = 0
            delete_count = 0

            for memory in self.stored_memories:
                if memory["operation"] == "NEW":
                    new_count += 1
                elif memory["operation"] == "UPDATE":
                    update_count += 1
                elif memory["operation"] == "DELETE":
                    delete_count += 1

            # Build the confirmation message in new styled format
            total_saved = new_count + update_count + delete_count

            # Use bold italic styling with an emoji as requested
            confirmation = f"**_Memory: 🧠 Saved {total_saved} memories..._**"

        # If no confirmation necessary, exit early
        if not confirmation:
            logger.debug("No memory confirmation message needed")
            return

        # Critical fix: Make a complete deep copy of the messages array
        try:
            logger.debug("Making deep copy of messages array for safe modification")
            messages_copy = copy.deepcopy(body["messages"])

            # Find the last assistant message
            last_assistant_idx = -1
            for i in range(len(messages_copy) - 1, -1, -1):
                if messages_copy[i].get("role") == "assistant":
                    last_assistant_idx = i
                    break

            # If found, modify the copy
            if last_assistant_idx != -1:
                # Get the original content
                original_content = messages_copy[last_assistant_idx].get("content", "")

                # Append the confirmation message
                messages_copy[last_assistant_idx]["content"] = (
                    original_content + f" {confirmation}"
                )

                # Replace the entire messages array in body
                logger.debug(
                    f"Replacing messages array with modified copy containing confirmation: {confirmation}"
                )
                body["messages"] = messages_copy
            else:
                logger.debug("No assistant message found to append confirmation")

        except Exception as e:
            logger.error(f"Error adding confirmation message: {e}")
            # Don't modify anything if there's an error

    # Cleanup method for aiohttp session and background tasks
    async def cleanup(self):
        """Clean up resources when filter is being shut down"""
        logger.info("Cleaning up Adaptive Memory Filter")
        
        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done() and not task.cancelled():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    # Expected when cancelling
                    pass
                except Exception as e:
                    logger.error(f"Error while cancelling task: {e}")
        
        # Clear task tracking set
        self._background_tasks.clear()
        
        # Close any open sessions
        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()
            
        # Clear memory caches to help with GC
        self._memory_embeddings = {}
        self._relevance_cache = {}
        
        logger.info("Adaptive Memory Filter cleanup complete")

    def _convert_dict_to_memory_operations(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert a dictionary returned by the LLM into the expected list of memory operations.

        Handles cases where the LLM returns a dict containing a list (e.g., {"memories": [...]})
        or a flatter structure. Avoids adding unnecessary prefixes.
        """
        if not isinstance(data, dict) or not data:
            return []

        operations: List[Dict[str, Any]] = []
        seen_content = set()

        # --- Primary Handling: Check for a key containing a list of operations ---
        # Common keys LLMs might use: "memories", "memory_operations", "results", "operations"
        list_keys = ["memories", "memory_operations", "results", "operations"]
        processed_primary = False
        for key in list_keys:
            if key in data and isinstance(data[key], list):
                logger.info(
                    f"Found list of operations under key '{key}', processing directly."
                )
                for item in data[key]:
                    if isinstance(item, dict):
                        # Extract fields directly, provide defaults
                        op = item.get("operation", "NEW").upper()  # Default to NEW
                        content = item.get(
                            "content", item.get("memory", item.get("value"))
                        )  # Check common content keys
                        tags = item.get("tags", [])

                        # Basic validation
                        if op not in ["NEW", "UPDATE", "DELETE"]:
                            continue
                        if (
                            not content
                            or not isinstance(content, str)
                            or len(content) < 5
                        ):
                            continue  # Skip empty/short content
                        if not isinstance(tags, list):
                            tags = [str(tags)]  # Ensure tags is a list

                        # Add if content is unique
                        if content not in seen_content:
                            operations.append(
                                {"operation": op, "content": content, "tags": tags}
                            )
                            seen_content.add(content)
                processed_primary = True
                break  # Stop after processing the first found list

        # --- Fallback Handling: If no primary list found, try simple key-value flattening ---
        if not processed_primary:
            logger.info(
                "No primary operations list found, attempting fallback key-value flattening."
            )
            # Helper maps for simple tag inference (less critical now)
            identity_keys = {"name", "username", "location", "city", "country", "age"}
            goal_keys = {"goal", "objective", "plan"}
            preference_keys = {
                "likes",
                "dislikes",
                "interests",
                "hobbies",
                "favorite",
                "preference",
            }
            relationship_keys = {"family", "friend", "brother", "sister"}
            ignore_keys = {"notes", "meta", "trivia"}

            for key, value in data.items():
                lowered_key = key.lower()
                if (
                    lowered_key in ignore_keys
                    or not isinstance(value, (str, int, float, bool))
                    or not str(value).strip()
                ):
                    continue

                content = str(value).strip()
                if len(content) > 5 and content not in seen_content:
                    # Simple tag inference
                    tag = "preference"  # Default tag
                    if lowered_key in identity_keys:
                        tag = "identity"
                    elif lowered_key in goal_keys:
                        tag = "goal"
                    elif lowered_key in relationship_keys:
                        tag = "relationship"

                    # Format simply: "Key: Value" unless key is generic
                    generic_keys = {"content", "memory", "text", "value", "result", "data"}
                    if key.lower() in generic_keys:
                        content_to_save = content # Use content directly
                    else:
                        # Prepend the key for non-generic keys
                        content_to_save = f"{key.replace('_', ' ').capitalize()}: {content}"

                    operations.append(
                        {
                            "operation": "NEW",
                            "content": content_to_save,
                            "tags": [tag],
                        }
                    )
                    seen_content.add(content)

        logger.info(f"Converted dict response into {len(operations)} memory operations")
        return operations
