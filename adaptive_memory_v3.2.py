import json
import copy  # Add deepcopy import
import traceback
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, Set, Tuple
import logging
import re
import asyncio
import pytz
import difflib
from difflib import SequenceMatcher
import random
import time
import os  # Added for local embedding model discovery
from urllib.parse import urlparse
import uuid

# ----------------------------
# Metrics & Monitoring Imports
# ----------------------------
try:
    from prometheus_client import Counter, Histogram, REGISTRY, generate_latest  # type: ignore
except ImportError:
    # Fallback: define dummy Counter/Histogram if prometheus_client not installed
    class _NoOpMetric:
        def __init__(self, *args, **kwargs):
            pass
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass

    Counter = Histogram = _NoOpMetric

# Define Prometheus metrics (or no-op if client missing)
EMBEDDING_REQUESTS = Counter('adaptive_memory_embedding_requests_total', 'Total number of embedding requests', ['provider'])
EMBEDDING_ERRORS = Counter('adaptive_memory_embedding_errors_total', 'Total number of embedding errors', ['provider'])
EMBEDDING_LATENCY = Histogram('adaptive_memory_embedding_latency_seconds', 'Latency of embedding generation', ['provider'])

RETRIEVAL_REQUESTS = Counter('adaptive_memory_retrieval_requests_total', 'Total number of get_relevant_memories calls', [])
RETRIEVAL_ERRORS = Counter('adaptive_memory_retrieval_errors_total', 'Total number of retrieval errors', [])
RETRIEVAL_LATENCY = Histogram('adaptive_memory_retrieval_latency_seconds', 'Latency of get_relevant_memories execution', [])

# Embedding model imports
from sentence_transformers import SentenceTransformer
import numpy as np

import aiohttp
from aiohttp import ClientError, ClientSession
from fastapi.requests import Request
from fastapi import APIRouter, Response
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
    memory_bank: Optional[str] = None  # Memory bank assignment
    confidence: Optional[float] = None  # Confidence score
    dynamic_tags: List[Dict[str, Any]] = []  # NEW - Dynamic keyword tags with confidence scores


class Filter:
    # Class-level singleton attributes to avoid missing attribute errors
    _embedding_model = None # Keep the underlying attribute name
    _memory_embeddings = {}
    _relevance_cache = {}

    @property
    def _local_embedding_model(self): # RENAMED from embedding_model
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                # Use the model name from valves for local loading too
                local_model_name = self.valves.embedding_model_name if self.valves.embedding_provider_type == 'local' else 'all-MiniLM-L6-v2'
                logger.info(f"Loading local embedding model: {local_model_name}")
                self._embedding_model = SentenceTransformer(local_model_name)
            except Exception as e:
                logger.error(f"Failed to load local SentenceTransformer model: {e}")
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
        """Adaptive Memory configuration valves"""
        
        # --- NEW: Provider Lock Properties ---
        # These prevent accidental overrides when explicitly configured
        _provider_locked: bool = False
        _lock_reason: str = ""
        
        # --- Memory System ---
        embedding_provider_type: Literal["local", "openai_compatible"] = Field(
            default="local",
            description="Type of embedding provider ('local' for SentenceTransformer or 'openai_compatible' for API)",
        )
        embedding_model_name: str = Field(
            default="all-MiniLM-L6-v2",  # Default to the local model
            description="Name of the embedding model to use (e.g., 'all-MiniLM-L6-v2', 'text-embedding-3-small')",
        )
        embedding_api_url: Optional[str] = Field(
            default=None,
            description="API endpoint URL for the embedding provider (required if type is 'openai_compatible')",
        )
        embedding_api_key: Optional[str] = Field(
            default=None,
            description="API Key for the embedding provider (required if type is 'openai_compatible')",
        )
        # ------ End Embedding Model Configuration ------

        # ------ Begin Dynamic Tagging Configuration ------ # NEW
        enable_dynamic_tagging: bool = Field(
            default=True,
            description="Enable or disable LLM-generated dynamic tags for memories"
        )
        max_dynamic_tags: int = Field(
            default=5,
            description="Maximum number of dynamic tags to generate per memory"
        )
        min_tag_confidence: float = Field(
            default=0.6,
            description="Minimum confidence score (0-1) required for a dynamic tag to be included"
        )
        tag_validation_regex: str = Field(
            default="^[a-z0-9_]{1,30}$",
            description="Regex pattern for validating dynamic tags (lowercase, numbers, underscore, 1-30 chars)"
        )
        tag_blacklist: str = Field(
            default="user,memory,general,specific,important,personal,work,fact,info,data,detail",
            description="Comma-separated list of blacklisted (too generic) tag words"
        )
        tag_generation_prompt: str = Field(
            default='''You are a memory tagging assistant. Analyze the given memory and generate 2-3 relevant tags.

MEMORY TAGGING RULES:
1. Generate 2-3 SPECIFIC but REUSABLE keyword tags for the memory.
2. Tags MUST be:
   - Lowercase, no spaces (use underscores for multi-word concepts)
   - Only letters, numbers, and underscores (no special characters)
   - 1-30 characters long
   - Related to content type, domain, or key concepts
   - Not duplicating existing category tags or being too generic

3. AVOID generic tags like: user, memory, general, specific, important, personal, work, fact, info, data

4. For EACH tag, include a confidence score (0-1) based on how relevant and specific it is.

FORMATTING:
Return ONLY a JSON array containing objects with "tag" and "confidence" fields.
Example: [{"tag": "finance", "confidence": 0.9}, {"tag": "retirement_planning", "confidence": 0.85}, {"tag": "investment", "confidence": 0.7}]

EXAMPLES:

Memory: "User prefers dark mode in all applications"
Good Tags: [{"tag": "ui_preference", "confidence": 0.9}, {"tag": "theme", "confidence": 0.8}, {"tag": "accessibility", "confidence": 0.7}]
Bad Tags: ["dark_mode", "apps", "preference"] (too specific/generic/missing confidence)

Memory: "User is allergic to peanuts and shellfish"
Good Tags: [{"tag": "allergies", "confidence": 0.95}, {"tag": "food_restrictions", "confidence": 0.85}, {"tag": "health", "confidence": 0.75}]

Memory: "User's daughter Sarah is starting college next fall"
Good Tags: [{"tag": "family", "confidence": 0.9}, {"tag": "education", "confidence": 0.85}, {"tag": "life_event", "confidence": 0.8}]

Analyze the following memory and provide ONLY the JSON array output of tags with confidence scores:''',
            description="System prompt for generating tags for memories"
        )
        # ------ End Dynamic Tagging Configuration ------ # NEW

        # ------ Begin Background Task Management Configuration ------
        enable_summarization_task: bool = Field(
            default=True,
            description="Enable or disable the background memory summarization task"
        )
        summarization_interval: int = Field(
            default=7200,  # 2 hours performance setting
            description="Interval in seconds between memory summarization runs"
        )
        
        enable_error_logging_task: bool = Field(
            default=True,
            description="Enable or disable the background error counter logging task"
        )
        error_logging_interval: int = Field(
            default=1800,  # 30 minutes performance setting
            description="Interval in seconds between error counter log entries"
        )
        
        enable_date_update_task: bool = Field(
            default=True,
            description="Enable or disable the background date update task"
        )
        date_update_interval: int = Field(
            default=3600,  # 1 hour performance setting
            description="Interval in seconds between date information updates"
        )
        
        enable_model_discovery_task: bool = Field(
            default=True,
            description="Enable or disable the background model discovery task"
        )
        model_discovery_interval: int = Field(
            default=7200,  # 2 hours performance setting
            description="Interval in seconds between model discovery runs"
        )
        # ------ End Background Task Management Configuration ------
        
        # ------ Begin Summarization Configuration ------
        summarization_min_cluster_size: int = Field(
            default=3,
            description="Minimum number of memories in a cluster for summarization"
        )
        summarization_similarity_threshold: float = Field(
            default=0.7,
            description="Threshold for considering memories related when using embedding similarity"
        )
        summarization_max_cluster_size: int = Field(
            default=8,
            description="Maximum memories to include in one summarization batch"
        )
        summarization_min_memory_age_days: int = Field(
            default=7,
            description="Minimum age in days for memories to be considered for summarization"
        )
        summarization_strategy: Literal["embeddings", "tags", "hybrid"] = Field(
            default="hybrid",
            description="Strategy for clustering memories: 'embeddings' (semantic similarity), 'tags' (shared tags), or 'hybrid' (combination)"
        )
        summarization_memory_prompt: str = Field(
            default="""You are a memory summarization assistant. Your task is to combine related memories about a user into a concise, comprehensive summary.

Given a set of related memories about a user, create a single paragraph that:
1. Captures all key information from the individual memories
2. Resolves any contradictions (prefer newer information)
3. Maintains specific details when important
4. Removes redundancy
5. Presents the information in a clear, concise format

Focus on preserving the user's:
- Explicit preferences
- Identity details
- Goals and aspirations
- Relationships
- Possessions
- Behavioral patterns

Your summary should be factual, concise, and maintain the same tone as the original memories.
Produce a single paragraph summary of approximately 50-100 words that effectively condenses the information.

Example:
Individual memories:
- "User likes to drink coffee in the morning"
- "User prefers dark roast coffee"
- "User mentioned drinking 2-3 cups of coffee daily"

Good summary:
"User is a coffee enthusiast who drinks 2-3 cups daily, particularly enjoying dark roast varieties in the morning."

Analyze the following related memories and provide a concise summary.""",
            description="System prompt for summarizing clusters of related memories"
        )
        # ------ End Summarization Configuration ------
        
        # ------ Begin Filtering & Saving Configuration ------
        enable_json_stripping: bool = Field(
            default=True,
            description="Attempt to strip non-JSON text before/after the main JSON object/array from LLM responses."
        )
        enable_fallback_regex: bool = Field(
            default=True,  # Enable for performance fallback
            description="If primary JSON parsing fails, attempt a simple regex fallback to extract at least one memory."
        )
        enable_short_preference_shortcut: bool = Field(
            default=True,
            description="If JSON parsing fails for a short message containing preference keywords, directly save the message content."
        )
        # --- NEW: Deduplication bypass for short preference statements ---
        short_preference_no_dedupe_length: int = Field(
            default=120,  # Allow longer short-preference statements to bypass deduplication
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

        # --- NEW: Confidence Score Threshold ---
        min_confidence_threshold: float = Field(
            default=0.5,  # Default minimum confidence score to save a memory
            description="Minimum confidence score (0-1) required for an extracted memory to be saved. Scores below this are discarded."
        )
        # --- END NEW ---

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
            default=0.60, # Lowered default further for better relevance
            description="Minimum relevance score (0-1) for memories to be considered relevant for injection after scoring"
        )
        memory_threshold: float = Field(
            default=0.6,
            description="Threshold for similarity when comparing memories (0-1)",
        )

        # Upgrade plan configs
        vector_similarity_threshold: float = Field(
            default=0.60,  # Lowered default further for better relevance
            description="Minimum cosine similarity for initial vector filtering (0-1)"
        )
        # NEW: If vector similarities are confidently high, skip the expensive LLM relevance call even
        #       when `use_llm_for_relevance` is True. This reduces overall LLM usage (Improvement #5).
        llm_skip_relevance_threshold: float = Field(
            default=0.93, # Slightly higher to reduce frequency of LLM calls (performance tuning)
            description="If *all* vector-filtered memories have similarity >= this threshold, treat the vector score as final relevance and skip the additional LLM call."
        )
        top_n_memories: int = Field(
            default=3, # Performance setting
            description="Number of top similar memories to pass to LLM",
        )
        cache_ttl_seconds: int = Field(
            default=86400,
            description="Cache time-to-live in seconds (default 24 hours)",
        )

        # --- Relevance Calculation Configuration ---
        use_llm_for_relevance: bool = Field(
            default=True,  # Enable LLM fallback by default to ensure relevance scoring works when embeddings are unavailable
            description="Use LLM call for final relevance scoring. When True, the system will query the LLM to rank candidate memories if vector similarity alone is not sufficient or embeddings are unavailable.",
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

        # NEW: Dedicated threshold for embedding-based duplicate detection (higher because embeddings are tighter)
        embedding_similarity_threshold: float = Field(
            default=0.97,
            description="Threshold (0-1) for considering two memories duplicates when using embedding similarity."
        )

        similarity_threshold: float = Field(
            default=0.95,  # Tighten duplicate detection to minimise false positives
            description="Threshold for detecting similar memories (0-1) using text or embeddings"
        )

        # Time settings
        timezone: str = Field(
            default="Asia/Dubai",
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
            default='''You are an automated JSON data extraction system. Your ONLY function is to identify user-specific, persistent facts, preferences, goals, relationships, or interests from the user's messages and output them STRICTLY as a JSON array of operations.

**ABSOLUTE OUTPUT REQUIREMENT: FAILURE TO COMPLY WILL BREAK THE SYSTEM.**
1.  Your **ENTIRE** response **MUST** be **ONLY** a valid JSON array starting with `[` and ending with `]`. 
2.  **NO EXTRA TEXT**: Do **NOT** include **ANY** text, explanations, greetings, apologies, notes, or markdown formatting (like ```json) before or after the JSON array. 
3.  **ARRAY ALWAYS**: Even if you find only one memory, it **MUST** be enclosed in an array: `[{"operation": ...}]`. Do **NOT** output a single JSON object `{...}`.
4.  **EMPTY ARRAY**: If NO relevant user-specific memories are found, output **ONLY** an empty JSON array: `[]`.

**JSON OBJECT STRUCTURE (Each element in the array):**
*   Each element **MUST** be a JSON object: `{"operation": "NEW", "content": "...", "tags": ["..."], "memory_bank": "...", "confidence": float}`
*   **confidence**: You **MUST** include a confidence score (float between 0.0 and 1.0) indicating certainty that the extracted text is a persistent user fact/preference. High confidence (0.8-1.0) for direct statements, lower (0.5-0.7) for inferences or less certain preferences.
*   **memory_bank**: You **MUST** include a `memory_bank` field, choosing from: "General", "Personal", "Work". Default to "General" if unsure.
*   **tags**: You **MUST** include a `tags` field with a list of relevant tags from: ["identity", "behavior", "preference", "goal", "relationship", "possession"].

**INFORMATION TO EXTRACT (User-Specific ONLY):**
*   **Explicit Preferences/Statements:** User states "I love X", "My favorite is Y", "I enjoy Z". Extract these verbatim with high confidence.
*   **Identity:** Name, location, age, profession, etc. (high confidence)
*   **Goals:** Aspirations, plans (medium/high confidence depending on certainty).
*   **Relationships:** Mentions of family, friends, colleagues (high confidence).
*   **Possessions:** Things owned or desired (medium/high confidence).
*   **Behaviors/Interests:** Topics the user discusses or asks about (implying interest - medium confidence).

**RULES (Reiteration - Critical):**
+1. **JSON ARRAY ONLY**: `[`...`]` - Nothing else!
+2. **CONFIDENCE REQUIRED**: Every object needs a `"confidence": float` field.
+3. **MEMORY BANK REQUIRED**: Every object needs a `"memory_bank": "..."` field.
+4. **TAGS REQUIRED**: Every object needs a `"tags": [...]` field.
+5. **USER INFO ONLY**: Discard trivia, questions *to* the AI, temporary thoughts.

**FAILURE EXAMPLES (DO NOT DO THIS):**
*   `Okay, here is the JSON: [...]` <-- INVALID (extra text)
*   ` ```json
[{"operation": ...}]
``` ` <-- INVALID (markdown)
*   `{"memories": [...]}` <-- INVALID (not an array)
*   `{"operation": ...}` <-- INVALID (not in an array)
*   `[{"operation": ..., "content": ..., "tags": [...]}]` <-- INVALID (missing confidence/bank)

**GOOD EXAMPLE OUTPUT (Strictly adhere to this):**
```
[
  {
    "operation": "NEW",
    "content": "User has been a software engineer for 8 years",
    "tags": ["identity", "behavior"],
    "memory_bank": "Work",
    "confidence": 0.95
  },
  {
    "operation": "NEW",
    "content": "User has a cat named Whiskers",
    "tags": ["relationship", "possession"],
    "memory_bank": "Personal",
    "confidence": 0.9
  },
  {
    "operation": "NEW",
    "content": "User prefers working remotely",
    "tags": ["preference", "behavior"],
    "memory_bank": "Work",
    "confidence": 0.7
  },
  {
    "operation": "NEW",
    "content": "User's favorite book might be The Hitchhiker's Guide to the Galaxy",
    "tags": ["preference"],
    "memory_bank": "Personal",
    "confidence": 0.6
  }
]
```

Analyze the following user message(s) and provide **ONLY** the JSON array output. Double-check your response starts with `[` and ends with `]` and contains **NO** other text whatsoever.''', # Use triple single quotes for multiline string
            description="System prompt for memory identification (Emphasizing strict JSON array output and required fields)",
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
            'cache_ttl_seconds', 'max_retries', 'max_injected_memory_length',
            'summarization_min_cluster_size', 'summarization_max_cluster_size', # Added
            'summarization_min_memory_age_days', 'max_dynamic_tags',  # Added dynamic tagging validator
        )
        def check_non_negative_int(cls, v, info):
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{info.field_name} must be a non-negative integer")
            return v

        @field_validator(
            'save_relevance_threshold', 'relevance_threshold', 'memory_threshold',
            'vector_similarity_threshold', 'similarity_threshold',
            'summarization_similarity_threshold',
            'llm_skip_relevance_threshold',  # New field included
            'embedding_similarity_threshold',  # Validate new embedding threshold as 0-1
            'min_confidence_threshold',  # NEW: Validate confidence threshold as 0-1
            'min_tag_confidence', # NEW: Validate tag confidence threshold
            check_fields=False
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

        @field_validator('tag_validation_regex') # NEW: Validator for tag regex
        def check_valid_regex(cls, v):
            """Ensure the tag validation regex is a valid regular expression"""
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regular expression for tag_validation_regex: {e}")
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

        # ------ Begin Memory Bank Configuration ------
        allowed_memory_banks: List[str] = Field(
            default=["General", "Personal", "Work"],
            description="List of allowed memory bank names for categorization."
        )
        default_memory_bank: str = Field(
            default="General",
            description="Default memory bank assigned when LLM omits or supplies an invalid bank."
        )
        # ------ End Memory Bank Configuration ------

        # ------ Begin Error Handling & Guarding Configuration (single authoritative block) ------
        enable_error_counter_guard: bool = Field(
            default=True,
            description="Enable guard to temporarily disable LLM/embedding features if specific error rates spike."
        )
        error_guard_threshold: int = Field(
            default=5,
            description="Number of errors within the window required to activate the guard."
        )
        error_guard_window_seconds: int = Field(
            default=600,  # 10 minutes
            description="Rolling time-window (in seconds) over which errors are counted for guarding logic."
        )
        # ------ End Error Handling & Guarding Configuration ------

        @field_validator(
            'allowed_memory_banks', # Add validation for this field
            check_fields=False # Run even if other validation fails
        )
        def check_allowed_memory_banks(cls, v):
            if not isinstance(v, list) or not v or v == ['']:
                logger.warning(f"Invalid 'allowed_memory_banks' loaded: {v}. Falling back to default.")
                # Return the default defined in the model itself
                return cls.model_fields['allowed_memory_banks'].default
            # Ensure all items are strings and non-empty after stripping
            cleaned_list = [str(item).strip() for item in v if str(item).strip()]
            if not cleaned_list:
                logger.warning(f"Empty list after cleaning 'allowed_memory_banks': {v}. Falling back to default.")
                return cls.model_fields['allowed_memory_banks'].default
            return cleaned_list # Return the cleaned list

        # --- NEW Validator for Embedding Config ---
        @model_validator(mode="after")
        def check_embedding_config(self):
            if self.embedding_provider_type == "openai_compatible":
                if not self.embedding_api_key:
                    raise ValueError(
                        "API Key (embedding_api_key) is required when embedding_provider_type is 'openai_compatible'"
                    )
                if not self.embedding_api_url or not self.embedding_api_url.startswith(("http://", "https://")):
                    raise ValueError(
                        "A valid API URL (embedding_api_url) starting with http:// or https:// is required when embedding_provider_type is 'openai_compatible'"
                    )
            elif self.embedding_provider_type == "local":
                # Optionally add checks for local model availability if needed
                pass
            return self
        # --- End Pydantic Validators for Valves ---

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
        """Initialize memory filter"""
        # Singleton attributes, initialize first to prevent attribute errors
        Filter._embedding_model = None 
        Filter._memory_embeddings = {}
        Filter._relevance_cache = {}

        # Make class loggers available
        self.logger = logger

        # Initialize internal state
        self.valves = self.Valves()
        self.user_valves = {}  # Dict for user-specific valve overrides
        self.embedding_model = None
        self.metrics = {}
        self.error_counters = Counter()
        self.guard_end_time = {}
        self.guard_active = {}
        
        # Initialize error tracking
        self._error_message = None
        
        # Initialize custom configurations based on env
        self._update_valves_from_env()

        # Check and apply LLM provider overrides from environment, forcing an update on init
        self._apply_llm_provider_overrides(force=True)

        # Initialize MiniLM embedding model (singleton)
            logger.info("Applied initial provider configuration during plugin initialization")
            
            # Check if explicit provider settings were specified in environment
            import os
            if (os.environ.get("ADAPTIVE_MEMORY_LLM_PROVIDER") or 
                os.environ.get("ADAPTIVE_MEMORY_LLM_MODEL") or
                os.environ.get("ADAPTIVE_MEMORY_LLM_ENDPOINT") or
                os.environ.get("ADAPTIVE_MEMORY_LLM_API_KEY")):
                # Lock the provider to prevent automatic changes when explicit config provided
                self.lock_provider("Environment variables")
        except Exception as e:
            logger.error(f"Failed to configure provider during initialization: {e}")
        # --- END NEW ---

        self._initialized = True

    # --- DB Helper Stub ---
    async def _update_memory_in_db(self, memory_id: str, memory_data: Dict[str, Any]) -> bool:
        """Update a memory record in the database.

        NOTE: This is a placeholder implementation. Integrate with the actual
        database layer of OpenWebUI when available. Currently it logs and
        returns False to indicate no persistence was performed.
        """
        try:
            logger.debug(f"_update_memory_in_db placeholder called for {memory_id} (no-op).")
            # TODO: Implement actual DB update logic.
            return False
        except Exception as e:
            logger.error(f"_update_memory_in_db failed: {e}")
            return False

    async def _generate_and_validate_dynamic_tags(self, memory_content: str) -> List[Dict[str, Any]]:
        """Generate dynamic tags for a memory using LLM"""
        if not memory_content:
            return []

        tag_response = "Error: LLM call not attempted" # Default value in case of exception
        try:
            # Prompt the LLM to generate tags as JSON
            # Create instructional prompt to guide LLM
            system_prompt = """You are a memory tagging assistant.
Your job is to analyze a memory and extract meaningful tags that categorize its content.
You MUST respond ONLY with a valid JSON array containing tag objects.
Each tag should have a "tag" field containing a standardized tag name, and a "confidence" field (0.0-1.0)."""
            
            user_prompt = f"{self.valves.tag_generation_prompt}\n\n{memory_content}"
            
            # Create messages array for the new query_llm_with_retry implementation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Call LLM to get tag suggestions
            tag_response = await self.query_llm_with_retry(messages=messages)
            
            # If we got a response, try to extract JSON from it
            if tag_response and tag_response != "Error: LLM call not attempted":
                # Extract and parse JSON from LLM response
                try:
                    # Try direct JSON parsing first
                    tag_data = json.loads(tag_response)
                    logger.debug(f"Parsed dynamic tags JSON directly: {tag_data}")
                except json.JSONDecodeError:
                    # If direct parsing fails, try extracting JSON from the response
                    import re
                    json_match = re.search(r'\[.*?\]', tag_response, re.DOTALL)
                    if json_match:
                        try:
                            tag_data = json.loads(json_match.group(0))
                            logger.debug(f"Extracted dynamic tags JSON with regex: {tag_data}")
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse extracted JSON pattern: {json_match.group(0)}")
                            return []
                    else:
                        logger.warning(f"No JSON array pattern found in tag response: {tag_response[:100]}...")
                        return []
                
                # Validate the tags against regex and confidence threshold
                valid_tags = []
                if isinstance(tag_data, list) and tag_data:
                    # Compile the regex pattern once
                    tag_pattern = re.compile(self.valves.tag_validation_regex)
                    
                    # Split blacklist into set for faster lookups
                    blacklist_set = set(self.valves.tag_blacklist.lower().split(','))
                    
                    # Process each tag
                    for tag_obj in tag_data:
                        # Check basic structure
                        if not isinstance(tag_obj, dict) or "tag" not in tag_obj or "confidence" not in tag_obj:
                            continue
                            
                        tag = str(tag_obj["tag"]).strip().lower()
                        try:
                            confidence = float(tag_obj["confidence"])
                        except (ValueError, TypeError):
                            # Skip if confidence isn't a valid float
                            continue
                        
                        # Apply validation checks
                        if (
                            confidence >= self.valves.min_tag_confidence and
                            tag_pattern.match(tag) and
                            tag not in blacklist_set
                        ):
                            valid_tags.append({"tag": tag, "confidence": confidence})
                
                # Sort by confidence (highest first) and limit to max tags
                valid_tags.sort(key=lambda x: x["confidence"], reverse=True)
                return valid_tags[:self.valves.max_dynamic_tags]
                
            else:
                logger.warning("No response from LLM for tag generation")
                return []
                
        except Exception as e:
            logger.error(f"Error generating dynamic tags: {e}")
            return []

    async def query_llm_with_retry(self, messages=None, system_prompt=None, user_prompt=None) -> str:
        """Query LLM with retry logic, supporting multiple provider types.

        Args:
            messages: List of message dictionaries with role and content fields (preferred)
            system_prompt: System prompt (legacy, used only if messages not provided)
            user_prompt: User prompt (legacy, used only if messages not provided)

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
        logger.debug(f"LLM Query with provider={provider_type}, model={model}")

        # Create messages list if not provided
        if messages is None:
            if system_prompt is None or user_prompt is None:
                logger.error("Either messages or both system_prompt and user_prompt must be provided")
                return "Error: Missing required parameters"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

        # Set up HTTP headers
        headers = {"Content-Type": "application/json"}
        if provider_type == "openai_compatible" and api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Track attempts for retry logic
        attempt = 0
        last_error = None

        # Try multiple times based on max_retries setting
        while attempt <= max_retries:
            try:
                # Prepare the request data based on provider type
                if provider_type == "openai_compatible":
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.9,
                        "top_p": 0.95,
                        "max_tokens": 1000
                    }
                elif provider_type == "ollama":
                    payload = {
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "format": "json",  # Request JSON format explicitly
                        "options": {
                            "temperature": 0.9,
                            "top_p": 0.95,
                            "top_k": 80
                        }
                    }
                else:
                    logger.error(f"Unsupported provider type: {provider_type}")
                    return f"Error: Unsupported provider type: {provider_type}"

                # Make the async HTTP request
                async with aiohttp.ClientSession() as session:
                    async with session.post(api_url, json=payload, headers=headers) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"LLM API ({provider_type}) returned {response.status}: {error_text}")
                            self._error_message = "llm_connection_error"  # Set error message for status reporting
                            return f"Error: LLM API ({provider_type}) returned {response.status}: {error_text}"
                        
                        # Parse the response based on provider type
                        data = await response.json()
                        content = None
                        
                        if provider_type == "openai_compatible":
                            if data.get("choices") and data["choices"][0].get("message") and data["choices"][0]["message"].get("content"):
                                content = data["choices"][0]["message"]["content"]
                        elif provider_type == "ollama":
                            if data.get("message") and data["message"].get("content"):
                                content = data["message"]["content"]
                                
                        if content:
                            return content
                        else:
                            logger.error(f"Could not extract content from {provider_type} response: {data}")
                            return f"Error: Could not extract content from {provider_type} response"
                        
            except aiohttp.ClientConnectorError as e:
                last_error = e
                logger.error(f"Connection error (attempt {attempt+1}/{max_retries+1}): {e}")
                # Don't wait on the last attempt
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                
            except Exception as e:
                last_error = e
                logger.error(f"Error querying LLM (attempt {attempt+1}/{max_retries+1}): {e}")
                # Don't wait on the last attempt
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
            
            attempt += 1
        
        # If we reached here, all attempts failed
        error_msg = f"Error: Failed to query LLM after {max_retries+1} attempts. Last error: {last_error}"
        logger.error(error_msg)
        self._error_message = "llm_connection_error"  # Set error message for status reporting
        return error_msg
