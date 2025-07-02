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
import traceback
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union, Set
import logging
import re
import asyncio
import pytz
import difflib
from difflib import SequenceMatcher

# Embedding model imports
from sentence_transformers import SentenceTransformer
import numpy as np

import aiohttp
from aiohttp import ClientError, ClientSession
from fastapi.requests import Request
from pydantic import BaseModel, Field, model_validator

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
logger = logging.getLogger("openwebui.plugins.neural_recall")
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

        # Blacklist topics (comma-separated substrings)
        blacklist_topics: str = Field(
            default="trivia,news,weather,sports,politics,history",
            description="Comma-separated list of topics to ignore during memory extraction",
        )

        # Enable trivia filtering
        filter_trivia: bool = Field(
            default=True,
            description="Enable filtering of trivia/general knowledge memories after extraction",
        )

        # Whitelist keywords (comma-separated substrings)
        whitelist_keywords: str = Field(
            default="goal,interest,preference,project,plan,aspiration,relationship,identity",
            description="Comma-separated keywords that force-save a memory even if blacklisted",
        )

        # Maximum total memories per user
        max_total_memories: int = Field(
            default=200,
            description="Maximum number of memories per user; prune oldest beyond this",
        )

        # Minimum memory length
        min_memory_length: int = Field(
            default=15,
            description="Minimum length of memory content to be saved",
        )

        # Number of recent user messages to include in extraction context
        recent_messages_n: int = Field(
            default=5,
            description="Number of recent user messages to include in extraction prompt context",
        )

        # Relevance threshold for saving memories
        save_relevance_threshold: float = Field(
            default=0.9,
            description="Minimum relevance score to save a memory",
        )

        # Max length of injected memory content (characters)
        max_injected_memory_length: int = Field(
            default=300,
            description="Maximum length of each injected memory snippet",
        )

        # Provider selection
        provider: Literal["OpenRouter"] = Field(
            default="OpenRouter",
            description="LLM provider: OpenRouter (cloud)",
        )

        # OpenRouter settings
        openrouter_url: str = Field(
            default="https://openrouter.ai/api/v1",
            description="OpenRouter API URL",
        )
        openrouter_api_key: Optional[str] = Field(
            default="",
            description="OpenRouter API Key",
        )
        openrouter_model: str = Field(
            default="nvidia/llama-3.1-nemotron-70b-instruct:free",
            description="OpenRouter model name",
        )

        # Memory processing settings
        related_memories_n: int = Field(
            default=3,
            description="Number of related memories to consider",
        )
        relevance_threshold: float = Field(
            default=0.85,
            description="Minimum relevance score (0-1) for memories to be considered relevant",
        )
        memory_threshold: float = Field(
            default=0.85,
            description="Threshold for similarity when comparing memories (0-1)",
        )

        # Upgrade plan configs
        vector_similarity_threshold: float = Field(
            default=0.5,
            description="Minimum cosine similarity for vector filtering (0-1)",
        )
        top_n_memories: int = Field(
            default=3,
            description="Number of top similar memories to pass to LLM",
        )
        cache_ttl_seconds: int = Field(
            default=86400,
            description="Cache time-to-live in seconds (default 24 hours)",
        )

        # Deduplicate identical memories
        deduplicate_memories: bool = Field(
            default=True,
            description="Prevent storing duplicate or very similar memories",
        )

        similarity_threshold: float = Field(
            default=0.9, description="Threshold for detecting similar memories (0-1)"
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
            default="""You are a memory manager for a language model. You process user messages to extract information worth remembering for future conversations.

IMPORTANT: **STRICTLY DO NOT save trivia, general knowledge, or facts unrelated to the user's personal information, preferences, goals, or relationships.** Only save information that is **specific to the user**.

If the user mentions **what they know, want to learn, or asks for explanations**, save that as a **user knowledge, goal, or interest memory**.

If the user **asks about a topic** (e.g., Roman Empire, WW2, quantum computing, blockchain), **ALWAYS save a memory about their interest in that topic**, NOT the facts themselves.  
For example, save:  
`"User is interested in Roman history"`  
NOT trivia facts about the Roman Empire.

Additional examples:
- User asks: "Can you tell me about quantum computing?"  
  Save: `"User is interested in quantum computing"`
- User says: "Explain how blockchain works"  
  Save: `"User wants to learn about blockchain technology"`
- User asks: "What is the weather like in Paris?"  
  Save: `"User is interested in Paris weather"`
- User asks: "Who won the World Cup in 2018?"  
  Save: `"User is interested in football/soccer history"`
- User says: "I love Italian food"  
  Save: `"User likes Italian cuisine"`
- User says: "My sister Sarah is a doctor"  
  Save: `"User's sister is named Sarah and is a doctor"`

IMPORTANT: When the user **asks about any topic**, **ALWAYS create an implicit interest memory** about that topic, even if no explicit interest is stated.

STRICT RULES:
- Do NOT save generic statements like "I don't know" or "Tell me more".
- Do NOT save trivia, news, or facts unrelated to the user.
- Do NOT save empty or very short memories (<10 characters).
- Only save **user-specific, persistent information**.

Your task is to identify facts about the user from the provided text and existing memories. Your ONLY output should be a JSON array of memory operations. Do not include any other text.

Each memory operation should be one of:
- NEW: Create a new memory.
- UPDATE: Update an existing memory.
- DELETE: Remove an existing memory.

Output format MUST be a valid JSON array containing objects with these fields:
- operation: "NEW", "UPDATE", or "DELETE"
- id: memory id (required for UPDATE and DELETE)
- content: memory content (required for NEW and UPDATE)
- tags: array of relevant tags from these categories: ["identity", "behavior", "preference", "goal", "relationship", "possession"]

IMPORTANT: Your response must begin with a properly formatted JSON array like this:
[{"operation": "NEW", "content": "User likes coffee", "tags": ["preference"]}]

STRICT RULES:
- Your response **MUST** be valid JSON.
- Do **NOT** include any text, explanation, or formatting outside the JSON array.
- Do **NOT** wrap the JSON in code blocks or markdown.
- Do **NOT** add comments or extra text.
- If no memories, respond with an empty array: []

Rules for memory content:
- Include full context for understanding.
- Tag memories appropriately for better retrieval.
- Combine related information.
- Avoid storing temporary or query-like information.
- Include location, time, or date information when possible.
- Add context about the memory.
- If the user says "tomorrow", resolve it to a date.
- If a date/time specific fact is mentioned, add the date/time to the memory.

Important information types:
- Basic Identity (age, gender, location, job title, education level, etc.)
- Behaviors (interests, habits, etc.)
- Preferences (communication style, preferred language, etc.)
- Goals (goals, targets, aspirations, etc.)
- Relationships (personal and professional relationships)
- Possessions (important items owned or desired)

For UPDATE and DELETE operations, the 'id' MUST exactly match an ID from 'Existing Memories'. Do NOT generate new IDs.

If the text contains no useful information to remember, return an empty array: []""",
            description="System prompt for memory identification",
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
        self.valves = self.Valves()
        self.stored_memories = None
        self._error_message = None
        self._aiohttp_session = None

        # Background tasks tracking
        self._background_tasks = set()

        # Error counters
        self.error_counters = {
            "embedding_errors": 0,
            "llm_call_errors": 0,
            "json_parse_errors": 0,
            "memory_crud_errors": 0,
        }

        # Schedule periodic error counter logging
        self._error_log_task = asyncio.create_task(self._log_error_counters_loop())
        self._background_tasks.add(self._error_log_task)
        self._error_log_task.add_done_callback(self._background_tasks.discard)

        # Schedule periodic memory summarization
        self._summarization_task = asyncio.create_task(
            self._summarize_old_memories_loop()
        )
        self._background_tasks.add(self._summarization_task)
        self._summarization_task.add_done_callback(self._background_tasks.discard)

        # Model discovery results
        self.available_ollama_models = []
        self.available_openai_models = []

        # Add current date awareness for prompts
        self.current_date = datetime.now()
        self.date_info = self._update_date_info()

        # Schedule regular date updates
        self._date_update_task = self._schedule_date_update()

        # Schedule model discovery
        self._model_discovery_task = self._schedule_model_discovery()

    async def _summarize_old_memories_loop(self):
        """Periodically summarize old memories into concise summaries"""
        try:
            while True:
                await asyncio.sleep(3600)  # Run every hour
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
                            except Exception:
                                pass
                        logger.info("Summarized and pruned old memories")
                except Exception as e:
                    logger.error(f"Error in summarization loop: {e}")
        except asyncio.CancelledError:
            logger.debug("Summarization task cancelled")

        # Initialize MiniLM embedding model (singleton)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # In-memory store: memory_id -> embedding vector (np.array)
        self.memory_embeddings = {}

        # In-memory cache: (hash of user_emb + mem_emb) -> (score, timestamp)
        self.relevance_cache = {}

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
                await asyncio.sleep(300)  # every 5 minutes
                logger.info(f"Error counters: {self.error_counters}")
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
                    await asyncio.sleep(900)  # 15 minutes
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
                        # Wait for 1 hour before checking again
                        await asyncio.sleep(3600)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error(f"Error in model discovery: {e}")
                        await asyncio.sleep(300)  # Retry after 5 minutes on error
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

        return body

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
                                    "description": "💭 Retrieving relevant memories...",
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

                    # Safely emit completion status
                    if show_status:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": "☑ Memory retrieval complete",
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
                        "description": f"🙈 Error retrieving memories: {str(e)}",
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
        """Process outgoing messages for memory operations"""
        if not body or not isinstance(body, dict) or not __user__:
            logger.debug("Missing body or user information in outlet")
            return body

        # Check if the function is enabled for this user
        user_valves = self._get_user_valves(__user__)
        if not user_valves.enabled:
            logger.debug(
                f"Memory manager disabled for user {__user__.get('id', 'unknown')}"
            )
            return body

        try:
            if "messages" in body and len(body["messages"]) >= 2:
                # Process memory updates BEFORE waiting for the full LLM response
                memory_task = None
                user_message = None

                # Get the most recent user message (input to the assistant)
                user_messages = [m for m in body["messages"] if m["role"] == "user"]
                if user_messages:
                    user_message = user_messages[-1]["content"]

                    # Determine if status updates should be shown
                    show_status = user_valves.show_status

                    # Get user timezone if specified
                    user_timezone = (
                        user_valves.timezone
                        if user_valves.timezone
                        else self.valves.timezone
                    )

                    # Safely emit status update
                    if show_status:
                        await self._safe_emit(
                            __event_emitter__,
                            {
                                "type": "status",
                                "data": {
                                    "description": "💭 Processing memories...",
                                    "done": False,
                                },
                            },
                        )

                    # Start memory processing early in a background task
                    memory_task = asyncio.create_task(
                        self._process_user_memories(
                            user_message,
                            __user__["id"],
                            __event_emitter__,
                            show_status,
                            user_timezone,
                        )
                    )

                    # Track the task
                    self._background_tasks.add(memory_task)
                    memory_task.add_done_callback(self._background_tasks.discard)

                # If we've started memory processing, wait for it to complete
                if memory_task:
                    try:
                        self.stored_memories = await memory_task

                        # Add confirmation message if enabled and operations were performed
                        if self.valves.show_status and (
                            self.stored_memories or self._error_message
                        ):
                            await self._add_confirmation_message(body)
                            # Removed redundant (Memory:Saved) suffix
                    except asyncio.CancelledError:
                        logger.debug("Memory processing task cancelled")
                    except Exception as e:
                        logger.error(
                            f"Error in memory processing task: {e}\n{traceback.format_exc()}"
                        )
                        if show_status:
                            await self._safe_emit(
                                __event_emitter__,
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"🙈 Error processing memories: {str(e)}",
                                        "done": True,
                                    },
                                },
                            )

        except Exception as e:
            logger.error(f"Error in outlet: {e}\n{traceback.format_exc()}")
            await self._safe_emit(
                __event_emitter__,
                {
                    "type": "status",
                    "data": {
                        "description": f"🙈 Error processing memories: {str(e)}",
                        "done": True,
                    },
                },
            )

        return body

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

        user_valves = __user__.get("valves", {})

        # If valves is already a dict, use it directly
        if isinstance(user_valves, dict):
            return self.UserValves(**user_valves)

        # If valves is an object, try to convert it
        try:
            # If it's already a UserValves object, return it
            if isinstance(user_valves, self.UserValves):
                return user_valves

            # Otherwise try to extract attributes
            return self.UserValves(
                enabled=getattr(user_valves, "enabled", True),
                show_status=getattr(user_valves, "show_status", True),
                timezone=getattr(user_valves, "timezone", ""),
            )
        except (AttributeError, TypeError):
            # Default to enabled if extraction fails
            logger.debug("Could not determine user valves settings, using defaults")
            return self.UserValves()

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
        logger.info(f"Injected memories:\n{memory_context}")

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
        __event_emitter__: Optional[Callable[[Any], Awaitable[None]]] = None,
        show_status: bool = True,
        user_timezone: str = None,
    ) -> List[Dict[str, Any]]:
        """Process user message for memory operations"""
        logger.debug(f"Processing memories for message: {user_message[:50]}...")

        # Get existing memories for context
        existing_memories = await self._get_formatted_memories(user_id)

        # Identify new memories from user message
        # Collect recent N user messages for better context
        recent_messages = []
        try:
            if "messages" in body and isinstance(body["messages"], list):
                user_msgs = [
                    m["content"] for m in body["messages"] if m.get("role") == "user"
                ]
                recent_messages = user_msgs[-self.valves.recent_messages_n :]
        except Exception:
            recent_messages = [user_message]

        combined_input = "\n".join(recent_messages) if recent_messages else user_message

        memories = await self.identify_memories(
            combined_input, existing_memories, user_timezone
        )

        # Filter memories post-extraction
        filtered_memories = []
        blacklist = [
            s.strip().lower()
            for s in self.valves.blacklist_topics.split(",")
            if s.strip()
        ]
        min_len = self.valves.min_memory_length
        min_relevance = self.valves.save_relevance_threshold

        whitelist = [
            s.strip().lower()
            for s in self.valves.whitelist_keywords.split(",")
            if s.strip()
        ]

        trivia_patterns = [
            r"\b\d{4}\b",  # years
            r"\b(first|origin|history|began|introduced|released|launched|adopted|inspired|footprint|reduction|impact|adoption|controversy|environmental|generation|model|product|device|tablet|company|brand|market|sales|units|million|billion|percent|percentage|school|education|technology|feature|update|version|generation|aluminum|recycled|initiative|initiative|initiative|initiative)\b",
            r"\bapple\b",
            r"\bipad\b",
            r"\bpencil\b",
            r"\btablet\b",
            r"\bdevice\b",
            r"\bproduct\b",
            r"\bmodel\b",
            r"\bcompany\b",
            r"\bbrand\b",
            r"\bmarket\b",
            r"\bsales\b",
            r"\bunits\b",
            r"\bmillion\b",
            r"\bbillion\b",
            r"\bpercent\b",
            r"\bpercentage\b",
            r"\bschool\b",
            r"\beducation\b",
            r"\btechnology\b",
            r"\bfeature\b",
            r"\bupdate\b",
            r"\bversion\b",
            r"\bgeneration\b",
            r"\baluminum\b",
            r"\brecycled\b",
            r"\binitiative\b",
        ]

        for mem in memories:
            content = mem.get("content", "") or ""
            content_lower = content.lower()
            # Skip if too short
            if len(content) < min_len:
                continue
            # Check whitelist keywords
            is_whitelisted = any(w in content_lower for w in whitelist)
            # Skip if contains blacklisted topic substring AND not whitelisted
            if any(b in content_lower for b in blacklist) and not is_whitelisted:
                continue
            # Trivia regex filter
            if self.valves.filter_trivia and not is_whitelisted:
                if any(re.search(pat, content_lower) for pat in trivia_patterns):
                    logger.info(f"Filtered trivia memory: {content[:50]}...")
                    continue
            # Explicit meta-request phrase filter
            meta_phrases = [
                "interest in fun fact",
                "interest in trivia",
                "interest in joke",
                "interest in story",
            ]
            if any(phrase in content_lower for phrase in meta_phrases):
                logger.info(f"Filtered meta-request phrase memory: {content[:50]}...")
                continue
            # LLM-based meta-request classification
            try:
                classification_prompt = f"""Classify the following memory as either META or FACT.

META means it is a meta-request, question, or command to the AI, not persistent user info.

FACT means it is a persistent fact about the user.

Respond with only META or FACT.

Memory:
\"\"\"{content}\"\"\""""

                result = await self.query_llm_with_retry(
                    "You are a classifier.", classification_prompt
                )
                if result:
                    result_clean = result.strip().upper()
                    if "META" in result_clean and "FACT" not in result_clean:
                        logger.info(f"Filtered meta-request memory: {content[:50]}...")
                        continue
            except Exception:
                pass
            # Skip if relevance score is too low
            relevance = mem.get("relevance", 1.0)
            if relevance < min_relevance:
                continue
            filtered_memories.append(mem)

        # Enforce max total memories per user
        try:
            existing_memories_sorted = sorted(
                existing_memories, key=lambda m: m.get("created_at", datetime.utcnow())
            )
            total_after = len(existing_memories_sorted) + len(filtered_memories)
            max_total = self.valves.max_total_memories
            if total_after > max_total:
                num_to_prune = total_after - max_total
                # Prune oldest existing memories
                prune_ids = [m["id"] for m in existing_memories_sorted[:num_to_prune]]
                for pid in prune_ids:
                    try:
                        await delete_memory_by_id(
                            pid, user=Users.get_user_by_id(user_id)
                        )
                        logger.info(
                            f"Pruned old memory {pid} to enforce max total memories"
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        if filtered_memories:
            logger.info(f"Filtered to {len(filtered_memories)} valid memory operations")
            success = await self.process_memories(filtered_memories, user_id)

            # Emit completion status
            if show_status:
                if success:
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "🧠 Memories updated",
                                "done": True,
                            },
                        },
                    )
                else:
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": "🙈 Failed to update memories",
                                "done": True,
                            },
                        },
                    )
        else:
            logger.debug("No new memories identified")
            # Emit completion status (no new memories found)
            if show_status:
                await self._safe_emit(
                    __event_emitter__,
                    {
                        "type": "status",
                        "data": {
                            "description": "",
                            "done": True,
                        },
                    },
                )

        return memories

    async def identify_memories(
        self,
        input_text: str,
        existing_memories: Optional[List[Dict[str, Any]]] = None,
        user_timezone: str = None,
    ) -> List[Dict[str, Any]]:
        """Identify memory operations from the input text"""
        import time

        start = time.perf_counter()
        logger.debug("Starting memory identification")
        try:
            # Build the prompt with existing memories for context
            system_prompt = self.valves.memory_identification_prompt

            # Add category filter configuration
            category_config = []
            if self.valves.enable_identity_memories:
                category_config.append(
                    "- Basic Identity information (age, gender, location, etc.)"
                )
            if self.valves.enable_behavior_memories:
                category_config.append(
                    "- Behavior information (interests, habits, etc.)"
                )
            if self.valves.enable_preference_memories:
                category_config.append(
                    "- Preference information (likes, dislikes, etc.)"
                )
            if self.valves.enable_goal_memories:
                category_config.append(
                    "- Goal information (aspirations, targets, etc.)"
                )
            if self.valves.enable_relationship_memories:
                category_config.append(
                    "- Relationship information (friends, family, etc.)"
                )
            if self.valves.enable_possession_memories:
                category_config.append(
                    "- Possession information (things owned or desired)"
                )

            if category_config:
                system_prompt += "\n\nCurrently configured to collect ONLY these categories of information:\n"
                system_prompt += "\n".join(category_config)

            if existing_memories:
                memory_strings = []
                for mem in existing_memories:
                    memory_strings.append(f"ID: {mem['id']}, CONTENT: {mem['memory']}")

                system_prompt += f"\n\nExisting memories:\n{json.dumps(memory_strings)}"

            # Get the current date and time for context
            current_datetime = self.get_formatted_datetime(user_timezone)

            # Add concise current datetime info
            tzname = current_datetime.tzname() or "UTC"
            offset_hours = (
                int(current_datetime.utcoffset().total_seconds() // 3600)
                if current_datetime.utcoffset()
                else 0
            )
            offset_str = f"GMT{'+' if offset_hours >=0 else ''}{offset_hours}"
            system_prompt += f"""
Current datetime: as of {current_datetime.strftime('%Y-%m-%d %H:%M')} {offset_str}
"""

            # Add important reminder about JSON format
            system_prompt += """
VERY IMPORTANT: Your response must be a valid JSON array like this:
[{"operation": "NEW", "content": "User likes coffee", "tags": ["preference"]}]

Do not format your response like this (this is invalid):
["NEW", "id": "123", "content": "User likes coffee", "tags": ["preference"]]
"""

            logger.debug(
                f"Memory extraction prompt:\n{system_prompt}\nUser input:\n{input_text}"
            )

            # Query the LLM with retries
            response = await self.query_llm_with_retry(system_prompt, input_text)

            logger.debug(f"Raw LLM response: {response}")

            # Emit raw LLM response for debugging
            try:
                if __event_emitter__:
                    await self._safe_emit(
                        __event_emitter__,
                        {
                            "type": "status",
                            "data": {
                                "description": f"LLM raw response: {response[:200]}",
                                "done": False,
                            },
                        },
                    )
            except Exception:
                pass

            if not response or response.startswith("Error:"):
                if response:
                    self._error_message = response
                    logger.error(f"Error from LLM: {response}")
                return []

            # Log the raw response for debugging
            logger.debug(f"Raw LLM response: {response[:500]}")

            # Extract and parse JSON
            memory_operations = self._extract_and_parse_json(response)

            logger.debug(f"Parsed memory operations: {memory_operations}")

            if memory_operations is None:
                logger.warning("Failed to parse any valid JSON from LLM response")
                return []

            if not isinstance(memory_operations, list):
                logger.warning(
                    f"Invalid memory operations format (not a list): {memory_operations}"
                )
                return []

            # Validate the operations
            valid_operations = []
            existing_ids = (
                {mem["id"] for mem in existing_memories} if existing_memories else set()
            )

            for op in memory_operations:
                try:
                    # Attempt to fix and validate the operation
                    if self._validate_memory_operation(op):
                        # Validate IDs for UPDATE/DELETE operations
                        if op["operation"] in ["UPDATE", "DELETE"]:
                            if op["id"] not in existing_ids:
                                logger.warning(
                                    f"Invalid memory ID for {op['operation']}: {op['id']}"
                                )
                                continue
                        valid_operations.append(op)
                    else:
                        logger.warning(f"Invalid memory operation format: {op}")
                except Exception as e:
                    logger.warning(f"Error validating memory operation: {e}, op: {op}")

            logger.debug(f"Validated memory operations: {valid_operations}")

            logger.info(f"Identified {len(valid_operations)} valid memory operations")
            logger.info(f"Memory extraction took {time.perf_counter() - start:.2f}s")
            return valid_operations

        except Exception as e:
            self.error_counters["llm_call_errors"] += 1
            logger.error(f"Error identifying memories: {e}\n{traceback.format_exc()}")
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
        if not text:
            logger.warning("Empty text provided to JSON parser")
            return None

        # Clean the text
        text = text.strip()

        # Try direct parsing first (most efficient if valid)
        try:
            parsed = json.loads(text)
            logger.debug("Successfully parsed JSON directly")
            return parsed
        except json.JSONDecodeError:
            pass

        # Handle special case seen in logs: ["NEW", "id": "9e4d6c2b-...", "content": "...", "tags": [...] ]
        malformed_pattern = r'\["(NEW|UPDATE|DELETE)"(?:\s*,\s*|)(?:"id":|)"([^"]*)"(?:\s*,\s*|)(?:"content":|)"([^"]*)"(?:\s*,\s*|)(?:"tags":|)(\[[^\]]*\])'
        match = re.search(malformed_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                operation = match.group(1).upper()
                id_value = match.group(2).strip() if match.group(2) else None
                content = match.group(3).strip()
                tags_str = match.group(4).strip()

                try:
                    tags = json.loads(tags_str)
                except json.JSONDecodeError:
                    # If tags can't be parsed, default to empty list
                    tags = []

                memory_op = {"operation": operation, "content": content, "tags": tags}
                if id_value:
                    memory_op["id"] = id_value

                logger.debug(
                    f"Successfully parsed malformed JSON using pattern: {memory_op}"
                )
                return [memory_op]
            except Exception as e:
                logger.warning(f"Failed to fix malformed JSON with pattern: {e}")

        # Try to find JSON array using regex (more robust approach)
        json_array_pattern = r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]"
        array_match = re.search(json_array_pattern, text, re.DOTALL)

        if array_match:
            json_text = array_match.group(0)
            try:
                parsed = json.loads(json_text)
                logger.debug("Successfully parsed JSON array using regex")
                return parsed
            except json.JSONDecodeError:
                logger.debug("Found JSON-like array but couldn't parse it")

        # Try to find JSON objects one by one
        json_object_pattern = r'\{\s*"operation"\s*:\s*"(NEW|UPDATE|DELETE)".*?\}'
        object_matches = re.finditer(
            json_object_pattern, text, re.DOTALL | re.IGNORECASE
        )

        operations = []
        for match in object_matches:
            try:
                obj = json.loads(match.group(0))
                operations.append(obj)
            except json.JSONDecodeError:
                continue

        if operations:
            logger.debug(
                f"Successfully extracted {len(operations)} operations from text"
            )
            return operations

        # Try extracting from code blocks
        code_block_pattern = r"```(?:json)?(.*?)```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)

        if code_blocks:
            for block in code_blocks:
                try:
                    parsed = json.loads(block.strip())
                    logger.debug("Successfully parsed JSON from code block")
                    return parsed
                except json.JSONDecodeError:
                    continue

        # Last resort: Try to extract key-value pairs directly
        # This handles cases like: ID: abc123, OPERATION: NEW, CONTENT: User likes coffee
        if "ID:" in text and ("OPERATION:" in text or "CONTENT:" in text):
            try:
                operations = []
                # Extract memory entries with regex
                memory_pattern = r"ID:\s*([^,\n]+)(?:,\s*OPERATION:\s*([^,\n]+))?,\s*(?:ID:\s*([^,\n]+),\s*)?CONTENT:\s*([^\n]+)"
                matches = re.findall(memory_pattern, text, re.IGNORECASE | re.DOTALL)

                for match in matches:
                    mem_id = match[0].strip()
                    operation = match[1].strip().upper() if match[1] else "UPDATE"
                    content_id = match[2].strip() if match[2] else mem_id
                    content = match[3].strip()

                    operations.append(
                        {
                            "operation": operation,
                            "id": content_id,
                            "content": content,
                            "tags": [],
                        }
                    )

                if operations:
                    logger.debug(
                        f"Successfully extracted {len(operations)} operations using key-value pattern"
                    )
                    return operations
            except Exception as e:
                logger.debug(f"Failed manual JSON construction: {e}")

        self.error_counters["json_parse_errors"] += 1
        logger.warning(f"Failed to parse JSON from text: {text[:200]}...")
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
            try:
                user_embedding = self.embedding_model.encode(
                    current_message, normalize_embeddings=True
                )
            except Exception as e:
                self.error_counters["embedding_errors"] += 1
                logger.error(
                    f"Error computing embedding for user message: {e}\n{traceback.format_exc()}"
                )
                user_embedding = None

            if user_embedding is not None:
                similarities = []
                for mem in existing_memories:
                    mem_id = mem.get("id")
                    mem_emb = self.memory_embeddings.get(mem_id)
                    if mem_emb is not None:
                        # Cosine similarity
                        sim = float(np.dot(user_embedding, mem_emb))
                        similarities.append((sim, mem))
                # Sort by similarity descending
                similarities.sort(reverse=True, key=lambda x: x[0])
                # Filter by threshold
                sim_threshold = self.valves.vector_similarity_threshold
                top_n = self.valves.top_n_memories
                filtered = [mem for sim, mem in similarities if sim >= sim_threshold][
                    :top_n
                ]
                logger.info(
                    f"Vector filter selected {len(filtered)} of {len(existing_memories)} memories"
                )
                existing_memories = filtered

            # Build the prompt
            memory_strings = []
            for mem in existing_memories:
                memory_strings.append(f"ID: {mem['id']}, CONTENT: {mem['memory']}")

            system_prompt = self.valves.memory_relevance_prompt
            user_prompt = f"""Current user message: "{current_message}"

Available memories:
{json.dumps(memory_strings)}

Rate the relevance of each memory to the current user message."""

            # Add current datetime for context
            current_datetime = self.get_formatted_datetime(user_timezone)
            user_prompt += f"""

Current datetime: {current_datetime.strftime('%A, %B %d, %Y %H:%M:%S')} ({current_datetime.tzinfo})"""

            # For each memory, check cache or call LLM for relevance score
            import time as time_module

            now = time_module.time()
            ttl_seconds = self.valves.cache_ttl_seconds

            relevance_data = []
            uncached_memories = []
            uncached_indices = []

            # Prepare user embedding
            try:
                user_emb = self.embedding_model.encode(
                    current_message, normalize_embeddings=True
                )
            except Exception as e:
                logger.error(
                    f"Error computing embedding for user message (cache check): {e}\n{traceback.format_exc()}"
                )
                user_emb = None

            for idx, mem in enumerate(existing_memories):
                mem_id = mem.get("id")
                mem_emb = self.memory_embeddings.get(mem_id)
                if user_emb is None or mem_emb is None:
                    uncached_memories.append(mem)
                    uncached_indices.append(idx)
                    continue

                key = hash((user_emb.tobytes(), mem_emb.tobytes()))
                cached = self.relevance_cache.get(key)
                if cached:
                    score, ts = cached
                    if now - ts < ttl_seconds:
                        logger.info(f"Cache hit for memory {mem_id}")
                        relevance_data.append(
                            {"memory": mem["memory"], "id": mem_id, "relevance": score}
                        )
                        continue  # use cached
                # else, need to call LLM
                logger.info(f"Cache miss for memory {mem_id}")
                uncached_memories.append(mem)
                uncached_indices.append(idx)

            # If any uncached, build prompt and call LLM once
            if uncached_memories:
                # Build prompt with only uncached memories
                uncached_memory_strings = [
                    f"ID: {mem['id']}, CONTENT: {mem['memory']}"
                    for mem in uncached_memories
                ]
                uncached_prompt = f"""Current user message: "{current_message}"

Available memories:
{json.dumps(uncached_memory_strings)}

Rate the relevance of each memory to the current user message."""

                # Add current datetime for context
                current_datetime = self.get_formatted_datetime(user_timezone)
                uncached_prompt += f"""

Current datetime: {current_datetime.strftime('%A, %B %d, %Y %H:%M:%S')} ({current_datetime.tzinfo})"""

                uncached_response = await self.query_llm_with_retry(
                    system_prompt, uncached_prompt
                )

                if not uncached_response or uncached_response.startswith("Error:"):
                    if uncached_response:
                        logger.error(
                            f"Error from LLM during memory relevance: {uncached_response}"
                        )
                    return []

                uncached_relevance_data = self._extract_and_parse_json(
                    uncached_response
                )

                if not uncached_relevance_data or not isinstance(
                    uncached_relevance_data, list
                ):
                    logger.warning("Failed to parse relevance data from LLM response")
                    return []

                # Save scores to cache and append to relevance_data
                for item in uncached_relevance_data:
                    mem_id = item.get("id")
                    score = item.get("relevance")
                    mem_text = item.get("memory")
                    if mem_id and isinstance(score, (int, float)):
                        relevance_data.append(
                            {"memory": mem_text, "id": mem_id, "relevance": score}
                        )
                        # Save to cache
                        mem_emb = self.memory_embeddings.get(mem_id)
                        if user_emb is not None and mem_emb is not None:
                            key = hash((user_emb.tobytes(), mem_emb.tobytes()))
                            self.relevance_cache[key] = (score, now)

            # Filter by relevance threshold and sort by relevance
            relevant_memories = []
            threshold = self.valves.relevance_threshold

            for item in relevance_data:
                if not isinstance(item, dict):
                    continue

                memory = item.get("memory")
                relevance = item.get("relevance")
                id_val = item.get("id")

                if memory and isinstance(relevance, (int, float)) and id_val:
                    if relevance >= threshold:
                        relevant_memories.append(
                            {"id": id_val, "memory": memory, "relevance": relevance}
                        )

            # Sort by relevance (descending)
            relevant_memories.sort(key=lambda x: x["relevance"], reverse=True)

            # Limit to configured number
            logger.info(
                f"Found {len(relevant_memories)} relevant memories above threshold {threshold}"
            )
            logger.info(f"Memory retrieval took {time.perf_counter() - start:.2f}s")
            return relevant_memories[: self.valves.related_memories_n]

        except Exception as e:
            logger.error(
                f"Error getting relevant memories: {e}\n{traceback.format_exc()}"
            )
            return []

    async def process_memories(
        self, memories: List[Dict[str, Any]], user_id: str
    ) -> bool:
        """Process memory operations"""
        try:
            user = Users.get_user_by_id(user_id)
            if not user:
                logger.error(f"User not found: {user_id}")
                return False

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

                # Check each new memory against existing ones
                for memory_dict in memories:
                    if memory_dict["operation"] == "NEW":
                        # Format the memory content
                        operation = MemoryOperation(**memory_dict)
                        formatted_content = self._format_memory_content(operation)

                        # Check for similarity with existing memories
                        is_duplicate = False
                        for existing_content in existing_contents:
                            similarity = self._calculate_memory_similarity(
                                formatted_content, existing_content
                            )
                            if similarity >= self.valves.similarity_threshold:
                                logger.debug(
                                    f"Skipping duplicate memory (similarity: {similarity:.2f}): {formatted_content[:50]}..."
                                )
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            processed_memories.append(memory_dict)
                    else:
                        # Keep all UPDATE and DELETE operations
                        processed_memories.append(memory_dict)
            else:
                processed_memories = memories

            # Process the filtered memories
            for memory_dict in processed_memories:
                try:
                    # Validate memory operation
                    operation = MemoryOperation(**memory_dict)
                    # Execute the memory operation
                    await self._execute_memory_operation(operation, user)
                except ValueError as e:
                    logger.error(f"Invalid memory operation: {e} {memory_dict}")
                    continue
                except Exception as e:
                    logger.error(f"Error executing memory operation: {e} {memory_dict}")
                    continue

            logger.info(
                f"Successfully processed {len(processed_memories)} memory operations"
            )
            # Add confirmation message if any memory was added or updated
            if processed_memories:
                try:
                    from fastapi.requests import Request  # ensure import

                    # Find the last assistant message and append confirmation
                    # This is a safe operation, no error if no assistant message
                    for i in reversed(range(len(self._last_body.get("messages", [])))):
                        msg = self._last_body["messages"][i]
                        if msg.get("role") == "assistant":
                            msg["content"] += " (Memory:Saved)"
                            break
                except Exception:
                    pass
            return True
        except Exception as e:
            logger.error(f"Error processing memories: {e}\n{traceback.format_exc()}")
            return False

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

                # Compute and store embedding
                embedding = self.embedding_model.encode(
                    formatted_content, normalize_embeddings=True
                )
                # Extract new memory ID from result if possible
                # For now, skip ID extraction and skip caching
                # TODO: update with real ID extraction if API returns it

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

                    # Compute and update embedding
                    embedding = self.embedding_model.encode(
                        formatted_content, normalize_embeddings=True
                    )
                    self.memory_embeddings[operation.id] = embedding

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
        """Query LLM with retry logic"""
        max_retries = self.valves.max_retries
        retry_delay = self.valves.retry_delay
        attempt = 0

        import random

        while attempt <= max_retries:
            try:
                # Increment attempt counter
                attempt += 1

                # Validate API configuration and dispatch
                if self.valves.provider == "LM Studio":
                    if not self.valves.lmstudio_url:
                        return "Error: LM Studio URL not configured"
                    if not self.valves.lmstudio_model:
                        return "Error: LM Studio model not configured"
                    response = await self._query_openai(
                        model=self.valves.lmstudio_model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        api_url=self.valves.lmstudio_url,
                        api_key=None,
                    )
                elif self.valves.provider == "OpenRouter":
                    if not self.valves.openrouter_url:
                        return "Error: OpenRouter URL not configured"
                    if not self.valves.openrouter_api_key:
                        return "Error: OpenRouter API key not configured"
                    if not self.valves.openrouter_model:
                        return "Error: OpenRouter model not configured"
                    response = await self._query_openai(
                        model=self.valves.openrouter_model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        api_url=self.valves.openrouter_url,
                        api_key=self.valves.openrouter_api_key,
                    )
                elif self.valves.provider == "Ollama":
                    response = await self._query_openai(
                        model=self.valves.ollama_model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        api_url="http://host.docker.internal:11434",
                        api_key=None,
                    )
                else:
                    return f"Error: Unknown provider {self.valves.provider}"

                # Check if the response indicates an error that should be retried
                if response.startswith("Error:"):
                    lower_resp = response.lower()
                    if (
                        "timeout" in lower_resp
                        or "temporarily unavailable" in lower_resp
                    ):
                        logger.warning(
                            f"Timeout/unavailable error on attempt {attempt}, will retry: {response}"
                        )
                        if attempt <= max_retries:
                            sleep_time = retry_delay * (
                                2 ** (attempt - 1)
                            ) + random.uniform(0, 0.5)
                            await asyncio.sleep(sleep_time)
                            continue
                    elif (
                        "rate limit" in lower_resp
                        or "too many requests" in lower_resp
                        or "429" in lower_resp
                    ):
                        logger.warning(
                            f"Rate limit error on attempt {attempt}, backing off: {response}"
                        )
                        if attempt <= max_retries:
                            sleep_time = retry_delay * (
                                2 ** (attempt - 1)
                            ) + random.uniform(0, 1.0)
                            await asyncio.sleep(sleep_time)
                            continue

                # If we got a valid response or a non-retryable error, return it
                return response

            except Exception as e:
                logger.error(f"Error on attempt {attempt} when querying LLM: {e}")
                if attempt <= max_retries:
                    sleep_time = retry_delay * (2 ** (attempt - 1)) + random.uniform(
                        0, 0.5
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    return f"Error: Failed after {max_retries} attempts: {str(e)}"

        # Should never reach here, but just in case
        return f"Error: Failed after {max_retries} attempts"

    async def _query_ollama(
        self, model: str, system_prompt: str, user_prompt: str
    ) -> str:
        """Query Ollama API"""
        session = await self._get_aiohttp_session()
        url = "http://host.docker.internal:11434/api/chat"

        # Validate model availability if we've discovered models
        if self.available_ollama_models and model not in self.available_ollama_models:
            model_suggestion = ""
            if self.available_ollama_models:
                model_suggestion = f" Available models: {', '.join(self.available_ollama_models[:5])}..."
                if len(self.available_ollama_models) > 5:
                    model_suggestion += (
                        f" and {len(self.available_ollama_models) - 5} more"
                    )
            logger.warning(
                f"Model '{model}' not found in available Ollama models.{model_suggestion}"
            )

        # Make sure model has current date awareness
        date_context = f"\nToday's date is {self.date_info['weekday']}, {self.date_info['month']} {self.date_info['day']}, {self.date_info['year']}. Current time is {self.date_info['iso_time']}."
        enhanced_system_prompt = system_prompt + date_context

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.1  # Lower temperature for more deterministic outputs
            },
        }

        try:
            async with session.post(url, json=payload, timeout=30) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return f"Error: Ollama API returned {response.status}: {error_text}"

                data = await response.json()
                if "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                else:
                    return f"Error: Unexpected Ollama API response format: {data}"
        except asyncio.TimeoutError:
            return "Error: Ollama API request timed out"
        except ClientError as e:
            return f"Error: Ollama API connection error: {str(e)}"
        except Exception as e:
            return f"Error: Ollama API error: {str(e)}"

    async def _query_openai(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        api_url: str,
        api_key: Optional[str] = None,
    ) -> str:
        """Query OpenAI API or Ollama API"""
        # Dispatch based on provider
        if self.valves.provider == "Ollama":
            return await self._query_ollama(model, system_prompt, user_prompt)

        session = await self._get_aiohttp_session()

        # Use /completions for LM Studio, /chat/completions for others
        if self.valves.provider == "LM Studio":
            url = f"{api_url.rstrip('/')}/completions"
        else:
            url = f"{api_url.rstrip('/')}/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Validate model availability if we've discovered models
        if self.available_openai_models and model not in self.available_openai_models:
            model_suggestion = ""
            if self.available_openai_models:
                model_suggestion = (
                    f" Available models: {', '.join(self.available_openai_models[:5])}"
                )
                if len(self.available_openai_models) > 5:
                    model_suggestion += (
                        f" and {len(self.available_openai_models) - 5} more"
                    )
            logger.warning(
                f"Model '{model}' not found in available OpenAI models.{model_suggestion}"
            )

        # Make sure model has current date awareness
        date_context = f"\nToday's date is {self.date_info['weekday']}, {self.date_info['month']} {self.date_info['day']}, {self.date_info['year']}. Current time is {self.date_info['iso_time']}."
        enhanced_system_prompt = system_prompt + date_context

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,  # Lower temperature for more deterministic outputs
        }

        try:
            async with session.post(
                url, headers=headers, json=payload, timeout=30
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    return f"Error: OpenAI API returned {response.status}: {error_text}"

                data = await response.json()
                if (
                    "choices" in data
                    and len(data["choices"]) > 0
                    and "message" in data["choices"][0]
                ):
                    return data["choices"][0]["message"]["content"]
                else:
                    return f"Error: Unexpected OpenAI API response format: {data}"
        except asyncio.TimeoutError:
            return "Error: OpenAI API request timed out"
        except ClientError as e:
            return f"Error: OpenAI API connection error: {str(e)}"
        except Exception as e:
            return f"Error: OpenAI API error: {str(e)}"

    async def _add_confirmation_message(self, body: Dict[str, Any]) -> None:
        """Add a confirmation message about memory operations"""
        if (
            not body
            or "messages" not in body
            or not body["messages"]
            or not self.valves.show_status
        ):
            return

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

            # Only create a confirmation if operations were performed
            if new_count > 0 or update_count > 0 or delete_count > 0:
                # Generate a natural-sounding confirmation message
                parts = []

                if new_count > 0:
                    parts.append(
                        f"added {new_count} memor{'y' if new_count == 1 else 'ies'}"
                    )

                if update_count > 0:
                    parts.append(
                        f"updated {update_count} memor{'y' if update_count == 1 else 'ies'}"
                    )

                if delete_count > 0:
                    parts.append(
                        f"deleted {delete_count} memor{'y' if delete_count == 1 else 'ies'}"
                    )

                if parts:
                    if len(parts) == 1:
                        confirmation = f"*🧠 I've {parts[0]}*"
                    elif len(parts) == 2:
                        confirmation = f"*🧠 I've {parts[0]} and {parts[1]}*"
                    else:
                        confirmation = (
                            f"*🧠 I've {', '.join(parts[:-1])}, and {parts[-1]}*"
                        )

        if confirmation:
            # Find the last assistant message and append the confirmation
            for i in reversed(range(len(body["messages"]))):
                if body["messages"][i]["role"] == "assistant":
                    body["messages"][i]["content"] += f" {confirmation}"
                    break

    # Cleanup method for aiohttp session and background tasks
    async def cleanup(self):
        """Clean up resources on shutdown"""
        logger.info("Performing cleanup of Intelligent Memory Manager resources")

        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    # Wait briefly for task to cancel
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

        # Close aiohttp session
        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()
            self._aiohttp_session = None

        logger.info("Cleanup completed")
