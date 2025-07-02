"""
Adaptive Memory v3.1 - Advanced Memory System for OpenWebUI
Author: AG

---

# Overview

Adaptive Memory is a sophisticated plugin that provides **persistent, personalized memory capabilities** for Large Language Models (LLMs) within OpenWebUI. It enables LLMs to remember key information about users across separate conversations, creating a more natural and personalized experience.

The system **dynamically extracts, filters, stores, and retrieves** user-specific information from conversations, then intelligently injects relevant memories into future LLM prompts. Memory should persist for a user across different chat sessions.

---

# Key Features

1.  **Intelligent Memory Extraction:** Automatically identifies facts, preferences, relationships, goals, etc., using an LLM.
2.  **Configurable Storage & Retrieval:** Uses embedding similarity and relevance scoring (vector-based or LLM-enhanced) with tunable thresholds.
3.  **Multi-layered Filtering:** Deduplication (semantic/text), confidence scoring, min length, trivia/topic filtering.
4.  **Memory Banks:** Categorize memories (e.g., Personal, Work) for focused retrieval.
5.  **User Controls:** Enable/disable via user settings, manage memories via commands.
6.  **Flexible Embeddings:** Supports local SentenceTransformers or OpenAI-compatible APIs.
7.  **Observability:** Prometheus metrics and health endpoints.
8.  **Adaptive Management:** Summarization and pruning options.

---

# Recent Improvements (v3.0)

1.  Optimized Relevance Calculation (Vector-only option, LLM skip threshold)
2.  Enhanced Deduplication (Embedding-based option)
3.  Intelligent Memory Pruning (FIFO / Least Relevant options)
4.  Cluster-Based Summarization
5.  LLM Call Optimization (High-confidence vector skip)
6.  Resilient JSON Parsing (Fallbacks, structured output requests)
7.  Background Task Management Valves
8.  Input Validation for Valves
9.  Refined Filtering Logic (Defaults, shortcuts)
10. Generalized LLM Provider Support (Feature #12)
11. Memory Banks (Feature #11)
12. Fixed Config Persistence (Issue #19)

# Recent Improvements (v3.1)

1.  Memory Confidence Scoring & Filtering
2.  Flexible Embedding Provider Support (Local/API Valves)
3.  Local Embedding Model Auto-Discovery
4.  Embedding Dimension Validation
5.  Prometheus Metrics Instrumentation
6.  Health & Metrics Endpoints (`/adaptive-memory/health`, `/adaptive-memory/metrics`)
7.  UI Status Emitters for Retrieval
8.  Debugging & Robustness Fixes (Issue #15 - Thresholds, Visibility)
9.  Minor Fixes (`prometheus_client` import)
10. User Guide Section (Consolidated Docs in Docstring)

---

# --- Adaptive Memory Plugin: User Guide --- #

## Overview
Adaptive Memory allows your assistant to remember user-specific information across chat sessions, creating more natural and personalized interactions. It learns facts, preferences, and goals from your messages, stores them, and injects relevant snippets back into the conversation context.

**Persistence:** Memories are tied to your user ID and *should* be available across different chat sessions you start. If this doesn't seem to be happening, ensure the plugin is enabled in your OpenWebUI user settings.

## How It Works
1.  **Learning:** When you send a message, the plugin analyzes it (using an LLM) to extract potential memories (facts, preferences, goals). Status: `📝 Extracting...`
2.  **Filtering:** Extracted memories are filtered based on confidence score, length, relevance, and configured blacklists/whitelists. Duplicates are also removed.
3.  **Storing:** Valid new memories are saved with tags (e.g., `identity`, `preference`), a confidence score, and assigned to a memory bank (e.g., `General`, `Personal`, `Work`). Status: `✅ Added X new memories...` or `⚠️ Memory save skipped...` (if filtered/duplicate).
4.  **Retrieving:** Before the assistant generates a response, the plugin searches your saved memories for snippets relevant to the current conversation using semantic similarity. Status: `🧠 Retrieving...`
5.  **Injecting:** The most relevant memories are added to the system prompt (invisible to you unless `show_memories` is enabled) to give the assistant context about you. Status: `✅ Injecting X memories...`

## Configuration (Valves)
Adjust these settings via `Admin Panel` → `Plugins` → `Adaptive Memory`:

*   **Core Settings**
    *   `enabled` (bool): Globally enable/disable the plugin *for your user*. (Default: true)
    *   `show_status` (bool): Show status indicators (📝, 🧠, ✅, ⚠️) during operations. Helps with troubleshooting. (Default: true)
    *   `show_memories` (bool): Display the actual text of injected memories in the chat context (mainly for debugging). (Default: true)
    *   `memory_format` ("bullet" | "paragraph" | "numbered"): How injected memories are formatted if shown. (Default: "bullet")

*   **Embedding & Relevance (Tuning Retrieval)**
    *   `embedding_provider_type` ("local" | "openai_compatible"): Where to get embeddings from. "local" uses SentenceTransformers (ensure installed); "openai_compatible" uses an API. (Default: "local")
    *   `embedding_model_name` (str): Model name. For "local", it's the SentenceTransformer name (e.g., `all-MiniLM-L6-v2`). For "openai_compatible", it's the API model ID (e.g., `text-embedding-3-small`). Auto-discovery attempts local models in `~/.cache/torch/sentence_transformers`. (Default: "all-MiniLM-L6-v2")
    *   `embedding_api_url` / `embedding_api_key`: Needed only if using "openai_compatible".
    *   `vector_similarity_threshold` (0-1): **Key Tuning Knob.** Initial similarity cutoff. Lower this (e.g., 0.4-0.55) if retrieval feels too strict or memories aren't being found often enough. Higher values require closer matches. (Default: 0.60)
    *   `relevance_threshold` (0-1): **Key Tuning Knob.** Final score cutoff after optional LLM check. Also lower this if retrieval is too strict, matching the `vector_similarity_threshold` if `use_llm_for_relevance` is false. (Default: 0.60)
    *   `use_llm_for_relevance` (bool): Use a second LLM call to score relevance after vector search (more accurate but slower/costlier). If false, uses vector score directly against `relevance_threshold`. (Default: false)
    *   `llm_skip_relevance_threshold` (0-1): If `use_llm_for_relevance` is true, skip the LLM call anyway if all vector scores are *already* above this confidence level. (Default: 0.93)

*   **Deduplication**
    *   `deduplicate_memories` (bool): Prevent saving near-duplicates. (Default: true)
    *   `use_embeddings_for_deduplication` (bool): Use semantic similarity (embeddings) vs. basic text similarity for deduplication. (Default: true)
    *   `embedding_similarity_threshold` (0-1): Cutoff for embedding-based duplicates (usually high, e.g., 0.97).
    *   `similarity_threshold` (0-1): Cutoff for text-based duplicates (if embedding dedupe is off). (Default: 0.95)

*   **Memory Management**
    *   `max_total_memories` (int): Max memories before pruning oldest or least relevant. (Default: 200)
    *   `pruning_strategy` ("fifo" | "least_relevant"): How to prune when full. (Default: "fifo")
    *   `min_memory_length` (int): Minimum characters needed to save a memory. (Default: 8)
    *   `min_confidence_threshold` (0-1): Discard extracted memories below this LLM-assigned confidence. Raise if too many uncertain memories are saved; lower if potentially useful inferences are discarded. (Default: 0.5)
    *   `enable_summarization_task` / `summarization_interval` / etc.: Settings for background summarization of old memories.

*   **Content Filtering**
    *   `filter_trivia` (bool): Try to auto-filter common knowledge. (Default: true)
    *   `blacklist_topics` / `whitelist_keywords` (csv): Comma-separated terms to ignore or force-save.

*   **LLM Provider (for Extraction/Summarization)**
    *   `llm_provider_type` ("ollama" | "openai_compatible"), `llm_model_name`, `llm_api_endpoint_url`, `llm_api_key`: Configure the LLM used for internal tasks (memory extraction, relevance scoring if enabled, summarization).

*   **Memory Banks**
    *   `allowed_memory_banks` (list): Which banks are valid (e.g., ["General", "Personal", "Work"]). Edit this list to add/remove banks.
    *   `default_memory_bank` (str): Bank used if LLM doesn't specify a valid one from the allowed list. (Default: "General")

## Troubleshooting
*   **Memories not saving?**
    *   Check `show_status`: Look for `📝` followed by `⚠️ Memory save skipped`. Reasons include: too short (`min_memory_length`), low confidence (`min_confidence_threshold`), filtered (`filter_trivia`, blacklist), duplicate (`deduplicate_memories`).
    *   Look for `⚠️ Memory error: json_parse_error`. This means the LLM extracting memories gave an invalid response. Check OpenWebUI server logs for the raw response. Improving the `memory_identification_prompt` or using a more capable `llm_model_name` for extraction might help.
    *   Ensure the memory plugin is enabled in your main OpenWebUI user settings.

*   **Memories not being injected/retrieved?**
    *   Check `show_status`: Do you see `🧠 Retrieving...`? If not, check user settings.
    *   If retrieval happens but nothing is injected (`✅ Injecting 0 memories...`), it likely means no saved memories met the relevance criteria for the current message. Try lowering `vector_similarity_threshold` and `relevance_threshold` (e.g., to 0.5 or 0.45) and test again. Remember relevance depends on the *current message* context.
    *   Ensure you are in a new chat session *with the same user* where memories were previously saved. Memory is user-specific, not global.

*   **Status stuck on "Extracting..."?** 
    This might indicate an issue with the LLM configured for extraction (`llm_model_name`, `llm_api_endpoint_url`) or a timeout. Check OpenWebUI server logs.

*   **Seeing relevance/similarity threshold warnings in logs?** 
    This usually means the `vector_similarity_threshold` and `relevance_threshold` values might be slightly different, which is fine if `use_llm_for_relevance` is `true`, but can cause confusion if it's `false`. Ensure they match if you're using vector-only relevance.

*   **Local embedding model errors ('Failed to load', 'Dimension mismatch')?**
    *   Ensure the `embedding_model_name` in Valves corresponds to a model correctly downloaded by `sentence-transformers` (usually in `~/.cache/torch/sentence_transformers`).
    *   Dimension mismatch errors can occur if you change the embedding model after some memories have already been saved with embeddings from a *different* dimension model. Currently, there's no automatic fix for this; you might need to clear old memories or manually manage the transition.

---

# Roadmap
The following improvements and features are planned (see `roadmap.md` for details):

1.  Refactor Large Methods: Improve code readability.
2.  Dynamic Memory Tagging: Allow LLM to generate keyword tags.
3.  Personalized Response Tailoring: Use preferences to guide LLM style.
4.  Verify Cross-Session Persistence: Confirm memory availability across sessions.
5.  Improve Config Handling: Better defaults, debugging for Valves.
6.  Enhance Retrieval Tuning: Improve semantic relevance beyond keywords.
7.  Improve Status/Error Feedback: More specific UI messages & logging.
8.  Expand Documentation: More details in User Guide.
9.  Always-Sync to RememberAPI (Optional): Provide an **optional** mechanism to automatically sync memories to an external RememberAPI service **in addition to** storing them locally in OpenWebUI. This allows memory portability across different tools that support RememberAPI (e.g., custom GPTs, Claude bots) while maintaining the local memory bank. **Privacy Note:** Enabling this means copies of your memories are sent externally to RememberAPI. Use with caution and ensure compliance with RememberAPI's terms and privacy policy.
10. Enhance Status Emitter Transparency: Improve clarity and coverage.
11. Optional PII Stripping on Save: Automatically detect and redact common PII patterns before saving memories.

---

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` (if available) or standard open-source contribution guidelines.

---

# License

This project is licensed under the MIT License.
