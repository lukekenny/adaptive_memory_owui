# System Patterns

*   **Architecture Overview:** The system is an OpenWebUI "Filter" Function plugin implemented as a single Python script (`adaptive_memory_v2.py`). It intercepts user input (`inlet`) and LLM output (`outlet`) streams using asynchronous operations (`asyncio`).
*   **Key Components:**
    *   `Filter` Class: Main plugin logic container.
    *   `Valves` / `UserValves` (Pydantic Models): Global and user-specific configuration managed via OpenWebUI admin panel.
    *   Background Tasks (`asyncio`): Handles periodic memory summarization, error logging, date updates, and LLM model discovery.
    *   Embedding Model: Sentence Transformer (`all-MiniLM-L6-v2`) singleton for generating memory embeddings.
    *   In-Memory Stores: Dictionaries for `memory_embeddings` (ID -> vector) and `relevance_cache` (message/memory hash -> score).
    *   Memory Identification (`identify_memories`): Uses LLM prompts (including existing memories, category config, datetime) to extract potential memories.
    *   JSON Parsing (`_extract_and_parse_json`): Robustly extracts JSON from LLM responses, handling common malformations.
    *   Memory Validation (`_validate_memory_operation`): Ensures extracted operations have correct format and required fields.
    *   Filtering Pipeline (`_process_user_memories`): Applies multiple layers (length, blacklist/whitelist, regex trivia, meta-request phrases, LLM META/FACT classification) to extracted memories.
    *   Relevance Assessment (`get_relevant_memories`): **Conditional process based on `use_llm_for_relevance` valve:**
        1.  **Always:** Pre-filters memories using cosine similarity > `vector_similarity_threshold`, takes top `top_n_memories`.
        2.  **If `use_llm_for_relevance` is `False`:** Uses vector similarity score directly, filters by `relevance_threshold`, returns top `related_memories_n`.
        3.  **If `use_llm_for_relevance` is `True`:** Checks `relevance_cache`; calls LLM (`memory_relevance_prompt`) for uncached, vector-filtered memories; updates cache; filters by `relevance_threshold`; returns top `related_memories_n`.
    *   Deduplication Logic (`process_memories`, `_calculate_memory_similarity`): Prevents storing NEW memories too similar (Jaccard + SequenceMatcher) to existing ones.
    *   Memory Execution (`process_memories`, `_execute_memory_operation`): Interacts with OpenWebUI's memory API (`add_memory`, `delete_memory_by_id`) to perform validated NEW/UPDATE/DELETE operations.
    *   Memory Formatting (`_format_memory_content`, `_format_memories_for_context`): Adds tags to stored memories and formats relevant memories (bullet, numbered, paragraph) for injection.
    *   Context Injection (`_inject_memories_into_context`): Adds formatted relevant memories and anti-meta-comment instructions to the system prompt.
    *   Output Filtering (`inlet`): Removes specific meta-explanations from previous assistant messages before injection.
    *   LLM Interaction (`query_llm_with_retry`, `_query_openai`, `_query_ollama`): Handles API calls to configured provider (OpenRouter, Ollama) with retries and date context injection.
*   **Data Flow:**
    1.  `inlet`: Receives user message, chat history, user info.
    2.  Filter meta-comments from previous assistant message(s) in history.
    3.  Call `get_relevant_memories`:
        a.  Retrieve all user memories via OpenWebUI API.
        b.  Generate embeddings for user message and memories (if not already stored).
        c.  Filter memories by vector similarity (cosine sim > `vector_similarity_threshold`), take top `top_n_memories`.
        d.  **If `use_llm_for_relevance` is `False`:**
            i. Filter results from (c) using vector score >= `relevance_threshold`.
            ii. Return top `related_memories_n`.
        e.  **If `use_llm_for_relevance` is `True`:**
            i. Check `relevance_cache` for memories from (c).
            ii. Call LLM (`memory_relevance_prompt`) for uncached memories to get relevance scores.
            iii. Update `relevance_cache`.
            iv. Filter memories by score >= `relevance_threshold`.
            v. Return top `related_memories_n`.
    4.  If relevant memories found and `show_memories` is true, call `_inject_memories_into_context`:
        a.  Format memories based on `memory_format`.
        b.  Prepend anti-meta-comment instruction.
        c.  Inject into system prompt.
    5.  OpenWebUI sends modified prompt to LLM.
    6.  `outlet`: Receives LLM response and original input body.
    7.  If user message exists, start `_process_user_memories` task in background:
        a.  Call `identify_memories`:
            i.  Construct system prompt with instructions, examples, enabled categories, existing memories, datetime.
            ii. Call LLM (`query_llm_with_retry`) with recent user messages.
            iii. Parse/validate JSON response (`_extract_and_parse_json`, `_validate_memory_operation`).
        b.  Filter extracted operations (length, blacklist/whitelist, trivia regex, meta phrases).
        c.  Apply confidence filtering using `min_confidence_threshold` valve (discard memories with low confidence scores).
        d.  Emit status message if any memories were filtered due to low confidence.
        e.  Enforce `max_total_memories` (prune oldest existing if needed).
        f.  Call `process_memories`:
            i.  If `deduplicate_memories`, check NEW operations against existing using `_calculate_memory_similarity` > `similarity_threshold`.
            ii. For non-duplicate/UPDATE/DELETE ops, call `_execute_memory_operation`:
                *   Format content with tags (`_format_memory_content`).
                *   Call OpenWebUI `add_memory`/`delete_memory_by_id`.
                *   Update in-memory `memory_embeddings`.
                *   Attempt cache invalidation.
    8.  Await `_process_user_memories` task completion.
    9.  If `show_status`, call `_add_confirmation_message` to append status (e.g., "🧠 I've added 1 memory") to the assistant message.
    10. Return final body (with potentially modified assistant message) to OpenWebUI.
*   **Design Patterns:** Plugin architecture (Filter Function), Singleton pattern for embedding model, Configuration via Pydantic models (Valves), Background tasks (`asyncio`), Caching, Retry logic with exponential backoff.
*   **Critical Paths:** Memory extraction LLM calls, Relevance assessment (**conditional vector/LLM path**), Filtering logic, Memory storage/retrieval API calls, Prompt injection formatting, JSON parsing/validation.
*   **Interfaces:** `inlet`/`outlet` methods with OpenWebUI. Internal interaction with OpenWebUI memory APIs (`add_memory`, `delete_memory_by_id`, `Memories.get_memories_by_user_id`) and User model (`Users.get_user_by_id`). External interaction with LLM API (OpenRouter/Ollama) and Sentence Transformer model. 