# Project Brief

*   **Core Goal:** Develop a self-contained OpenWebUI plugin ("Adaptive Memory") that provides personalized, persistent, and adaptive memory capabilities for LLMs within the platform. It should dynamically extract, store, retrieve, and inject user-specific information to enable context-aware, evolving conversations.
*   **Key Requirements:**
    *   Extract user facts, preferences, goals, interests via LLM prompts and conversation history.
    *   Filter irrelevant data (trivia, meta-requests) using regex, LLM classification, keywords.
    *   Implement multi-layer filtering (blacklist/whitelist, thresholds).
    *   Support configurable memory categories (identity, behavior, preference, goal, relationship, possession).
    *   Deduplicate and summarize memories over time (using embeddings/SequenceMatcher and LLM summarization).
    *   Inject relevant, concise memories into LLM prompts efficiently.
    *   Filter meta-explanations from LLM output.
    *   All settings configurable via OpenWebUI Valves.
    *   Must be a self-contained OpenWebUI Filter plugin.
    *   No external dependencies beyond Python stdlib and specified libraries (sentence-transformers, numpy, aiohttp, etc.).
*   **Scope:** Focus solely on the Adaptive Memory plugin functionality within OpenWebUI. Excludes external databases, servers, or UI changes outside the plugin's scope.
*   **Success Metrics:**
    *   Improved personalization and context-awareness in LLM conversations over time.
    *   Accurate extraction and storage of user-specific information.
    *   Effective filtering of irrelevant data.
    *   Robustness and stability within the OpenWebUI environment.
    *   Compliance with OpenWebUI plugin architecture and constraints.
    *   Positive user feedback on memory effectiveness. 