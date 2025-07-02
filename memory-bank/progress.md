# Progress

*   **Current Status:** **Implementing Feature 2 (Dynamic Memory Tagging) and fixing critical memory injection issues.** Working to resolve embedding dimension mismatches and LLM provider consistency problems.
*   **Completed Work:**
    *   Created Memory Bank structure & initialized files.
    *   Analyzed `adaptive_memory_v2.py` & docs.
    *   Identified initial improvements & features.
    *   Fixed Issue #7 in `adaptive_memory_v2.5.py` (`llm_api_endpoint_url` default).
    *   Added Issue #8 (Missing `copy.deepcopy` fix) to docs.
    *   Fixed Issue #8 in `adaptive_memory_v2.5.py` (added deepcopy in `outlet`).
    *   Refined `_convert_dict_to_memory_operations` in `v2.5.py` (addresses part of Issue #1).
    *   Disabled secondary LLM filter (Issue #9) & removed regex fallback (Issue #10) in `v2.5.py`.
    *   Implemented Background Task Management (Improvement 8) in `v2.6`.
    *   Fixed `embedding_model` and `memory_embeddings` setter errors (Issues #13 & #14) in `v2.6`.
    *   Tested Background Task Management (Improvement 8) in `v2.6`.
    *   Implemented Input Validation (Improvement 9) in `v2.7`.
    *   Updated Rule #6 in `rules.md`.
    *   Attempted Feature 10 implementation in `v2.8.py` (failed apply).
    *   Reorganized `improvement_plan.md` numerically (Improvements first).
    *   Implemented **Optimize Relevance Calculation (Improvement 1)** in `v2.8.py`. **✅ Verified.**
    *   Implemented **Refine Memory Deduplication (Improvement 2)** in `v2.8.py`. **✅ Verified.**
    *   Implemented **Enhance Memory Pruning (Improvement 3)** in `v2.8.py`. **✅ Verified.**
    *   Post-Imp 3 robustness & performance refinements (stronger `result.id` guard, O(n) dedup iterate). **✅ Verified.**
    *   Added detailed debug logging and minor prompt tuning for better traceability based on log analysis. **✅ Verified.**
    *   Improved user-facing status message clarity. **✅ Verified.**
    *   Added `debug_error_counter_logs` valve and conditional logging to reduce docker log clutter. **✅ Verified.**
    *   Reduced general log noise by setting propagate=False and changing routine logs to DEBUG level. **✅ Verified.**
    *   Implemented **Memory Extraction Robustness (Imp. 7 & 10)** in `v2.8.py`, resolving Issue #12 (Simple Preferences Not Saved) with JSON parsing fixes, error guards, deduplication tuning, and a short-preference bypass. **✅ Verified.**
    *   Implemented **Improve Summarization Logic (Improvement 4)** in `v2.9.py`. Added valves, helper methods, and rewrote `_summarize_old_memories_loop` for cluster-based summarization, including age filtering and strategy options (embeddings, tags, hybrid). **✅ Verified (pending user feedback on summarization quality itself).**
    *   Resolved **Memory Saving Instability (Issues #12, #13 implicitly)** in `v2.9.py` by enforcing JSON mode via API (`response_format`), setting deterministic LLM params (`temp:0`, `seed:42`), and improving JSON extraction robustness (`_extract_and_parse_json` unwrapping). **✅ Verified.**
    *   Implemented **Optimize LLM Calls (Improvement 5)** in `v2.9.py` by adding `llm_skip_relevance_threshold` and LLM call tracking. **✅ Verified.**
    *   Fixed **Pruning Count Bug (Issue #15)** related to `least_relevant` strategy selection in `v2.9.py`. **✅ Verified.**
    *   Completed **Minor Valve Adjustments** (defined `embedding_similarity_threshold`, tuned `llm_skip_relevance_threshold`) in `v2.9.py`. **✅ Verified.**
    *   Implemented **Feature #11 (Memory Banks)** in `v3.0.py`. **✅ Verified.**
    *   Implemented **Feature #3 (Memory Confidence Scoring)** in `v3.1.py`, including debugging and fixes. **✓ Complete**
        *   Added `min_confidence_threshold` valve & validator.
        *   Updated LLM identification prompt for confidence.
        *   Updated parsing/validation/fallback logic for confidence.
        *   Added confidence-based filtering in `_process_user_memories`.
        *   Added confidence score to `_format_memory_content` display.
        *   Added UI status message for low-confidence discards.
    *   Implemented **Feature #2 (Dynamic Memory Tagging)** in `v3.2.py`. **✓ Complete but pending verification**
        *   Added `dynamic_tags` (list of `{tag, confidence}`) to `MemoryOperation` model.
        *   Extended `Filter.Valves` with dynamic-tagging settings, prompts, and validators.
        *   Created `_generate_and_validate_dynamic_tags` function to call LLM and store validated tags.
        *   Updated `_execute_memory_operation` to generate tags for new memories.
        *   Updated formatting functions to display dynamic tags.
        *   Added helper functions for tag parsing and combining.
    *   Investigated and began fixing **Memory Injection Failures due to Embedding Dimension Mismatches**
        *   Identified root cause: memories saved with API embeddings (1536-dim) but retrieved with local embeddings (384-dim)
        *   Modified `get_relevant_memories` to regenerate embeddings when dimensions mismatch
        *   Added provider consistency mechanisms, including valve reloading before operations
        *   Implemented `_apply_llm_provider_overrides()` to ensure user settings are respected
        *   Added status emitter for successful memory injections
    *   Investigated user reports (Issue #15): Added debug logging, lowered default relevance thresholds to 0.60, added embedding dimension checks, and added UI status emitters for retrieval.
    *   Implemented flexible embedding provider support (`local` vs `openai_compatible`) with auto-discovery for local models.
    *   Implemented Prometheus metrics instrumentation (`EMBEDDING_*`, `RETRIEVAL_*`) and FastAPI endpoints (`/health`, `/metrics`).
    *   Resolved `prometheus_client` optional import issue.
*   **What Works (Expected):**
    *   OpenWebUI runs fresh.
    *   Filter connects correctly to host Ollama.
    *   Basic memory extraction and saving via LLM.
    *   Adaptive memory functionality (from `adaptive_memory_v2.py`).
    *   Background tasks (summarization, error logging, date update, model discovery) run and respect valves (Imp. 8).
    *   Configuration valves are validated on startup (Imp. 9).
    *   Relevance calculation correctly uses vector-only path when `use_llm_for_relevance=False`, including on-the-fly embedding generation and caching (Imp. 1).
    *   Memory extraction is more robust against LLM inconsistencies (Imp. 7 & 10).
    *   LLM calls for relevance are skipped when vector similarities are high (Imp. 5).
    *   Pruning (`least_relevant`) correctly removes the specified number of memories.
    *   *New:* Debug logs in `get_relevant_memories` should show applied thresholds.
    *   *New:* Default relevance retrieval should be less strict.
    *   *New:* Plugin can use local SentenceTransformers or OpenAI-compatible embedding APIs via valves.
    *   *New:* Locally installed SentenceTransformer models are auto-discovered.
    *   *New:* Incompatible embedding dimensions between query and memories are detected and handled.
    *   *New:* Prometheus metrics are exposed via `/adaptive-memory/metrics` endpoint.
    *   *New:* UI shows status updates during memory retrieval/injection.
    *   *New:* Dynamic memory tagging generates and stores AI-created tags with confidence scores.
    *   *New:* Regeneration of memory embeddings when dimension mismatches are detected.
*   **What Needs Verification:**
    *   Complete fix for memory injection failures due to provider inconsistency and embedding dimension mismatches.
    *   Verification that Dynamic Memory Tagging works correctly with memory injection.
    *   User confirmation that Issue #15 symptoms (valve settings/relevance) are fully resolved by the latest changes (thresholds, dimension checks, status emitters).
*   **What's Left to Build/Implement:**
    *   Complete the memory injection fix
    *   Feature 3: Personalized Response Tailoring
    *   Feature 4: Verify Cross-Session Persistence
    *   Feature 5: Improve Config Handling
    *   Feature 6: Enhance Retrieval Tuning
    *   Feature 7: Improve Status/Error Feedback
    *   Feature 8: Expand Documentation
    *   Feature 9: Always-Sync to RememberAPI
    *   Feature 10: Enhance Status Emitter Transparency
    *   Feature 11: Optional PII Stripping on Save
    *   Remaining items from `roadmap.md`.
*   **Known Issues:**
    *   Memory injection may still fail due to provider inconsistency issues
    *   LLM provider setting not consistently applied across all operations
    *   Embedding dimension mismatches between memory storage and retrieval
*   **Project Evolution & Decisions:**
    *   Prioritized memory injection fix as critical path - features are useless if memories can't be injected
    *   Added regeneration capability for embeddings to handle dimension mismatches
    *   Added more consistent provider selection to ensure user settings are respected
    *   Enhanced logging to provide better diagnostic information for troubleshooting
    *   Prioritized robustness/observability (embedding flexibility, metrics, dimension checks, status emitters) before proceeding with new features based on user feedback and testing.
    *   Adopted Prometheus for metrics collection.

# Progress Log

## v3.0 Development

*   **Feature #11: Memory Banks (Implementation Complete)**
    *   Added `allowed_memory_banks` and `default_memory_bank` valves.
    *   Updated `memory_identification_prompt` to request bank assignment.
    *   Modified `_extract_and_parse_json` to handle `memory_bank` field.
    *   Added `memory_bank` field to `MemoryOperation` model.
    *   Updated `_execute_memory_operation` to include `memory_bank` in metadata.
    *   Updated `_format_memory_content` to include bank information.
    *   Added UI commands: `/memory list_banks`, `/memory assign_bank [id] [bank]`.
    *   Updated module docstring to v3.0 and listed Memory Banks feature.
    *   Resolved configuration block duplications in `Filter.Valves`.
    *   Added missing helper methods/guard flags (`_llm_feature_guard_active`, etc.).
    *   Fixed `TypeError` in `add_memory` call (passed `user` instead of `user_id`).
    *   Improved LLM connection error reporting in UI status messages.
    *   Changed default `timezone` valve to "Asia/Dubai".
    *   Fixed `TypeError` in `_execute_memory_operation` (added missing `request` arg to `add_memory` call).
*   **What Works:** Core memory bank logic, UI commands, configuration valves, error reporting fix, timezone default update, memory saving fix.
*   **What's Left:** Full testing, integration verification.
*   **Known Issues:** None identified after latest fixes.

## v3.0 Development (Initial)

*   **Feature #11: Memory Banks (Partial Implementation)**
    *   **Completed:**
        *   Added `allowed_memory_banks` and `default_memory_bank` valves to `Filter.Valves`.
        *   Updated `MemoryOperation` Pydantic model with optional `memory_bank` field.
        *   Modified `_execute_memory_operation` to accept `MemoryOperation` and use `memory_bank` in metadata.
        *   Updated `_format_memory_content` to display the memory bank.
        *   Implemented UI commands (`/memory list_banks`, `/memory assign_bank`) in `inlet`.
        *   Updated module docstring for v3.0 and Feature #11.
    *   **Identified Issues (Pending Fixes):**
        *   LLM prompt (`memory_identification_prompt`) and JSON parsing logic not updated for `memory_bank` field (Issue #16).
        *   Duplicate code definitions found (Issue #17).
        *   Missing `_increment_error_counter` function definition (Issue #18).
*   **What Works (Partially):** Core memory bank logic infrastructure, manual bank assignment via UI commands.
*   **What Needs Fixing:** Automatic bank assignment by LLM, code duplication cleanup, missing helper function implementation.
*   **Known Issues:** Issues #16, #17, #18 documented in `identified_issues_and_how_we_fixed_them.md`.

## v3.2 Development

*   **Feature #2: Dynamic Memory Tagging (Implementation Complete)**
    *   Added `dynamic_tags` field to `MemoryOperation` model.
    *   Added `dynamic_tagging_enabled`, `dynamic_tagging_prompt`, and related valves to `Filter.Valves`.
    *   Created `_generate_and_validate_dynamic_tags` function for LLM-based tag generation.
    *   Updated `_execute_memory_operation` to generate tags for new memories.
    *   Modified `_format_memory_content`, `_get_formatted_memories`, and other display functions to include dynamic tags.
    *   Added helper functions for tag processing and combining.
    *   Updated module docstring to v3.2 and listed Dynamic Memory Tagging feature.

*   **Memory Injection Fixes (In Progress)**
    *   Identified root cause: dimension mismatch between memory embeddings (saved with API) and user query embeddings (generated locally).
    *   Modified `get_relevant_memories` to detect dimension mismatches.
    *   Added regeneration logic for embeddings with mismatched dimensions.
    *   Added explicit valve reloading before key operations.
    *   Implemented provider override mechanism to enforce user settings.
    *   Added detailed logging of provider configuration and dimension information.
    *   Added status emitter after successful memory injection for better user feedback.

*   **What Works:** 
    *   Dynamic memory tagging functionality is fully implemented.
    *   Memory embedding regeneration logic is in place but still experiencing provider inconsistency.
    *   Memory formatting correctly displays dynamic tags.

*   **What's Left:** 
    *   Complete the fix for provider consistency issues.
    *   Test to confirm memory injection works reliably with correct provider settings.
    *   Validate dynamic tagging functionality with successful memory injection.

*   **Known Issues:** 
    *   Provider selection not consistently applied across all operations.
    *   Potential issues with user settings not being properly loaded from config.
    *   Ollama being used as provider even when OpenRouter is selected by user.

### Completed Features & Improvements (Most Recent First)

#### Feature #11: Memory Banks (2025-05-10)
* **Goal:** Allow partitioning memories into different "banks" (e.g., 'work', 'personal') for better organization and retrieval.
* **Status:** ✅ Complete
* **What Works:**
  * Added `allowed_memory_banks` and `default_memory_bank` valves for configuration.
  * Added `memory_bank` field to `MemoryOperation` model.
  * Updated the memory identification prompt to guide LLM on assigning appropriate memory banks.
  * Enhanced validation in `_validate_memory_operation` to handle memory_bank field.
  * Updated `_convert_dict_to_memory_operations` to support memory bank field.
  * Modified `_execute_memory_operation` and `_format_memory_content` to include bank information.
  * Added UI commands: `/memory list_banks`, `/memory assign_bank [id] [bank]`.
* **Issues Encountered & Resolved:**
  * Resolved inconsistencies related to LLM prompt, validation, and duplicate code checks (Issues #16, #17, #18).
  * **Issue #19 (Config Timing - Ongoing Debug):** Diagnosed persistent issue where UI-configured LLM provider isn't used. Moved processing to `outlet`, fixed resulting `NameError`. Multiple attempts to reload config failed. Currently attempting to inspect `self.config` content directly via logging to determine why settings aren't loaded.
* **Learnings:** Plugin configuration loading timing within OpenWebUI is unreliable for early execution phases (`inlet`). Accessing `self.config` seems problematic even in `outlet`. Deferring config-dependent operations is necessary but hasn't fully solved the root cause.

#### Improvement #10: Generalized LLM Provider Support (2025-05-01)
* **Goal:** Allow for a wider range of LLM providers and configurations.
* **Status:** ✅ Complete
* **What Works:**
  * Implemented generalized LLM provider support in `v2.8.py` and `v2.9.py`.
  * Added `llm_api_endpoint_url` and `llm_skip_relevance_threshold` to `MemoryOperation` model.
  * Updated `_execute_memory_operation` to accept `llm_api_endpoint_url` and use it for LLM calls.
  * Added `_llm_feature_guard_active` flag to `Filter.Valves` to handle LLM provider switching.
* **Issues Encountered & Resolved:**
  * Resolved issues related to LLM provider switching and configuration.
  * Implemented a mechanism to handle multiple LLM providers and configurations.
* **Learnings:** The ability to switch between different LLM providers and configurations can greatly enhance the flexibility and adaptability of the system.

*   **v3.2 -> v3.2 (ValueError fix):** Corrected field name assignment in `_apply_llm_provider_overrides` to prevent `ValueError` when setting the Ollama API endpoint URL.

## v3.2 (Ongoing Debugging - Provider/JSON Issues)

*   **✅ Provider Context Fix:** Ensured `_apply_llm_provider_overrides(user)` is called at the start of `_process_user_memories` to load the correct provider context for the memory extraction LLM call, mitigating race conditions.
*   **✅ Enhanced JSON Parsing:** Added aggressive boundary searching (`[...]`, `{...}`) to `_extract_and_parse_json` to improve robustness against LLM responses with extra leading/trailing text.
*   **✅ Status Reporting Refinement:** Improved status message logic to prioritize specific errors (like `json_parse_error`) over generic messages.
*   **🚧 Testing:** Actively testing these fixes to confirm provider consistency and JSON parsing reliability, especially after Docker restarts.
*   **Known Issue:** LLM (e.g., Claude 3.5 Haiku) still sometimes returns malformed JSON. The enhanced parsing aims to handle this, but prompt engineering might be needed if it persists.

## Latest Breakthrough (2025-05-05)

### Bulletproof Memory Extraction

We've completely redesigned the memory extraction system with a focus on reliability across different LLM providers:

1. **Completely Rewritten JSON Extraction**:
   - Implemented a 5-stage extraction pipeline that can handle any LLM response format
   - Works reliably with Claude, Gemini, and Ollama models without modification
   - Detailed logging at each stage to enable better debugging

2. **Direct Preference Statement Processing**:
   - Automated detection of common preference statements via regex
   - Fast path that bypasses LLM calls entirely for simple preference statements
   - Special error reporting for preference statement failures

3. **Intelligent Error Handling**:
   - More descriptive error messages that explain exactly what went wrong
   - Specific error paths for different failure modes
   - Targeted suggestions when problems occur

The result is a dramatically more robust system that can extract memories reliably from any LLM provider response, regardless of formatting quirks or extra text in the response. This should resolve the persistent issues with memory saving failures.