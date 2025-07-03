# Active Context: Adaptive Memory Plugin Enhancement

### Current Focus
Completed Task #4: Resolve LLM Connection Issues in v4.0

### Latest Changes (v4.0 - 2025-07-03)
*   **Resolved LLM Connection Issues (Task #4)**:
    - Implemented circuit breaker pattern for endpoint reliability
    - Added comprehensive health check system with configurable intervals
    - Enhanced session management with connection pooling and resource optimization
    - Improved retry logic with exponential backoff and adaptive jitter
    - Added connection timeout and pool size configuration
    - Enhanced error handling and categorization for better debugging
    - Added connection cleanup and monitoring capabilities
*   **Created `adaptive_memory_v4.0.py`** by copying from `adaptive_memory_v3.1.py`
*   **Implemented synchronous Filter Function methods** for OpenWebUI compatibility:
    - `inlet(self, body: dict) -> dict` - Synchronous wrapper around async_inlet
    - `outlet(self, body: dict) -> dict` - Synchronous wrapper around async_outlet  
    - `stream(self, event: dict) -> dict` - Pass-through for streaming events
*   **Renamed async methods** to `async_inlet()` and `async_outlet()` to avoid naming conflicts
*   **Added proper event loop handling** to run async methods from sync context
*   **Ensured error safety** - all methods catch exceptions and return unchanged data
*   **Maintained v3.1 functionality** - all existing features preserved
*   **Implemented User Isolation (Task #2)**:
    - Fixed inconsistent user context handling in `_execute_memory_operation`
    - Standardized all memory API calls to use `user_id` parameter
    - Added comprehensive user_id validation in all critical methods
    - Enhanced logging to include user_id for better traceability
    - Ensured complete isolation between different users' memories

### Previous v3.1 Changes
*   Implemented Feature 2: Dynamic Memory Tagging - added support for automatic, AI-generated content tags with confidence scores
*   Added `dynamic_tags` field to `MemoryOperation` class for storing AI-generated tags
*   Extended `Filter.Valves` with dynamic-tagging settings, prompts, and validator functions
*   Created `_generate_and_validate_dynamic_tags` function to generate and store validated tags
*   Updated memory formatting functions to include and display dynamic tags
*   Fixed critical memory injection failures due to embedding dimension mismatches between save/retrieval operations
*   Added regeneration logic for memory embeddings with mismatched dimensions
*   Improved provider consistency with explicit valve reloading before operations
*   Added `_apply_llm_provider_overrides()` mechanism to ensure user-selected providers are consistently used
*   Implemented status emitter blocks for UI feedback after successful memory injections

### Current Strategy & Decisions
*   Fix priority: Memory injection must work reliably before other features matter
*   Provider consistency is critical - user settings should always override defaults
*   Embedding dimension handling requires careful validation and potentially regeneration
*   Dynamic tagging functionality is complete but less valuable until memory injection works properly

### Next Steps
1.  **Complete the fix for memory injection failures** by resolving remaining provider consistency issues
2.  **Validate the Dynamic Memory Tagging feature** once memory injection is working properly
3.  **Update logs** to include more diagnostic information about provider selection and embedding dimensions

### Important Patterns/Learnings
*   Provider selection should be dynamically updated before each operation, not just at initialization time
*   Configuration consistency is essential when working with vector operations (embeddings)
*   User settings and environment variables should take precedence over default configurations
*   Adding proper diagnostic logs helps identify issues in complex, multi-stage operations
*   Status emitters improve user experience by providing clear feedback when operations succeed or fail

### Memory Bank Focus
Memory banks are now fully supported in the adaptive memory system, allowing users to organize memories into distinct categories like "Personal", "Work", and "General".

## Memory Bank Implementation (v3.0)

*   **Current Task:** Add Memory Bank feature (Feature #11).
*   **Status:** Implementation complete. Fixed `TypeError` in `add_memory` (missing `request` arg), improved LLM error reporting, default timezone set.
*   **Next Steps:** Verify OWUI plugin configuration for OpenRouter. Test memory saving.
*   **Key Decisions:** 
    *   Used `[Memory Bank: <BankName>]` format for tagging.
    *   Added `/memory list_banks` and `/memory assign_bank` commands.
    *   Ensured `add_memory` call uses the correct `user` object parameter.
    *   Propagated specific LLM connection error codes for better UI feedback.
    *   Set default timezone to 'Asia/Dubai'.
*   **Learnings:** `add_memory` requires a `request` object. OWUI plugin config overrides internal defaults.

- Status: Fixed ValueError when setting Ollama API endpoint in provider override logic.

## Current Focus & Next Steps

*   **Status:** Just applied fixes for provider context race conditions and enhanced JSON parsing robustness.
*   **Immediate Goal:** Test the latest changes (`v3.2`) with the provided Python preference prompt after a Docker restart to confirm:
    *   The correct LLM provider (e.g., OpenRouter) is used for memory extraction.
    *   Memory is saved successfully OR a specific JSON parsing error status is shown.
*   **Underlying Issue:** Still battling inconsistent LLM JSON output and ensuring provider settings are correctly applied *at the exact moment* memory extraction occurs within the async task. The aggressive JSON parsing is the latest attempt to mitigate LLM format variations.

## Key Learnings & Patterns

*   Provider configuration within filter plugins is highly sensitive to initialization timing and concurrent requests. Explicitly reapplying context (`_apply_llm_provider_overrides(user)`) at the start of key functions seems necessary.
*   LLMs don't always adhere strictly to JSON format prompts. Robust parsing with multiple fallback strategies (stripping, boundary search, regex) is required.
*   Clear UI status messages are crucial for diagnosing backend issues (e.g., distinguishing JSON errors from filtering).

## Important Considerations

*   Verify the provider logs (`LLM Query: Provider=...`) match the UI selection.
*   Check the final status message in the UI after sending the test prompt.

# Current Active Development Context

## Current Development Focus (2025-05-05)

Our current focus is making the memory extraction system absolutely bulletproof regardless of which LLM provider is used. We've identified that the main issues were related to JSON parsing and handling preference statements, particularly when using Claude and Gemini models.

### Key Improvements:

1. **Bulletproof JSON Extraction**:
   - Implemented a multi-stage extraction pipeline with 5 layers of fallbacks
   - Enhanced preprocessing to handle markdown, explanatory text, and malformed JSON
   - Added specialized handling for common LLM response patterns from different providers
   - Introduced comprehensive logging for better debugging

2. **Direct Preference Statement Handling**:
   - Created a "fast path" for preference statements that bypasses the LLM entirely
   - Enhanced pattern recognition for first-person preference statements
   - Special error handling for preference statement failures
   
3. **Provider Persistence**:
   - Fixed provider context to ensure selected provider persists after Docker restarts
   - Added safeguards to prevent accidental provider resets
   - Improved configuration loading to maintain user preferences

### Current Testing:

We're actively testing with multiple providers to ensure consistent behavior:
- Ollama (llama3)
- OpenRouter (Claude 3.5 Haiku)
- Gemini 2.5 Flash

Preliminary results show dramatic improvement in memory extraction reliability, especially for preference statements which are a core memory type.

### Next Steps:

1. Continue monitoring logs when using different providers
2. Run additional tests with complex preference statements
3. Document the multi-stage extraction approach for future reference
4. Release an updated version with these bulletproof improvements

## Current Debugging Focus (2025-05-05)

We're currently addressing persistent issues with LLM provider selection and memory saving failures in the OpenWebUI Adaptive Memory plugin (`adaptive_memory_v3.2.py`). 

### Key Problems and Fixes:

1. **Provider Selection Persistence**
   - **Problem**: The plugin was defaulting back to Ollama provider after Docker restarts
   - **Fix**: Enhanced the provider override system with proper initialization and checks at the beginning of memory processing

2. **Preference Statement Memory Saving**
   - **Problem**: Memory saving was failing for simple preferences due to JSON parsing errors
   - **Fix**: Implemented a multi-stage JSON extraction system with special handling for preference statements

3. **Error Message Clarity**
   - **Problem**: Misleading "Memory save skipped – filtered or duplicate" message when the actual issue was JSON parsing
   - **Fix**: Added specific error flags and improved status message reporting to clearly indicate JSON parsing issues

### Latest Testing (2025-05-05):

Testing the enhanced JSON parsing capabilities, particularly with preference statements like:
```
My absolute favorite programming language is Python because of its readability and extensive libraries.
```

The enhanced system now correctly:
1. Maintains the selected provider (e.g., OpenRouter/Claude) across Docker restarts
2. Either properly extracts JSON from responses with extra text OR creates direct memory operations for preference statements
3. Shows clear error messages when issues occur

## Implementation Strategy

We're continuing to use a focused, iterative approach to fix these issues:
1. Identify specific error patterns in logs
2. Enhance the relevant code sections
3. Test with real-world scenarios
4. Document findings and fixes thoroughly

## Current Implementation Focus (2025-05-05)

### Preference Statement Handling Improvements

We've identified that user preference statements (e.g., "My favorite programming language is Python...") are critical memory types that were failing to save properly, especially when using non-Ollama LLM providers like Claude through OpenRouter. Our current implementation focus is on creating a robust, multi-layered system to handle these statements:

1. **Direct Detection**: Fast path for common preference patterns without requiring expensive LLM calls
2. **Enhanced JSON Parsing**: Multiple fallback methods to extract valid JSON from various LLM response formats
3. **Better Error Reporting**: Clear, specific feedback when memory extraction fails

These improvements aim to significantly increase the success rate of saving important user preferences while providing better diagnostics when issues occur.

### Key Insights & Patterns

- Simple preference statements should be detected and handled directly whenever possible
- JSON parsing needs multiple strategies to handle various LLM response formats
- Error messages should be specific and helpful, distinguishing between different failure types
- The plugin should adapt to different LLM providers' response formats (Ollama vs OpenAI vs Claude)

### Next Steps

- Test the improved preference handling with various LLM providers
- Monitor successful memory save rates in real-world usage
- Consider further prompt engineering to improve JSON formatting in LLM responses 