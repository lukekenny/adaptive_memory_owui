# Adaptive Memory Plugin - Improvement & Feature Roadmap

## Overview

This document outlines the planned improvements and features for the `adaptive_memory` plugin. For completed items, see `completed_improvements.md`. All implementations **must** strictly adhere to the guidelines defined in `rules.md`.

The roadmap is organized into four key areas:
1. **Core Improvements (Items 1, 5-8)**: Enhancing stability, performance, and maintainability
2. **User Experience (Items 2-4, 9)**: Adding user-facing features and controls
3. **Integration & Portability (Item 10)**: Enabling cross-platform memory access
4. **Privacy & Monitoring (Items 11-12)**: Improving transparency and data protection

## Implementation Guidelines

### General Principles
- **Backward Compatibility**: All changes must maintain compatibility with existing memory stores
- **Error Resilience**: Graceful degradation on failures; never break core memory functionality
- **Performance First**: Optimize for minimal latency impact on chat interactions
- **Clear Documentation**: Update docstrings and user guide with each change
- **Testing Coverage**: Unit tests for new functionality, integration tests for features

### Code Style
- Follow PEP 8 guidelines
- Type hints required for all new functions
- Comprehensive docstrings (Google style)
- Max function length: 50 lines
- Max file length: 1000 lines (split if exceeded)

### Error Handling
- Use custom exception types for specific errors
- Log all errors with appropriate context
- Provide user-friendly error messages
- Implement retry mechanisms where appropriate

---

## Detailed Feature Plans

### 1. Refactor Large Methods (Improvement 6)

**Goal:** Improve code readability, maintainability, and testability by breaking down large methods.
**Complexity:** Low-Medium
**Confidence:** High
**Status:** ⏳ Pending

**Technical Details:**
*   **Target Methods for Refactoring:**
    ```python
    _process_user_memories:
      - extract_memories_from_message()
      - validate_and_filter_memories()
      - process_memory_operations()
      
    identify_memories:
      - prepare_llm_prompt()
      - extract_memory_operations()
      - validate_memory_format()
      
    get_relevant_memories:
      - compute_vector_similarities()
      - filter_by_threshold()
      - score_relevance_with_llm()
    ```

*   **Design Patterns to Apply:**
    - Strategy Pattern for filtering mechanisms
    - Factory Pattern for memory operation creation
    - Observer Pattern for status updates
    - Chain of Responsibility for memory processing pipeline

*   **Validation & Testing:**
    - Unit tests for each new helper method
    - Integration tests ensuring identical behavior
    - Performance benchmarks before/after
    - Memory usage monitoring

*   **Documentation Requirements:**
    - Updated function docstrings
    - Flow diagrams for complex operations
    - Example usage in comments
    - Performance considerations noted

**Subtasks:**
*   **Identify Large Methods:**
    - Target `_process_user_memories`, `identify_memories`, `get_relevant_memories`, others > ~100 lines
    - Profile methods to identify bottlenecks
    - Map data flow and dependencies
    - Document current behavior thoroughly

*   **Plan Refactoring:**
    - Create detailed method signatures
    - Define input/output contracts
    - Identify shared utilities
    - Plan error handling strategy

*   **Implement Refactoring:**
    - Create helper methods
    - Move code blocks
    - Update variable passing
    - Maintain type hints
    - Add comprehensive logging

*   **Documentation Updates:**
    - Update all docstrings
    - Add explanatory comments
    - Create debugging guide
    - Document performance implications

*   **Testing:**
    - Unit test coverage > 90%
    - Integration tests for workflows
    - Performance regression testing
    - Memory leak checking

**Expected Successful Output:**
- Methods under 50 lines
- Improved test coverage
- Clearer error handling
- Better debugging capabilities
- No performance regression
- Comprehensive documentation

### 2. Dynamic Memory Tagging (Feature 2)

**Goal:** Allow LLM to generate relevant keyword tags for memories during extraction.
**Complexity:** Medium
**Confidence:** Medium
**Status:** ✅ Implemented - Awaiting Verification

**Technical Details:**
*   **Tag Generation Prompt:**
    ```
    System: You are a memory tagging assistant. Analyze the given memory and generate 2-3 relevant tags.
    Tags should be:
    - Specific but reusable
    - Lower case, no spaces
    - Related to content type
    - Avoid duplicating existing categories
    
    Example Memory: "User prefers dark mode in all applications"
    Good Tags: ["ui_preference", "theme", "accessibility"]
    Bad Tags: ["dark_mode", "apps", "preference"] (too specific/generic)
    ```

*   **Tag Storage Format:**
    ```python
    # Implemented as part of MemoryOperation model:
    dynamic_tags: Optional[List[Dict[str, Any]]] = None  # [{"tag": str, "confidence": float}, ...]
    ```

*   **Tag Validation Rules:**
    - Implemented in `_validate_dynamic_tags`
    - Min confidence: 0.65 (configurable via `dynamic_tag_min_confidence_threshold` valve)
    - Allowed chars: a-z, 0-9, _ (enforced via regex validation)
    - Max tags: 5 total (configurable)

**Implementation Status:**
- ✅ Added `dynamic_tags` field to `MemoryOperation` Pydantic model
- ✅ Created new valves for controlling dynamic tagging (enabled/disabled, min confidence, etc.)
- ✅ Implemented `_generate_and_validate_dynamic_tags` function for LLM-based tag generation
- ✅ Modified `_execute_memory_operation` to include tag generation for new memories
- ✅ Updated memory formatting functions to display dynamic tags in memory content
- ✅ Enhanced memory retrieval to include dynamic tag information
- ⚠️ Awaiting verification after fixing memory injection issues to confirm end-to-end functionality

**Note:** While the implementation is complete, verification requires resolving the embedding dimension mismatch issue currently preventing memory injection.
### 3. Personalized Response Tailoring (Prompt Injection) (Feature 7)

**Goal:** Leverage stored preferences to instruct LLM to tailor response style/content.
**Complexity:** Medium
**Confidence:** Medium
**Status:** ⏳ Pending

**Subtasks:**
*   **Identify Preference Memories:** Determine relevant tags/types.
*   **Modify Injection Logic:** Modify `_inject_memories_into_context` to add explicit instructions to system prompt based on detected preferences (e.g., "User prefers concise answers."). Filter injected memories for tailoring focus.
*   **Performance Testing:** Ensure the additional prompt instructions don't significantly impact token usage or response time.
*   **Documentation Updates:** Add explanation of the feature to the docstring and how it personalizes responses.
*   *(Adhere to `rules.md`: Primarily prompt engineering).*
*   **Test:** Store preferences; observe if subsequent LLM responses align better with stored preferences; measure response quality improvements.

**Expected Successful Output:** LLM responses more aligned with user preferences without performance impact. Implementation follows `rules.md`.

### 4. Verify Cross-Session Memory Persistence (Bug Fix/Verification)

**Goal:** Ensure memories saved for a user persist and are retrievable across different chat sessions initiated by the same user.
**Complexity:** Medium
**Confidence:** High (to verify), Medium (to fix if needed)
**Status:** ⏳ Pending Verification

**Subtasks:**
*   **Verify User ID Usage:** Add debug logging to trace `user_id` extraction from `__user__` object in `inlet`/`outlet` and confirm consistent use in calls to `add_memory`, `query_memory`, `delete_memory_by_id`.
*   **Test Cross-Session Retrieval:** Create memory in session A, start new session B with same user, attempt prompt that should retrieve memory from session A. Verify success/failure via logs/context.
*   **Investigate OWUI Behavior:** If user ID usage is correct but persistence fails, research OpenWebUI documentation/issues regarding plugin data scope and lifecycle across sessions.
*   **Implement Fix (If Necessary):** If a bug is found in ID handling, correct the logic. If it's an OWUI limitation, document it clearly.
*   **Add Graceful Handling for Missing ID:** Add checks at the start of `inlet`/`outlet` to verify `user` and `user['id']` exist. If not, log a clear error and skip memory operations for that request gracefully.
*   *(Adhere to `rules.md`)*

**Expected Successful Output:** Clear confirmation whether cross-session memory works as intended; bugs in ID handling fixed; plugin handles missing user data gracefully; or limitations documented.

### 5. Improve Configuration Handling & Validation (Bug Fix/Refinement)

**Goal:** Ensure all configuration 'Valves' (especially thresholds) are loaded correctly, applied consistently, and have sensible defaults.
**Complexity:** Medium
**Confidence:** High
**Status:** ⏳ Pending

**Subtasks:**
*   **Add Debug Logging:** Log the *actual* values of threshold valves (`vector_similarity_threshold`, `relevance_threshold`, `llm_skip_relevance_threshold`, `min_confidence_threshold`, `embedding_similarity_threshold`, `min_memory_length`) being used within critical functions (`get_relevant_memories`, `process_memories`, etc.).
*   **Verify Provider/Model Usage:** Log the specific provider, model, and URL being used in `query_llm_with_retry` and `_get_embedding` based on loaded Valves.
*   **Review Default Thresholds:** Re-evaluate default values based on user feedback (e.g., `vector_similarity_threshold=0.60`, `relevance_threshold=0.60`) and potentially lower them further or make them model-dependent if feasible.
*   **Targeted Testing:** Define test cases to explicitly verify valve application (e.g., set threshold=0.55, test prompt with similarity 0.5, 0.55, 0.6).
*   *(Adhere to `rules.md`)*

**Expected Successful Output:** Increased confidence that valves are working as configured; clearer debugging information; potentially better out-of-the-box relevance via improved defaults.

### 6. Enhance Memory Retrieval Tuning (Refinement)

**Goal:** Improve the semantic relevance of retrieved memories beyond exact matches, making the system feel less strict.
**Complexity:** Medium
**Confidence:** Medium
**Status:** ⏳ Pending

**Subtasks:**
*   **Implement Threshold Tuning (See Improvement 9):** Adjusting default relevance/similarity thresholds is the primary lever.
*   **Evaluate Default Embedding Model:** Test alternative local embedding models (beyond `all-MiniLM-L6-v2`) for potentially better semantic nuance capture vs. performance trade-off. Consider changing the default `embedding_model_name`.
*   **Consider Contextual Boosts (Connects to Feature 7/Personalization):** Explore ways to slightly boost relevance scores based on broader conversation context or known user preferences, even if direct embedding similarity is lower.
*   **Generate Contextual Queries (Optional):** Explore using a quick LLM call or keyword analysis to generate a more focused search query from the user's last message(s) instead of using the raw message embedding directly.
*   *(Adhere to `rules.md`)*
*   **Test:** Evaluate retrieval relevance with various semantic queries after tuning thresholds and potentially changing the default model.

**Expected Successful Output:** Memory retrieval feels more intuitive and less reliant on exact keyword matches; relevant concepts are surfaced more reliably.

### 7. Improve Processing Status & Error Feedback (UX/Robustness)

**Goal:** Provide clearer, more specific feedback to the user about memory operations, especially failures, and prevent indefinite hangs.
**Complexity:** Medium
**Confidence:** High
**Status:** ⏳ Pending

**Subtasks:**
*   **Implement Timeouts:** Add appropriate timeouts to external network calls (`query_llm_with_retry`, `_get_embedding_from_api`).
*   **Enhance JSON Error Logging:** Log the raw, non-parsable LLM response when a `json.JSONDecodeError` occurs in `_extract_and_parse_json`.
*   **Refine UI Status Messages:** Modify `_safe_emit` calls to provide specific reasons for skipped saves (filtered, duplicate, low confidence) or errors (LLM connection, invalid response, etc.). Utilize `_error_message` more effectively.
*   **Verify Monitoring:** Ensure Prometheus metrics/endpoints cover key operations and error types accurately.
*   *(Adhere to `rules.md`)*
*   **Test:** Simulate LLM errors, timeouts, invalid responses, filtering scenarios; verify specific and helpful status messages are shown to the user.

**Expected Successful Output:** Users receive clearer feedback on memory operations; plugin is more resilient to hangs and parsing errors; debugging is easier.

### 8. Expand Documentation & Usage Clarity (Docs)

**Goal:** Make it easier for users to understand and utilize the plugin's features and configurations.
**Complexity:** Low
**Confidence:** High
**Status:** ⏳ Pending

**Subtasks:**
*   **Expand Docstring User Guide:** Add sections explaining core concepts (extraction, relevance, banks), detail *all* important Valves, clarify bank usage/selection, add troubleshooting tips for common issues (relevance, JSON errors).
*   **Maintain Roadmap Docs:** Keep `roadmap.md` and `completed_improvements.md` updated.
*   *(Adhere to `rules.md`)*
*   **Test:** Review the updated User Guide for clarity and completeness.

**Expected Successful Output:** Users can more easily understand plugin features, configure valves, and use commands through improved in-plugin documentation.

### 9. Always-Sync to RememberAPI (Feature Add-on)

**Goal:** Automatically synchronize newly saved local memories to an external RememberAPI account in real-time, allowing memory portability across different services (ChatGPT, Claude, etc.) without manual export/import.
**Complexity:** Medium
**Confidence:** High
**Status:** ⏳ Pending

**Technical Details:**
*   **Configuration Schema:**
    ```python
    class RememberAPISyncConfig(BaseModel):
        enable_rememberapi_sync: bool = Field(
            default=False,
            description="Master toggle for RememberAPI synchronization"
        )
        rememberapi_endpoint: str = Field(
            default="https://api.rememberapi.com/v1/memories/add",
            description="RememberAPI endpoint URL"
        )
        rememberapi_api_key: SecretStr = Field(
            description="RememberAPI authentication key"
        )
        rememberapi_user_id_field: Literal[
            'id', 'email', 'name', 'username'
        ] = Field(
            default='id',
            description="User identifier field to use"
        )
        sync_retry_attempts: int = Field(
            default=3,
            description="Number of retry attempts for failed syncs"
        )
        sync_retry_delay: float = Field(
            default=1.0,
            description="Delay between retry attempts (seconds)"
        )
    ```

*   **Memory Payload Format:**
    ```json
    {
      "userId": "string",  // From configured user_id_field
      "content": "string", // Memory content
      "metadata": {
        "source": "adaptive_memory_v3",
        "tags": ["array", "of", "tags"],
        "memory_bank": "string",
        "confidence": 0.95,
        "created_at": "2024-01-20T12:00:00Z",
        "openwebui_memory_id": "uuid"
      }
    }
    ```

*   **Sync Process Flow:**
    ```mermaid
    sequenceDiagram
        participant OM as OpenWebUI Memory
        participant AM as Adaptive Memory
        participant RA as RememberAPI
        
        OM->>AM: add_memory()
        AM->>AM: _execute_memory_operation()
        AM->>OM: Save locally
        AM->>AM: create_task(_save_to_rememberapi)
        AM->>RA: POST /memories/add
        RA-->>AM: 201 Created
        AM->>AM: Log success
    ```

*   **Error Handling:**
    - Retry on 5xx errors
    - Log 4xx errors
    - Circuit breaker pattern
    - Exponential backoff

*   **Monitoring:**
    ```python
    REMEMBERAPI_METRICS = {
        'sync_attempts_total': Counter,
        'sync_success_total': Counter,
        'sync_failures_total': Counter,
        'sync_latency_seconds': Histogram,
        'sync_retry_total': Counter,
        'sync_circuit_breaks_total': Counter
    }
    ```

**Subtasks:**
*   **Add Valves:**
    - Define configuration schema
    - Add validation rules
    - Document all options
    - Add security notes

*   **Implement Dual-Write Logic:**
    - Modify `_execute_memory_operation`
    - Add sync triggering
    - Handle race conditions
    - Manage async tasks

*   **Implement Helper Function:**
    - Create `_save_to_rememberapi`
    - Add retry logic
    - Implement logging
    - Add metrics
    - Handle errors

*   **Add Monitoring:**
    - Define metrics
    - Add counters
    - Create dashboards
    - Set up alerts

*   **Documentation:**
    - Update User Guide
    - Add configuration guide
    - Create troubleshooting guide
    - Document metrics

*   **Testing:**
    - Unit tests
    - Integration tests
    - Load tests
    - Failure scenarios
    - Security testing

**Expected Successful Output:**
- Automatic, reliable syncing
- Clear error handling
- Comprehensive monitoring
- Detailed documentation
- Robust security

### 10. Enhance Status Emitter Transparency (UX/Debugging)

**Goal:** Improve the clarity, accuracy, and coverage of status messages emitted to the UI.
**Complexity:** Low-Medium
**Confidence:** High
**Status:** ⏳ Pending

**Technical Details:**
*   **Status Message Schema:**
    ```python
    class StatusMessage(BaseModel):
        type: Literal['info', 'success', 'warning', 'error']
        operation: str  # e.g., 'save', 'retrieve', 'sync'
        stage: str  # e.g., 'start', 'progress', 'complete'
        description: str
        details: Optional[Dict[str, Any]]
        emoji: str
        timestamp: datetime
    ```

*   **Standard Emojis:**
    ```python
    STATUS_EMOJIS = {
        'extract': '📝',
        'retrieve': '🧠',
        'save': '✅',
        'warning': '⚠️',
        'info': 'ℹ️',
        'sync': '🔗',
        'error': '❌',
        'filter': '🔍',
        'prune': '✂️',
        'summarize': '📚'
    }
    ```

### 11. Optional PII Stripping on Save (Privacy Enhancement)

**Goal:** Add an option to automatically detect and redact/anonymize common PII patterns before saving memories locally.
**Complexity:** Medium
**Confidence:** Medium (depending on desired robustness)
**Status:** ⏳ Pending

**Subtasks:**
*   **Add Valve:** Create `strip_pii_on_save` (bool, default: `False`) valve.
*   **Define PII Patterns:** Identify common regex patterns for emails, phone numbers, typical ID formats, potentially names (harder).
*   **Implement Stripping Logic:** Modify `_execute_memory_operation` (or a helper called before `add_memory`) to apply regex replacements if the valve is enabled. **Crucially, ensure stripping happens *before* the potential call to `_save_to_rememberapi`** so stripped data is sent externally too.
*   **Consider Advanced Options (Optional):** Evaluate libraries like `presidio-analyzer` or a dedicated LLM call for more robust PII detection, noting dependency/performance implications.
*   **Error Handling:** Ensure stripping failures don't prevent memory saving (log warnings).
*   **Document:** Explain the valve, its limitations (regex-based might not be perfect), and privacy implications.
*   *(Adhere to `rules.md`)*
*   **Test:** Save memories containing various PII patterns with the valve enabled/disabled; verify redaction works as expected both locally and in the data sent to the mock RememberAPI endpoint.

**Expected Successful Output:** Users can optionally enable basic PII stripping for enhanced privacy in stored memories, and this stripping applies to data synced externally via RememberAPI as well.

## Timeline & Priorities

### Phase 1: Core Stability (1-2 months)
- Items 1, 4, 5: Refactoring, persistence verification, config handling
- Success Metrics: Code coverage > 90%, No critical bugs, Clear documentation

### Phase 2: User Experience (2-3 months)
- Items 2-3, 6-8: Memory tagging, personalization, retrieval tuning, etc.
- Success Metrics: User satisfaction surveys, Improved memory retrieval metrics

### Phase 3: Integration (1-2 months)
- Item 9: RememberAPI sync
- Success Metrics: Sync success rate > 99%, < 100ms latency

### Phase 4: Privacy & Monitoring (1-2 months)
- Items 10-11: Status emitter, PII handling
- Success Metrics: All operations tracked, No PII leaks

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

### Getting Started
1. Fork the repository
2. Set up development environment
3. Pick an issue from the roadmap
4. Create a feature branch
5. Submit a PR

### Development Setup
```bash
# Clone repository
git clone https://github.com/your-repo/adaptive-memory.git

# Install dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 adaptive_memory
mypy adaptive_memory
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
