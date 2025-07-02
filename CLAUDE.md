# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Adaptive Memory is a sophisticated OpenWebUI Filter Function plugin that provides persistent, personalized memory capabilities for Large Language Models (LLMs). It enables LLMs to remember key information about users across separate conversations through dynamic extraction, filtering, storage, and retrieval of user-specific information.

## Development Commands

### Code Quality & Validation

Since this is a single-file OpenWebUI plugin:

1. **Code Quality Checks**:
   ```bash
   # Primary linting tool:
   ruff check adaptive_memory_v3.2.py
   
   # Syntax validation:
   python -m py_compile adaptive_memory_v3.2.py
   
   # Type checking (if mypy available):
   mypy adaptive_memory_v3.2.py --ignore-missing-imports
   ```

2. **Manual Testing**: Deploy the plugin to OpenWebUI and test via the interface

3. **Debugging**: Enable debug logging in OpenWebUI and monitor logs:
   ```bash
   # View OpenWebUI logs for plugin debug output
   docker logs openwebui_container_name -f
   
   # Alternative log locations:
   tail -f /var/log/openwebui/app.log
   ```

## Architecture & Key Components

### Plugin Architecture (OpenWebUI Filter Function)
The plugin follows OpenWebUI's Filter Function architecture with three main hooks:

1. **`inlet()`**: Pre-processes user messages before sending to LLM
   - Extracts memories from user messages
   - Filters and validates new memories
   - Stores valid memories in OpenWebUI's memory system

2. **`outlet()`**: Post-processes LLM responses
   - Injects relevant memories into conversation context
   - Handles memory management operations
   - Provides user feedback via status emitters

3. **`stream()`**: Handles streaming responses (currently unused)

### Core Memory Flow

1. **Memory Extraction** (`identify_memories()`)
   - Uses LLM to analyze messages for extractable information
   - Categorizes into memory types (identity, preference, goal, etc.)
   - Assigns confidence scores and memory banks

2. **Memory Filtering** (Multi-layered approach)
   - Confidence threshold filtering
   - Length validation
   - Trivia/blacklist filtering
   - Deduplication (semantic or text-based)

3. **Memory Storage**
   - Uses OpenWebUI's built-in memory API
   - Stores with embeddings for semantic search
   - Organized by memory banks (General, Personal, Work)

4. **Memory Retrieval** (`get_relevant_memories()`)
   - Vector similarity search using embeddings
   - Optional LLM relevance scoring
   - Configurable similarity thresholds

### Key Configuration (Valves)

Critical valves for tuning behavior:
- `vector_similarity_threshold` (default: 0.60) - Lower for more permissive retrieval
- `relevance_threshold` (default: 0.60) - Final cutoff after LLM scoring
- `min_confidence_threshold` (default: 0.5) - Filters uncertain extractions
- `embedding_model_name` (default: "all-MiniLM-L6-v2") - Local embedding model

## Development Guidelines

### Version Management
- Current active version: `adaptive_memory_v3.2.py`
- Previous versions stored in `Other Versions/` directory
- Create new version file after completing each major feature

### Implementation Rules (from rules.md)
1. **One feature per session** - Implement one feature/improvement at a time within the currently active version file
2. **Preserve functionality** - Never delete/truncate existing code or disrupt functionality
3. **Test thoroughly** - Verify all changes work correctly, including linting and compilation checks
4. **Mandatory automatic documentation** - IMMEDIATELY update ALL relevant documentation after ANY code change:
   - Issue log in `identified_issues_and_how_we_fixed_them.md`
   - Current status in `memory-bank/activeContext.md`
   - Progress record in `memory-bank/progress.md`
   - Status updates in roadmap files
5. **Error handling workflow** - Check `identified_issues_and_how_we_fixed_them.md` first for recurring errors; use sequential thinking for persistent issues

### Error Handling Patterns
- Always wrap external API calls in try-except blocks
- Use `_safe_emit()` for user status messages
- Log errors with full context using the JSON formatter
- Implement retry logic for transient failures

### Memory Operation Best Practices
1. **Deep copy dictionaries** when modifying in `outlet()` to avoid iteration errors
2. **Validate embeddings** dimension consistency before storage
3. **Handle missing user context** gracefully
4. **Rate limit LLM calls** to prevent overwhelming external services

## Common Issues & Solutions

### 1. Dictionary Iteration Errors
**Issue**: `RuntimeError: dictionary changed size during iteration`
**Solution**: Always deep copy `body` at start of `outlet()` function

### 2. Embedding Dimension Mismatch
**Issue**: Stored embeddings incompatible with new model
**Solution**: Clear old memories or implement migration logic

### 3. Memory Not Persisting
**Issue**: Memories not available across sessions
**Solution**: Verify user ID extraction and consistent usage in all memory operations

### 4. Low Relevance Scores
**Issue**: Relevant memories not being retrieved
**Solution**: Lower `vector_similarity_threshold` and `relevance_threshold` (try 0.45-0.55)

## Project Structure

```
OWUI_adaptive_memory/
├── adaptive_memory_v3.2.py      # Current active version
├── Other Versions/              # Previous iterations
├── memory-bank/                 # Project context and status
│   ├── activeContext.md        # Current development state
│   ├── progress.md             # Completed work log
│   └── projectbrief.md         # Core requirements
├── OWUI tech-docs/             # OpenWebUI documentation
├── roadmap.md                  # Future improvements plan
├── rules.md                    # Development guidelines
└── identified_issues_and_how_we_fixed_them.md  # Issue tracker
```

## Next Steps & Roadmap

Current priorities (see roadmap.md for detailed implementation plans):
1. **Refactor large methods** - Break down methods > 50 lines for better maintainability
2. **Verify dynamic memory tagging** - Confirm end-to-end functionality of LLM-generated tags
3. **Cross-session persistence verification** - Ensure memories persist across user sessions
4. **Configuration handling** - Validate all threshold valves are applied correctly
5. **Enhance retrieval tuning** - Improve semantic relevance beyond exact matches
6. **RememberAPI sync** - Enable cross-platform memory synchronization

## External Dependencies

The plugin is designed to be self-contained within OpenWebUI's environment:
- `sentence-transformers` - For local embeddings
- `numpy` - Vector operations
- `aiohttp` - Async HTTP client
- `prometheus_client` - Metrics (optional)

No additional dependencies should be added without careful consideration.