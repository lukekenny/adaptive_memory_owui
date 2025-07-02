# Technical Context

*   **Technology Stack:** Python (likely 3.11+), Pydantic, Aiohttp, Sentence Transformers (`all-MiniLM-L6-v2`), NumPy, Pytz, asyncio.
*   **Development Environment:** Standard Python development environment. Requires installation of dependencies listed in `requirements.txt` (if provided) or inferred from imports (`sentence-transformers`, `numpy`, `aiohttp`, `pydantic`, `pytz`). Relies on OpenWebUI environment for execution.
*   **Build/Deployment:** Single Python script (e.g., `adaptive_memory_v2.6.py`). **Deployment is done by copying the entire script content and pasting it into the OpenWebUI Function editor UI, then saving through the interface.** This replaces any previous version saved for that function. No file system manipulation or container restarts are typically needed *just* for code updates.
*   **Dependencies:**
    *   **Python Libraries:** `sentence-transformers`, `numpy`, `aiohttp`, `fastapi` (type hints/models), `pydantic`, `pytz`.
    *   **External Services:** Configured LLM provider API (OpenRouter or Ollama).
    *   **Internal (OpenWebUI):**
        *   Plugin execution environment (runs `inlet`/`outlet`).
        *   Memory storage API (`open_webui.routers.memories`: `add_memory`, `delete_memory_by_id`, `Memories.get_memories_by_user_id`).
        *   User context/model (`open_webui.models.users.Users`).
        *   Event emitter (`__event_emitter__`) for status updates.
        *   Configuration system (Valves).
*   **Constraints:**
    *   Must operate as a self-contained OpenWebUI Filter Function.
    *   No external databases or servers (memory stored via OpenWebUI API).
    *   Must adhere to OpenWebUI plugin interface (`inlet`, `outlet`, `cleanup`).
    *   Resource constraints of the hosting OpenWebUI instance (CPU, memory for embeddings/LLM calls).
    *   Relies on specific OpenWebUI internal API structures.
*   **Tooling:** Standard Python linting/formatting tools (assumed). Logging configured via Python `logging` module with custom JSON formatter. 