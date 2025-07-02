# Implementation Rules

These rules must be followed when implementing any improvements or features for the Adaptive Memory plugin:

0.  **Documentation Brevity & Clarity:** Keep all documentation updates (`rules.md`, Memory Bank files, `improvement_plan.md`, `identified_issues_and_how_we_fixed_them.md`) **concise and clear**. Focus on conveying essential information effectively to prevent file bloat.
1.  **One Feature Per Session/File:** Implement **one** feature/improvement from `improvement_plan.md` at a time, within the **currently active version file**.
2.  **Preserve Existing Functionality:** Do not delete/truncate existing code or disrupt functionality.
3.  **Post-Implementation Checks:** After implementing:
    *   **Review:** Check code against requirements (`improvement_plan.md`).
    *   **Linting:** Check and fix linter errors.
4.  **Testing:** Thoroughly test the implementation.
5.  **Error Handling & Debugging:**
    *   **Check History:** When encountering an error, **first check `identified_issues_and_how_we_fixed_them.md`**. 
        *   If it's a **recurring error** documented previously, **review *all* prior fix attempts** listed for that specific error to avoid proposing redundant solutions.
    *   **Sequential Thinking:** If an error is **not fixed on the first attempt** (or persists after trying documented solutions), **MUST use the sequential thinking tool (`mcp_sequentialthinking_sequentialthinking`)** to analyze the problem. Continue using this tool for subsequent attempts until the issue is resolved.
    *   **Issue Tracking:** Document significant obstacles, errors, or failed tests in `identified_issues_and_how_we_fixed_them.md`. **Focus on documenting unique errors or distinct solutions/attempts for recurring issues**, rather than creating duplicate entries for the same persistent problem. The goal is a useful troubleshooting guide.
6.  **File Versioning & New Session:** **After** successful implementation, checks, testing, and issue documentation for a single item:
    *   **User creates** a **new versioned file** (e.g., `v2.7` -> `v2.8`).
    *   Start a **new chat session** for the next item, working in the new file.
7.  **Architectural Compliance:** Adhere strictly to OpenWebUI Filter Function architecture (`OWUI tech-docs`).
8.  **Planning & Progress Tracking:** Detail subtasks in `improvement_plan.md`. Update its status (⏳, ✅, 🚧) after completing step 6.
9.  **No External Dependencies:** Script must remain self-contained.
10. **Mandatory Automatic Documentation Updates:** After ANY code change or fix, IMMEDIATELY update ALL relevant documentation without waiting for a reminder:
    *   **Issue Log:** Add the issue and fix to `identified_issues_and_how_we_fixed_them.md`
    *   **Memory Bank:** Update the current status in `memory-bank/activeContext.md`
    *   **Progress:** Record completed work in `memory-bank/progress.md`
    *   **Status:** Update the status in `improvement_plan.md` when a feature is complete
    This update must be done automatically as part of the code change workflow, not as a separate step requiring a reminder. 