# Product Context

*   **Problem Solved:** Standard LLM interactions lack persistent memory, forcing users to repeat information and preventing the model from developing context over time. Conversations feel impersonal and inefficient.
*   **Target User:** Users of OpenWebUI seeking more natural, personalized, and context-aware conversations with their LLMs, where the model remembers past interactions and user details.
*   **Desired Experience:** Users should feel the LLM understands their history, preferences, and goals without explicit reminders. Interactions should become more relevant and tailored over time, leading to more efficient and engaging conversations.
*   **Key Scenarios:**
    *   User shares a personal preference (e.g., "I prefer concise answers"). The LLM remembers and adapts its style (Preference memory).
    *   User mentions a long-term goal (e.g., "I'm learning Python for data science"). The LLM recalls this when relevant topics arise (Goal memory).
    *   User discusses a project detail or a personal relationship ("My sister Sarah is a doctor"). The LLM retains this context (Behavior/Relationship memory).
    *   The plugin filters out a user asking for trivia (e.g., "What's the capital of France?") from being stored as a personal memory, potentially storing "User is interested in geography" instead (Behavior/Interest memory).
    *   User asks about a topic ("Tell me about quantum computing"), an implicit interest memory is created.
    *   Older, related memories are periodically summarized into a more concise form. 