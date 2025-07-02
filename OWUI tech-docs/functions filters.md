[Skip to main content](https://docs.openwebui.com/features/plugin/functions/filter/#__docusaurus_skipToContent_fallback)

On this page

Welcome to the comprehensive guide on Filter Functions in Open WebUI! Filters are a flexible and powerful **plugin system** for modifying data _before it's sent to the Large Language Model (LLM)_ (input) or _after it’s returned from the LLM_ (output). Whether you’re transforming inputs for better context or cleaning up outputs for improved readability, **Filter Functions** let you do it all.

This guide will break down **what Filters are**, how they work, their structure, and everything you need to know to build powerful and user-friendly filters of your own. Let’s dig in, and don’t worry—I’ll use metaphors, examples, and tips to make everything crystal clear! 🌟

* * *

## 🌊 What Are Filters in Open WebUI? [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-what-are-filters-in-open-webui "Direct link to 🌊 What Are Filters in Open WebUI?")

Imagine Open WebUI as a **stream of water** flowing through pipes:

- **User inputs** and **LLM outputs** are the water.
- **Filters** are the **water treatment stages** that clean, modify, and adapt the water before it reaches the final destination.

Filters sit in the middle of the flow—like checkpoints—where you decide what needs to be adjusted.

Here’s a quick summary of what Filters do:

1. **Modify User Inputs (Inlet Function)**: Tweak the input data before it reaches the AI model. This is where you enhance clarity, add context, sanitize text, or reformat messages to match specific requirements.
2. **Intercept Model Outputs (Stream Function)**: Capture and adjust the AI’s responses **as they’re generated** by the model. This is useful for real-time modifications, like filtering out sensitive information or formatting the output for better readability.
3. **Modify Model Outputs (Outlet Function)**: Adjust the AI's response **after it’s processed**, before showing it to the user. This can help refine, log, or adapt the data for a cleaner user experience.

> **Key Concept:** Filters are not standalone models but tools that enhance or transform the data traveling _to_ and _from_ models.

Filters are like **translators or editors** in the AI workflow: you can intercept and change the conversation without interrupting the flow.

* * *

## 🗺️ Structure of a Filter Function: The Skeleton [​](https://docs.openwebui.com/features/plugin/functions/filter/\#%EF%B8%8F-structure-of-a-filter-function-the-skeleton "Direct link to 🗺️ Structure of a Filter Function: The Skeleton")

Let's start with the simplest representation of a Filter Function. Don't worry if some parts feel technical at first—we’ll break it all down step by step!

### 🦴 Basic Skeleton of a Filter [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-basic-skeleton-of-a-filter "Direct link to 🦴 Basic Skeleton of a Filter")

```codeBlockLines_e6Vv
from pydantic import BaseModel
from typing import Optional

class Filter:
    # Valves: Configuration options for the filter
    class Valves(BaseModel):
        pass

    def __init__(self):
        # Initialize valves (optional configuration for the Filter)
        self.valves = self.Valves()

    def inlet(self, body: dict) -> dict:
        # This is where you manipulate user inputs.
        print(f"inlet called: {body}")
        return body

    def stream(self, event: dict) -> dict:
        # This is where you modify streamed chunks of model output.
        print(f"stream event: {event}")
        return event

    def outlet(self, body: dict) -> None:
        # This is where you manipulate model outputs.
        print(f"outlet called: {body}")

```

* * *

### 🎯 Key Components Explained [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-key-components-explained "Direct link to 🎯 Key Components Explained")

#### 1️⃣ **`Valves` Class (Optional Settings)** [​](https://docs.openwebui.com/features/plugin/functions/filter/\#1%EF%B8%8F%E2%83%A3-valves-class-optional-settings "Direct link to 1️⃣-valves-class-optional-settings")

Think of **Valves** as the knobs and sliders for your filter. If you want to give users configurable options to adjust your Filter’s behavior, you define those here.

```codeBlockLines_e6Vv
class Valves(BaseModel):
    OPTION_NAME: str = "Default Value"

```

For example:

If you're creating a filter that converts responses into uppercase, you might allow users to configure whether every output gets totally capitalized via a valve like `TRANSFORM_UPPERCASE: bool = True/False`.

* * *

#### 2️⃣ **`inlet` Function (Input Pre-Processing)** [​](https://docs.openwebui.com/features/plugin/functions/filter/\#2%EF%B8%8F%E2%83%A3-inlet-function-input-pre-processing "Direct link to 2️⃣-inlet-function-input-pre-processing")

The `inlet` function is like **prepping food before cooking**. Imagine you’re a chef: before the ingredients go into the recipe (the LLM in this case), you might wash vegetables, chop onions, or season the meat. Without this step, your final dish could lack flavor, have unwashed produce, or simply be inconsistent.

In the world of Open WebUI, the `inlet` function does this important prep work on the **user input** before it’s sent to the model. It ensures the input is as clean, contextual, and helpful as possible for the AI to handle.

📥 **Input**:

- **`body`**: The raw input from Open WebUI to the model. It is in the format of a chat-completion request (usually a dictionary that includes fields like the conversation's messages, model settings, and other metadata). Think of this as your recipe ingredients.

🚀 **Your Task**:

Modify and return the `body`. The modified version of the `body` is what the LLM works with, so this is your chance to bring clarity, structure, and context to the input.

##### 🍳 Why Would You Use the `inlet`? [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-why-would-you-use-the-inlet "Direct link to -why-would-you-use-the-inlet")

1. **Adding Context**: Automatically append crucial information to the user’s input, especially if their text is vague or incomplete. For example, you might add "You are a friendly assistant" or "Help this user troubleshoot a software bug."

2. **Formatting Data**: If the input requires a specific format, like JSON or Markdown, you can transform it before sending it to the model.

3. **Sanitizing Input**: Remove unwanted characters, strip potentially harmful or confusing symbols (like excessive whitespace or emojis), or replace sensitive information.

4. **Streamlining User Input**: If your model’s output improves with additional guidance, you can use the `inlet` to inject clarifying instructions automatically!


##### 💡 Example Use Cases: Build on Food Prep [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-example-use-cases-build-on-food-prep "Direct link to 💡 Example Use Cases: Build on Food Prep")

###### 🥗 Example 1: Adding System Context [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-example-1-adding-system-context "Direct link to 🥗 Example 1: Adding System Context")

Let’s say the LLM is a chef preparing a dish for Italian cuisine, but the user hasn’t mentioned "This is for Italian cooking." You can ensure the message is clear by appending this context before sending the data to the model.

```codeBlockLines_e6Vv
def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
    # Add system message for Italian context in the conversation
    context_message = {
        "role": "system",
        "content": "You are helping the user prepare an Italian meal."
    }
    # Insert the context at the beginning of the chat history
    body.setdefault("messages", []).insert(0, context_message)
    return body

```

📖 **What Happens?**

- Any user input like "What are some good dinner ideas?" now carries the Italian theme because we’ve set the system context! Cheesecake might not show up as an answer, but pasta sure will.

###### 🔪 Example 2: Cleaning Input (Remove Odd Characters) [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-example-2-cleaning-input-remove-odd-characters "Direct link to 🔪 Example 2: Cleaning Input (Remove Odd Characters)")

Suppose the input from the user looks messy or includes unwanted symbols like `!!!`, making the conversation inefficient or harder for the model to parse. You can clean it up while preserving the core content.

```codeBlockLines_e6Vv
def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
    # Clean the last user input (from the end of the 'messages' list)
    last_message = body["messages"][-1]["content"]
    body["messages"][-1]["content"] = last_message.replace("!!!", "").strip()
    return body

```

📖 **What Happens?**

- Before: `"How can I debug this issue!!!"` ➡️ Sent to the model as `"How can I debug this issue"`

Note: The user feels the same, but the model processes a cleaner and easier-to-understand query.

##### 📊 How `inlet` Helps Optimize Input for the LLM: [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-how-inlet-helps-optimize-input-for-the-llm "Direct link to -how-inlet-helps-optimize-input-for-the-llm")

- Improves **accuracy** by clarifying ambiguous queries.
- Makes the AI **more efficient** by removing unnecessary noise like emojis, HTML tags, or extra punctuation.
- Ensures **consistency** by formatting user input to match the model’s expected patterns or schemas (like, say, JSON for a specific use case).

💭 **Think of `inlet` as the sous-chef in your kitchen**—ensuring everything that goes into the model (your AI "recipe") has been prepped, cleaned, and seasoned to perfection. The better the input, the better the output!

* * *

#### 🆕 3️⃣ **`stream` Hook (New in Open WebUI 0.5.17)** [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-3%EF%B8%8F%E2%83%A3-stream-hook-new-in-open-webui-0517 "Direct link to -3️⃣-stream-hook-new-in-open-webui-0517")

##### 🔄 What is the `stream` Hook? [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-what-is-the-stream-hook "Direct link to -what-is-the-stream-hook")

The **`stream` function** is a new feature introduced in Open WebUI **0.5.17** that allows you to **intercept and modify streamed model responses** in real time.

Unlike `outlet`, which processes an entire completed response, `stream` operates on **individual chunks** as they are received from the model.

##### 🛠️ When to Use the Stream Hook? [​](https://docs.openwebui.com/features/plugin/functions/filter/\#%EF%B8%8F-when-to-use-the-stream-hook "Direct link to 🛠️ When to Use the Stream Hook?")

- Modify **streaming responses** before they are displayed to users.
- Implement **real-time censorship or cleanup**.
- **Monitor streamed data** for logging/debugging.

##### 📜 Example: Logging Streaming Chunks [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-example-logging-streaming-chunks "Direct link to 📜 Example: Logging Streaming Chunks")

Here’s how you can inspect and modify streamed LLM responses:

```codeBlockLines_e6Vv
def stream(self, event: dict) -> dict:
    print(event)  # Print each incoming chunk for inspection
    return event

```

> **Example Streamed Events:**

```codeBlockLines_e6Vv
{'id': 'chatcmpl-B4l99MMaP3QLGU5uV7BaBM0eDS0jb','choices': [{'delta': {'content': 'Hi'}}]}
{'id': 'chatcmpl-B4l99MMaP3QLGU5uV7BaBM0eDS0jb','choices': [{'delta': {'content': '!'}}]}
{'id': 'chatcmpl-B4l99MMaP3QLGU5uV7BaBM0eDS0jb','choices': [{'delta': {'content': ' 😊'}}]}

```

📖 **What Happens?**

- Each line represents a **small fragment** of the model's streamed response.
- The **`delta.content` field** contains the progressively generated text.

##### 🔄 Example: Filtering Out Emojis from Streamed Data [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-example-filtering-out-emojis-from-streamed-data "Direct link to 🔄 Example: Filtering Out Emojis from Streamed Data")

```codeBlockLines_e6Vv
def stream(self, event: dict) -> dict:
    for choice in event.get("choices", []):
        delta = choice.get("delta", {})
        if "content" in delta:
            delta["content"] = delta["content"].replace("😊", "")  # Strip emojis
    return event

```

📖 **Before:** `"Hi 😊"`

📖 **After:** `"Hi"`

* * *

#### 4️⃣ **`outlet` Function (Output Post-Processing)** [​](https://docs.openwebui.com/features/plugin/functions/filter/\#4%EF%B8%8F%E2%83%A3-outlet-function-output-post-processing "Direct link to 4️⃣-outlet-function-output-post-processing")

The `outlet` function is like a **proofreader**: tidy up the AI's response (or make final changes) _after it’s processed by the LLM._

📤 **Input**:

- **`body`**: This contains **all current messages** in the chat (user history + LLM replies).

🚀 **Your Task**: Modify this `body`. You can clean, append, or log changes, but be mindful of how each adjustment impacts the user experience.

💡 **Best Practices**:

- Prefer logging over direct edits in the outlet (e.g., for debugging or analytics).
- If heavy modifications are needed (like formatting outputs), consider using the **pipe function** instead.

💡 **Example Use Case**: Strip out sensitive API responses you don't want the user to see:

```codeBlockLines_e6Vv
def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
    for message in body["messages"]:
        message["content"] = message["content"].replace("<API_KEY>", "[REDACTED]")
    return body

```

* * *

## 🌟 Filters in Action: Building Practical Examples [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-filters-in-action-building-practical-examples "Direct link to 🌟 Filters in Action: Building Practical Examples")

Let’s build some real-world examples to see how you’d use Filters!

### 📚 Example \#1: Add Context to Every User Input [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-example-1-add-context-to-every-user-input "Direct link to 📚 Example \#1: Add Context to Every User Input")

Want the LLM to always know it's assisting a customer in troubleshooting software bugs? You can add instructions like **"You're a software troubleshooting assistant"** to every user query.

```codeBlockLines_e6Vv
class Filter:
    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        context_message = {
            "role": "system",
            "content": "You're a software troubleshooting assistant."
        }
        body.setdefault("messages", []).insert(0, context_message)
        return body

```

* * *

### 📚 Example \#2: Highlight Outputs for Easy Reading [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-example-2-highlight-outputs-for-easy-reading "Direct link to 📚 Example \#2: Highlight Outputs for Easy Reading")

Returning output in Markdown or another formatted style? Use the `outlet` function!

```codeBlockLines_e6Vv
class Filter:
    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Add "highlight" markdown for every response
        for message in body["messages"]:
            if message["role"] == "assistant":  # Target model response
                message["content"] = f"**{message['content']}**"  # Highlight with Markdown
        return body

```

* * *

## 🚧 Potential Confusion: Clear FAQ 🛑 [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-potential-confusion-clear-faq- "Direct link to 🚧 Potential Confusion: Clear FAQ 🛑")

### **Q: How Are Filters Different From Pipe Functions?** [​](https://docs.openwebui.com/features/plugin/functions/filter/\#q-how-are-filters-different-from-pipe-functions "Direct link to q-how-are-filters-different-from-pipe-functions")

Filters modify data **going to** and **coming from models** but do not significantly interact with logic outside of these phases. Pipes, on the other hand:

- Can integrate **external APIs** or significantly transform how the backend handles operations.
- Expose custom logic as entirely new "models."

### **Q: Can I Do Heavy Post-Processing Inside `outlet`?** [​](https://docs.openwebui.com/features/plugin/functions/filter/\#q-can-i-do-heavy-post-processing-inside-outlet "Direct link to q-can-i-do-heavy-post-processing-inside-outlet")

You can, but **it’s not the best practice.**:

- **Filters** are designed to make lightweight changes or apply logging.
- If heavy modifications are required, consider a **Pipe Function** instead.

* * *

## 🎉 Recap: Why Build Filter Functions? [​](https://docs.openwebui.com/features/plugin/functions/filter/\#-recap-why-build-filter-functions "Direct link to 🎉 Recap: Why Build Filter Functions?")

By now, you’ve learned:

1. **Inlet** manipulates **user inputs** (pre-processing).
2. **Stream** intercepts and modifies **streamed model outputs** (real-time).
3. **Outlet** tweaks **AI outputs** (post-processing).
4. Filters are best for lightweight, real-time alterations to the data flow.
5. With **Valves**, you empower users to configure Filters dynamically for tailored behavior.

* * *

🚀 **Your Turn**: Start experimenting! What small tweak or context addition could elevate your Open WebUI experience? Filters are fun to build, flexible to use, and can take your models to the next level!

Happy coding! ✨

- [🌊 What Are Filters in Open WebUI?](https://docs.openwebui.com/features/plugin/functions/filter/#-what-are-filters-in-open-webui)
- [🗺️ Structure of a Filter Function: The Skeleton](https://docs.openwebui.com/features/plugin/functions/filter/#%EF%B8%8F-structure-of-a-filter-function-the-skeleton)
  - [🦴 Basic Skeleton of a Filter](https://docs.openwebui.com/features/plugin/functions/filter/#-basic-skeleton-of-a-filter)
  - [🎯 Key Components Explained](https://docs.openwebui.com/features/plugin/functions/filter/#-key-components-explained)
- [🌟 Filters in Action: Building Practical Examples](https://docs.openwebui.com/features/plugin/functions/filter/#-filters-in-action-building-practical-examples)
  - [📚 Example #1: Add Context to Every User Input](https://docs.openwebui.com/features/plugin/functions/filter/#-example-1-add-context-to-every-user-input)
  - [📚 Example #2: Highlight Outputs for Easy Reading](https://docs.openwebui.com/features/plugin/functions/filter/#-example-2-highlight-outputs-for-easy-reading)
- [🚧 Potential Confusion: Clear FAQ 🛑](https://docs.openwebui.com/features/plugin/functions/filter/#-potential-confusion-clear-faq-)
  - [**Q: How Are Filters Different From Pipe Functions?**](https://docs.openwebui.com/features/plugin/functions/filter/#q-how-are-filters-different-from-pipe-functions)
  - [**Q: Can I Do Heavy Post-Processing Inside `outlet`?**](https://docs.openwebui.com/features/plugin/functions/filter/#q-can-i-do-heavy-post-processing-inside-outlet)
- [🎉 Recap: Why Build Filter Functions?](https://docs.openwebui.com/features/plugin/functions/filter/#-recap-why-build-filter-functions)