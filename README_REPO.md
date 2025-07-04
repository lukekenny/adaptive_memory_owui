# 🧠 OpenWebUI Adaptive Memory Plugin

[![OpenWebUI](https://img.shields.io/badge/OpenWebUI-Compatible-blue)](https://github.com/open-webui/open-webui)
[![Version](https://img.shields.io/badge/Version-4.0-green)](https://github.com/alackmann/OWUI_adaptive_memory)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)

Give your AI conversations **persistent memory**! This plugin enables OpenWebUI to remember user preferences, facts, and context across all conversations, creating truly personalized AI interactions.

## ✨ Key Features

- 🧠 **Persistent Memory**: Remembers user preferences, identity, goals, and relationships
- 🎯 **4-Field Quick Setup**: Most users configure just 4 settings and you're done
- 🤖 **Multi-LLM Support**: Works with Ollama, OpenAI, Google Gemini, and compatible APIs
- 🔧 **Smart Auto-Config**: Automatically configures 90+ advanced settings based on your needs
- 🏷️ **Memory Organization**: Categorize memories into Personal, Work, Hobbies, Technical, etc.
- 🔍 **Duplicate Prevention**: Advanced deduplication prevents storing the same information twice
- 🛡️ **User Isolation**: Each user's memories are completely separate and secure
- 📱 **Sub-3B Model Support**: Works with small models via intelligent JSON repair system

## 🚀 Quick Start

### 1. Installation (30 seconds)
1. Open your OpenWebUI admin panel
2. Go to **Workspace** → **Functions** → **+** (Add Function)
3. Copy and paste the entire `adaptive_memory_v4.0.py` file
4. Click **Save** and toggle **Enable**

### 2. Configuration (1 minute)
Just configure these 4 settings:

| Setting | Options | Description |
|---------|---------|-------------|
| 🎯 **Setup Mode** | `simple` / `advanced` | Choose "simple" for auto-configuration |
| 🤖 **LLM Provider** | `ollama` / `openai_compatible` / `gemini` | Your AI service |
| 📝 **Model Name** | e.g., `llama3:latest` | Your AI model |
| 🧠 **Memory Mode** | `minimal` / `balanced` / `comprehensive` | How much to remember |

### 3. Test It Out
Try these conversations:
- "My favorite color is blue and I love hiking"
- "I work as a software engineer in San Francisco"
- "I prefer dark roast coffee in the morning"

The AI will remember these details for all future conversations!

## 🎯 Use Cases

### Personal Assistant
```
User: "My favorite color is blue"
AI: "I'll remember that you prefer blue!"

[Later conversation]
User: "What colors should I paint my room?"
AI: "Since you love blue, perhaps a calming blue accent wall..."
```

### Work Context
```
User: "I'm working on a Python project using FastAPI"
AI: "Got it! I'll remember your Python/FastAPI project."

[Days later]
User: "How do I handle authentication?"
AI: "For your FastAPI project, here are some authentication patterns..."
```

### Learning Journey
```
User: "I'm learning machine learning, focusing on computer vision"
AI: "I'll remember your ML/CV learning path!"

[Weeks later]
User: "What should I study next?"
AI: "Given your computer vision focus, you might want to explore..."
```

## 🏗️ Architecture

### Memory Types
- **🆔 Identity**: Name, role, location, background
- **❤️ Preferences**: Likes, dislikes, favorites
- **🎯 Goals**: Aspirations, projects, learning objectives
- **👥 Relationships**: Family, friends, colleagues
- **🏠 Possessions**: Devices, tools, belongings
- **🎭 Behaviors**: Patterns, habits, communication style

### Memory Organization
- **📁 Memory Banks**: Organize by Personal, Work, Hobbies, Technical
- **🏷️ Tags**: Automatic tagging for easy retrieval
- **📊 Relevance Scoring**: AI determines which memories matter for each conversation
- **🔄 Auto-Summarization**: Combines related memories to save space

## ⚙️ Advanced Configuration

<details>
<summary>🔧 Advanced Settings (Click to expand)</summary>

The plugin includes 90+ configurable settings organized into logical groups:

### 🎛️ Memory Behavior
- Memory sensitivity (how easily memories are created)
- Maximum memories per user
- Relevance thresholds
- Duplicate detection settings

### 🏛️ Memory Organization
- Custom memory categories
- Default categorization
- Timezone settings

### 🔧 Technical Settings
- Embedding providers and models
- API configurations
- Performance tuning
- Error handling

### 🎨 UI & Display
- Memory display formats
- Status message visibility
- Memory type toggles

</details>

## 🔒 Security & Privacy

- **🔐 User Isolation**: Each user's memories are completely separate
- **🛡️ No Cross-User Access**: Impossible for users to see others' memories
- **🔑 API Key Security**: Keys are securely handled within OpenWebUI
- **📝 Audit Trail**: All memory operations are logged for transparency

## 🚀 Performance

- **⚡ Fast Retrieval**: Vector similarity search for instant memory lookup
- **💾 Efficient Storage**: Smart deduplication and summarization
- **🔄 Async Processing**: Non-blocking memory operations
- **📊 Monitoring**: Built-in performance tracking and health checks

## 🛠️ Development

### Requirements
- Python 3.8+
- OpenWebUI instance
- No additional dependencies (everything is included)

### File Structure
```
adaptive_memory_v4.0.py    # Main plugin file (THE ONLY FILE YOU NEED)
README.md                  # Detailed documentation
Other Versions/            # Previous versions for reference
OWUI tech-docs/           # OpenWebUI architecture documentation
```

### Testing
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## 📈 Roadmap

- [ ] **Visual Memory Browser**: Web interface for exploring stored memories
- [ ] **Memory Analytics**: Insights into memory patterns and usage
- [ ] **Advanced Search**: Query memories with natural language
- [ ] **Memory Sharing**: Optional team/family memory sharing features
- [ ] **Integration APIs**: Connect with external knowledge bases

## 🤝 Contributing

We welcome contributions! Please:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Ensure the monolithic structure is maintained
4. **Test thoroughly**: All existing functionality must continue working
5. **Submit a PR**: Include detailed description of changes

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **📚 Documentation**: Check the [README.md](README.md) for detailed setup
- **🐛 Issues**: Report bugs via [GitHub Issues](https://github.com/alackmann/OWUI_adaptive_memory/issues)
- **💡 Discussions**: Join conversations in [GitHub Discussions](https://github.com/alackmann/OWUI_adaptive_memory/discussions)
- **🌟 Star the repo**: If this helps you, please star the repository!

## 🏆 Credits

- **OpenWebUI Team**: For creating an amazing platform
- **Contributors**: Everyone who has helped improve this plugin
- **Community**: Users who provided feedback and feature requests

---

**Transform your AI conversations from stateless interactions to personalized, context-aware experiences!** 🚀