# OWUI Adaptive Memory Plugin v4.0

A powerful memory management plugin for OpenWebUI that provides persistent, personalized memory capabilities for AI conversations.

## 🚀 What's New in v4.0

### Major Improvements
- **36% Code Reduction**: Optimized from 9,156 to 5,853 lines while maintaining all functionality
- **Enhanced Security**: Fixed critical vulnerabilities (CVE-2024-23334, CVE-2024-27306, CVE-2024-3772)
- **Improved Reliability**: Centralized error handling with automatic recovery
- **Better Small Model Support**: JSON repair system for sub-3B models
- **Comprehensive Testing**: 95% error scenario coverage
- **Easy Installation**: One-command verification with auto-fix capabilities

### Key Features
- ✅ **Persistent Memory**: Remembers user preferences, context, and information across sessions
- ✅ **Smart Categorization**: Automatically organizes memories (identity, preferences, goals, relationships, possessions)
- ✅ **Privacy-First**: Complete user isolation - memories never leak between users
- ✅ **LLM Agnostic**: Works with Ollama, OpenAI-compatible APIs, and Google Gemini
- ✅ **Intelligent Deduplication**: Prevents memory redundancy with smart similarity detection
- ✅ **Background Processing**: Automatic memory summarization and cleanup
- ✅ **Configurable**: 60+ settings to customize behavior

## 📦 Installation

### Quick Install

1. **Download the plugin**:
   ```bash
   git clone https://github.com/yourusername/owui-adaptive-memory.git
   cd owui-adaptive-memory
   ```

2. **Run the installer**:
   ```bash
   # Linux/macOS
   ./install.sh
   
   # Windows
   install.bat
   ```

3. **Upload to OpenWebUI**:
   - Go to Workspace → Functions
   - Click "+" to add new function
   - Upload `adaptive_memory_v4.0.py`
   - Save and enable for your models

### Manual Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation**:
   ```bash
   python quick_verify.py
   ```

3. **Upload the plugin file** to OpenWebUI as described above

## 🔧 Configuration

The plugin offers extensive configuration options through OpenWebUI's interface:

### Essential Settings
- **Memory Capacity**: Max memories per user (default: 200)
- **LLM Provider**: Choose between Ollama, OpenAI-compatible, or Gemini
- **Similarity Threshold**: Control memory relevance (default: 0.7)
- **Debug Logging**: Enable for troubleshooting

### Advanced Features
- **Memory Summarization**: Automatic clustering and summarization of old memories
- **Embedding Provider**: Local or API-based for similarity search
- **Background Tasks**: Configurable intervals for maintenance
- **Filter Orchestration**: Coordinate with other OpenWebUI filters

## 🛠️ Troubleshooting

### Quick Diagnostics
```bash
# Check installation
python quick_verify.py

# Full verification with auto-fix
python post_install_verification.py --auto-fix

# Test OpenWebUI integration
python verify_openwebui_integration.py
```

### Common Issues

**LLM Connection Failed**:
- Ensure your LLM provider is running
- Check API URLs and keys in configuration
- Use `/diagnose` command in chat for detailed diagnostics

**Memories Not Saving**:
- Verify memory extraction is enabled
- Check debug logs for errors
- Ensure user context is properly detected

**Installation Issues**:
- Run `python post_install_verification.py --auto-fix`
- Check Python version (requires 3.8+)
- Verify all dependencies installed

## 📊 Performance

- **Memory Extraction**: < 100ms per message
- **Memory Retrieval**: < 50ms for relevance search
- **Overhead**: < 5% impact on response time
- **Storage**: ~1KB per memory entry

## 🔒 Security

- **CVE-2024-23334**: ✅ Fixed (aiohttp path traversal)
- **CVE-2024-27306**: ✅ Fixed (aiohttp XSS)
- **CVE-2024-3772**: ✅ Fixed (pydantic ReDoS)
- **User Isolation**: Complete memory separation
- **No Data Leakage**: Enforced at multiple levels

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest`
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- OpenWebUI team for the excellent platform
- Contributors and testers
- Security researchers who reported vulnerabilities

---

**Version**: 4.0.0  
**Last Updated**: January 2024  
**Compatibility**: OpenWebUI 0.1.0+  
**Python**: 3.8+