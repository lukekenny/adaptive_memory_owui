# OpenWebUI Adaptive Memory Plugin v4.0

A monolithic OpenWebUI filter function that provides persistent memory capabilities for AI conversations, enabling personalized responses based on user history and preferences.

## What's New in v4.0

- **36% Code Reduction**: Optimized from 9,156 to 5,853 lines while maintaining all features
- **Security Fixes**: Updated dependencies to patch CVE-2024-3772, CVE-2024-23334, CVE-2024-27306
- **JSON Repair System**: Added support for sub-3B models with improved JSON parsing
- **Gemini API Fix**: Corrected authentication method for Gemini API integration
- **Enhanced Error Handling**: Improved circuit breaker pattern for better fault tolerance
- **Performance Optimizations**: Streamlined memory operations and reduced overhead

## Installation

OpenWebUI functions are installed by copying the code directly into the OpenWebUI interface:

1. **Open OpenWebUI Admin Panel**
   - Navigate to your OpenWebUI instance
   - Login with admin credentials

2. **Add the Function**
   - Go to "Workspace" → "Functions"
   - Click the "+" button to add a new function
   - Copy the entire contents of `adaptive_memory_v4.0.py`
   - Paste into the code editor
   - Click "Save"

3. **Enable the Filter**
   - Toggle the switch to enable the function
   - Go to "Workspace" → "Models"
   - Select the models you want to use with adaptive memory
   - Assign the filter to those models

4. **Configure Settings (Optional)**
   - Click on the function name to access configuration
   - Adjust settings like LLM provider, memory thresholds, etc.
   - Save your configuration

## Testing the Installation

After installation, test with simple statements:
- "My favorite color is blue"
- "I prefer Python over Java"
- "I work as a software engineer"

The filter will extract and remember these preferences for future conversations.

## Features

- **User-Specific Memory**: Each user has isolated memory storage
- **Multi-LLM Support**: Works with Ollama, OpenAI-compatible APIs, and Gemini
- **Automatic Memory Extraction**: Identifies and stores important information
- **Context Injection**: Seamlessly adds relevant memories to conversations
- **Flexible Memory Banks**: Organize memories by categories
- **Deduplication**: Prevents duplicate memory storage
- **Sub-3B Model Support**: JSON repair system for smaller models

## Configuration

The filter includes configurable valves for:
- LLM provider settings (Ollama, OpenAI, Gemini)
- Memory extraction thresholds
- Maximum memories to inject
- Debug logging options
- Memory bank categories

## Requirements

- OpenWebUI instance (latest version recommended)
- Python 3.8+ (for OpenWebUI)
- No additional installation steps required

## File Structure

```
adaptive_memory_v4.0.py    # Main filter file (copy this to OpenWebUI)
requirements.txt           # Dependencies reference (for development only)
tests/                     # Test suite (for development only)
```

## Troubleshooting

### Filter Not Working
- Ensure the filter is enabled for your model
- Check OpenWebUI logs for error messages
- Verify the filter saved successfully without syntax errors

### Memory Not Persisting
- Check that the filter is processing messages (look for log entries)
- Ensure the user is properly authenticated
- Verify the model has the filter assigned

### Import Errors
- The filter includes all necessary imports
- If you see import errors, ensure you copied the complete file
- Check that your OpenWebUI Python environment has basic packages

## Development

For development and testing:
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

## Security Notes

- All user memories are isolated by user ID
- No cross-user data access is possible
- API keys should be configured securely in OpenWebUI
- Dependencies have been updated to patch known vulnerabilities

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please ensure:
- Code maintains the monolithic structure (single file)
- All features remain functional
- Tests pass successfully
- Security best practices are followed