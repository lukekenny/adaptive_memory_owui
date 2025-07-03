# OWUI Adaptive Memory Plugin - Installation Guide

## Quick Installation Steps

### 1. Check Requirements
- Python 3.8+
- OpenWebUI installed and running
- Admin access to OpenWebUI

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install the Filter
1. Open OpenWebUI admin panel
2. Go to "Workspace" → "Functions"  
3. Click "+" to add new function
4. Upload the filter file OR copy/paste the content
5. Click "Save" to install
6. Enable the filter for your desired models

### 4. Troubleshooting Common Issues

#### "No Tools class found" Error
This error is misleading. The real causes are usually:
- File too large (try the minimal version first)
- Missing dependencies 
- Import errors during loading
- OpenWebUI version compatibility

**Solutions:**
1. Try the minimal filter first (`adaptive_memory_minimal.py`)
2. Check OpenWebUI logs for actual error details
3. Ensure all dependencies are installed
4. Use the synchronous version if async fails

#### Filter Not Loading
1. Check OpenWebUI logs for detailed errors
2. Verify file is valid Python syntax
3. Test with minimal version first
4. Check file permissions

#### Memory Not Working
1. Enable debug logging in filter valves
2. Check OpenWebUI logs for filter execution
3. Verify filter is enabled for the correct model
4. Test with simple preference statements

#### Dependencies Missing
```bash
# Install core dependencies
pip install sentence-transformers pydantic aiohttp numpy scikit-learn

# For development/testing
pip install pytest pytest-asyncio pytest-mock
```

### 5. Testing Installation

After installation, test with these simple prompts:
- "My favorite color is blue"
- "I live in New York"  
- "I work as a software engineer"

Check the logs to see if the filter processes these inputs.

### 6. Advanced Configuration

Once basic installation works, you can:
- Configure LLM providers in filter valves
- Adjust memory thresholds
- Enable advanced features
- Switch to the full version

### 7. Getting Help

If you encounter issues:
1. Check OpenWebUI logs (detailed error information)
2. Run the diagnostic script: `python install_diagnostics.py`
3. Try the minimal version first
4. Report issues with full log details

## Installation Process Options

### Option 1: Minimal Version (Recommended First)
Start with `adaptive_memory_minimal.py` to test basic installation:
- Smallest file size
- Minimal dependencies  
- Simple logging to verify functionality
- Good for troubleshooting installation issues

### Option 2: Synchronous Version
Use `adaptive_memory_v4.0_sync.py` if async methods cause issues:
- Full functionality with sync methods
- Better compatibility with some OpenWebUI versions
- Intermediate complexity

### Option 3: Full Version  
Use `adaptive_memory_v4.0.py` for complete functionality:
- All advanced features
- Filter orchestration system
- Full async support
- Largest file size

## Step-by-Step Installation

### 1. Prepare Your Environment
```bash
# Navigate to the plugin directory
cd /path/to/OWUI_adaptive_memory

# Install dependencies
pip install -r requirements.txt

# Run diagnostics (optional but recommended)
python install_diagnostics.py
```

### 2. Choose Your Version
Based on your needs and any issues found by diagnostics:

- **First time?** → Start with `adaptive_memory_minimal.py`
- **Installation issues?** → Try `adaptive_memory_v4.0_sync.py`  
- **Want full features?** → Use `adaptive_memory_v4.0.py`

### 3. Install in OpenWebUI

#### Method A: File Upload
1. Open OpenWebUI admin interface
2. Navigate to "Workspace" → "Functions"
3. Click the "+" button to add a new function
4. Click "Upload" and select your chosen filter file
5. Click "Save"

#### Method B: Copy/Paste
1. Open your chosen filter file in a text editor
2. Copy all the content
3. In OpenWebUI, go to "Workspace" → "Functions"
4. Click "+" to add a new function
5. Paste the content into the code editor
6. Click "Save"

### 4. Enable the Filter
1. After saving, the filter should appear in your functions list
2. Toggle the switch to enable it
3. Click on the filter name to configure it
4. Go to "Workspace" → "Models"  
5. Find your model and assign the filter to it

### 5. Test the Installation
1. Start a new chat with a model that has the filter enabled
2. Send a test message: "My favorite programming language is Python"
3. Check the OpenWebUI logs to see if the filter processed the message
4. Look for log messages like "Inlet called - processing user input"

## Common Installation Scenarios

### Scenario 1: Fresh OpenWebUI Installation
- Use the minimal version first
- Verify basic functionality 
- Upgrade to sync version if needed
- Finally try the full version

### Scenario 2: Existing OpenWebUI with Other Filters
- Check for conflicts with existing filters
- Use the filter orchestration features in the full version
- Monitor performance impact

### Scenario 3: Docker Installation
- Ensure dependencies are installed in the container
- Check volume mounts for persistence
- Verify network access for LLM APIs

### Scenario 4: Development Environment
- Install all testing dependencies
- Use the full version with debug logging
- Run the test suite to verify functionality

## Advanced Configuration

### LLM Provider Configuration
Configure the filter to work with your LLM provider:

```python
# For Ollama (default)
llm_provider_type = "ollama"
llm_api_endpoint_url = "http://localhost:11434/api/chat"
llm_model_name = "llama3.2"

# For OpenAI-compatible APIs
llm_provider_type = "openai_compatible"  
llm_api_endpoint_url = "https://api.openai.com/v1/chat/completions"
llm_model_name = "gpt-3.5-turbo"
```

### Memory Bank Configuration
Customize memory categories:

```python
allowed_memory_banks = ["Personal", "Work", "Hobbies", "Technical"]
default_memory_bank = "Personal"
```

### Performance Tuning
Adjust thresholds for optimal performance:

```python
similarity_threshold = 0.65  # Lower = more memories injected
max_memories_to_inject = 5   # Limit context size
```

## Troubleshooting Guide

### Problem: Filter Shows as Installed but Doesn't Work
**Symptoms:**
- Filter appears in functions list
- No log messages during chat
- Memory functionality not working

**Solutions:**
1. Check if filter is enabled for the model you're using
2. Verify the model assignment in "Workspace" → "Models"
3. Enable debug logging in filter valves
4. Check OpenWebUI logs for error messages

### Problem: "Import Error" During Installation
**Symptoms:**
- Error during save in OpenWebUI
- Missing module messages

**Solutions:**
1. Install missing dependencies: `pip install -r requirements.txt`
2. Try the minimal version with fewer dependencies
3. Check Python version compatibility (3.8+)

### Problem: Filter Causes OpenWebUI to Crash
**Symptoms:**
- OpenWebUI becomes unresponsive after enabling filter
- High CPU/memory usage

**Solutions:**
1. Disable the filter immediately
2. Try the minimal version first
3. Check for infinite loops in logs
4. Reduce processing complexity in valves

### Problem: Memory Not Persisting
**Symptoms:**
- Memories extracted but not saved
- No memory injection in conversations

**Solutions:**
1. Check OpenWebUI database permissions
2. Verify user authentication is working
3. Enable debug logging to trace memory operations
4. Test with simple preference statements

## Getting Support

### Before Asking for Help
1. Run `python install_diagnostics.py` and share the output
2. Check OpenWebUI logs for detailed error messages
3. Try the minimal version to isolate issues
4. Note your OpenWebUI version and environment details

### What to Include in Support Requests
- OpenWebUI version
- Python version  
- Operating system
- Installation method (Docker, native, etc.)
- Complete error messages from logs
- Output from diagnostic script
- Steps to reproduce the issue

### Log Locations
- **Docker:** `docker logs openwebui`
- **Native:** Check OpenWebUI startup logs
- **Filter logs:** Enable debug logging in filter valves

## Security Considerations

### API Keys
- Store LLM API keys securely
- Use environment variables when possible
- Don't commit keys to version control

### Memory Privacy
- Memories are user-specific and isolated
- Consider data retention policies
- Implement memory cleanup if needed

### Network Security
- Verify LLM API endpoints are trusted
- Use HTTPS for remote APIs
- Consider firewall rules for local services

## Performance Optimization

### For Large User Bases
- Monitor memory usage growth
- Implement periodic cleanup
- Consider external storage for memories
- Use caching for frequently accessed data

### For Resource-Constrained Systems
- Use the minimal version
- Reduce similarity thresholds  
- Limit max memories per user
- Disable advanced features

This guide should help you successfully install and configure the OWUI Adaptive Memory Plugin. Remember to start with the minimal version and gradually work up to the full version as you verify each step works correctly.