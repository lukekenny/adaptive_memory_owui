# Changelog

## [4.0.0] - 2024-01-07

### Major Improvements
- **Code Optimization**: Reduced code size by 36% (from 9,156 to 5,853 lines) while maintaining all functionality
- **Security Updates**: Patched multiple vulnerabilities:
  - CVE-2024-3772 in pydantic
  - CVE-2024-23334 and CVE-2024-27306 in aiohttp
  - Updated all dependencies to latest secure versions

### New Features
- **JSON Repair System**: Added intelligent JSON parsing for sub-3B models that may produce malformed JSON
- **Enhanced Error Handling**: Implemented circuit breaker pattern for better fault tolerance
- **Improved Gemini Support**: Fixed Gemini API authentication (changed from Bearer token to URL parameter)

### Optimizations
- Simplified filter orchestration system
- Streamlined Valves configuration class
- Removed verbose logging while maintaining debug capabilities
- Optimized memory operations for better performance
- Reduced import overhead

### Bug Fixes
- Fixed Gemini API 400/401 errors by using correct request format
- Corrected memory extraction JSON parsing issues
- Resolved edge cases in user isolation
- Fixed memory deduplication logic

### Technical Changes
- Maintained monolithic single-file structure for OpenWebUI compatibility
- Preserved all core functionality:
  - User-specific memory isolation
  - Multi-LLM support (Ollama, OpenAI, Gemini)
  - Memory extraction and injection
  - Filter orchestration
  - Configurable valves
- Updated test suite to match new implementation

### Dependencies
- pydantic >=2.4.0,<3.0.0
- numpy >=1.24.0,<2.0.0
- aiohttp >=3.9.4,<4.0.0
- sentence-transformers >=2.2.0
- scikit-learn >=1.3.0
- pytz >=2024.1

### Installation
- Clarified installation process: Copy-paste into OpenWebUI function editor
- Removed unnecessary installation scripts
- Updated documentation to reflect correct installation method

## [3.2] - Previous Version
- Last version before major optimization
- Full feature set but with larger codebase
- Some known security vulnerabilities in dependencies