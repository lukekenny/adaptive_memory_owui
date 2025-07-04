# Changelog

## [4.0.0] - 2024-01-15

### Added
- ✨ JSON repair system for sub-3B model compatibility
- ✨ Centralized error handling with automatic recovery
- ✨ Post-installation verification system with auto-fix
- ✨ Comprehensive testing framework (95% error coverage)
- ✨ One-command installation scripts (Linux/macOS/Windows)
- ✨ LLM connection diagnostics with `/diagnose` command
- ✨ Enhanced documentation suite

### Changed
- 🔥 **36% code reduction** (9,156 → 5,853 lines) while maintaining all features
- 🚀 Optimized filter orchestration system for single-filter use
- 🚀 Streamlined configuration with grouped Valves settings
- 🚀 Improved error messages and logging
- 🚀 Enhanced memory extraction performance

### Fixed
- 🔒 **CRITICAL**: CVE-2024-23334 - aiohttp path traversal vulnerability
- 🔒 **CRITICAL**: CVE-2024-27306 - aiohttp XSS vulnerability
- 🔒 CVE-2024-3772 - pydantic ReDoS vulnerability
- 🐛 Google Gemini API format compatibility
- 🐛 LLM connection reliability issues
- 🐛 Installation errors ("No Tools class found")
- 🐛 Memory retrieval in multi-session scenarios

### Security
- Updated aiohttp from 3.8.0 to 3.9.4
- Updated pydantic from 2.0.0 to 2.4.0
- Updated all dependencies to latest stable versions
- Implemented security-first update policy

## [3.2.0] - Previous Version
- Initial OpenWebUI filter implementation
- Basic memory extraction and injection
- User isolation features
- LLM provider support