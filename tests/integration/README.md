# OpenWebUI Adaptive Memory Plugin - Integration Tests

This directory contains comprehensive integration tests for the OpenWebUI Adaptive Memory Plugin, focusing on API interactions, memory lifecycle management, error handling, and edge cases.

## Overview

The integration tests validate:

1. **Memory Lifecycle**: Extraction, storage, retrieval, update, and deletion
2. **API Interactions**: OpenWebUI API calls with proper error handling
3. **Error Scenarios**: Rate limiting, timeouts, network failures, invalid responses
4. **Edge Cases**: Large batches, concurrent operations, deduplication, user isolation
5. **Performance**: Circuit breakers, caching, graceful degradation
6. **Data Integrity**: Validation, transactional consistency

## Test Structure

```
tests/integration/
├── README.md                           # This file
├── __init__.py                        # Package initialization
├── fixtures.py                        # Shared test fixtures and mocks
├── test_config.py                     # Test configuration and scenarios
├── run_integration_tests.py           # Test runner script
├── test_openwebui_integration.py     # Main integration tests
├── test_openwebui_interface.py       # Interface compatibility tests
├── test_example_with_mocks.py        # Example test patterns
└── mocks/                            # Mock implementations
    ├── __init__.py
    ├── openwebui_api_mock.py         # OpenWebUI API mock
    ├── llm_api_mock.py               # LLM API mocks
    ├── embedding_api_mock.py         # Embedding API mocks
    ├── websocket_mock.py             # WebSocket mock
    └── recording_utils.py            # Request/response recording
```

## Running Tests

### Basic Usage

```bash
# Run all integration tests with default settings
python tests/integration/run_integration_tests.py

# Run a specific scenario
python tests/integration/run_integration_tests.py --scenario high_load

# Run all scenarios
python tests/integration/run_integration_tests.py --all-scenarios

# Run with debug output
python tests/integration/run_integration_tests.py --debug
```

### Direct pytest Usage

```bash
# Run specific test file
pytest tests/integration/test_openwebui_integration.py -v

# Run specific test class
pytest tests/integration/test_openwebui_integration.py::TestMemoryLifecycle -v

# Run specific test method
pytest tests/integration/test_openwebui_integration.py::TestMemoryLifecycle::test_memory_extraction_from_conversation -v

# Run with specific markers
pytest tests/integration -m "not slow" -v
```

## Test Scenarios

### 1. Happy Path
- Normal operation with minimal errors
- Tests basic functionality
- Default scenario for quick validation

### 2. High Load
- High volume concurrent operations
- Tests scalability and performance
- Validates batch operations

### 3. Unreliable Network
- Simulates network issues
- Tests retry logic and resilience
- Validates timeout handling

### 4. API Migration
- Tests compatibility between API versions
- Validates backward compatibility
- Tests version negotiation

## Test Classes

### TestMemoryLifecycle
Tests the complete memory lifecycle:
- Memory extraction from conversations
- Storage via OpenWebUI API
- Retrieval and search operations
- Update and deletion
- Bulk operations

### TestErrorHandling
Tests error scenarios:
- API failures with retry logic
- Rate limiting detection
- Network timeouts
- Invalid response handling

### TestEdgeCases
Tests edge cases:
- Large memory batches
- Concurrent operations
- Memory deduplication
- User data isolation

### TestFilterIntegration
Tests filter's integration with APIs:
- Inlet memory injection
- Outlet memory extraction
- Command processing
- API version compatibility

### TestPerformanceAndResilience
Tests performance characteristics:
- Circuit breaker functionality
- Memory caching
- Graceful degradation
- Compression for large contexts

### TestDataIntegrity
Tests data integrity:
- Memory data validation
- Transactional consistency

## Configuration

### Environment Variables

```bash
# API Configuration
export OPENWEBUI_BASE_URL="http://localhost:8080"
export OPENWEBUI_API_KEY="your_api_key"

# Test Configuration
export TEST_ENV="local"
export TEST_DATA_DIR="./tests/integration/test_data"
export ENABLE_PERFORMANCE_TESTS="true"
export ENABLE_DEBUG_LOGGING="false"
export SAVE_API_RECORDINGS="false"

# Timeout Settings
export DEFAULT_TIMEOUT_SECONDS="30"
export LONG_RUNNING_TIMEOUT_SECONDS="300"
```

### Test Data Configuration

Edit `test_config.py` to customize:
- API endpoints and versions
- Test data sizes
- Error injection rates
- Performance thresholds
- Validation rules

## Mock Servers

### OpenWebUI Memory API Mock

Simulates the OpenWebUI memory API with:
- CRUD operations for memories
- Rate limiting simulation
- Error injection
- Request/response recording

### LLM API Mock

Supports multiple providers:
- OpenAI
- Ollama
- Anthropic
- Custom providers

### Embedding API Mock

Simulates embedding generation:
- Local model simulation
- API-based embeddings
- Deterministic embeddings for testing

### WebSocket Mock

Simulates real-time connections:
- Event streaming
- Connection management
- Network issue simulation

## Writing New Tests

### 1. Create Test Function

```python
@pytest.mark.asyncio
async def test_new_feature(openwebui_memory_api_mock):
    """Test description"""
    # Arrange
    api_mock = openwebui_memory_api_mock
    user_id = "test_user"
    
    # Act
    result = await api_mock.some_operation(user_id)
    
    # Assert
    assert result is not None
    assert result.status == "success"
```

### 2. Use Fixtures

```python
@pytest.mark.asyncio
async def test_with_fixtures(
    openwebui_memory_api_mock,
    memory_search_scenario,
    performance_monitor
):
    """Test using multiple fixtures"""
    # Fixtures provide pre-configured test data and monitoring
    pass
```

### 3. Add Error Scenarios

```python
@pytest.mark.asyncio
async def test_error_handling(openwebui_memory_api_mock_with_errors):
    """Test error handling"""
    api_mock = openwebui_memory_api_mock_with_errors
    api_mock.error_sequence = [APIError.RATE_LIMIT, None]
    
    # Test retry logic
    with pytest.raises(Exception):
        await api_mock.create_memory(...)
```

## Best Practices

1. **Use Fixtures**: Leverage existing fixtures for common setup
2. **Test Isolation**: Each test should be independent
3. **Mock External Services**: Use mock servers for all external dependencies
4. **Test Data**: Use generators for consistent test data
5. **Error Testing**: Test both success and failure paths
6. **Performance**: Monitor performance metrics in tests
7. **Async Testing**: Use `pytest.mark.asyncio` for async tests

## Debugging Tests

### Enable Debug Logging

```bash
# Via test runner
python tests/integration/run_integration_tests.py --debug

# Via pytest
pytest tests/integration -v -s --log-cli-level=DEBUG
```

### Save API Recordings

```bash
export SAVE_API_RECORDINGS="true"
pytest tests/integration/test_openwebui_integration.py
# Check test_recordings.json for API calls
```

### Run Single Test

```bash
# Focus on a specific test for debugging
pytest tests/integration/test_openwebui_integration.py::TestMemoryLifecycle::test_memory_extraction_from_conversation -v -s
```

## Performance Testing

### Run Performance Tests

```bash
# Enable performance tests
python tests/integration/run_integration_tests.py --performance

# Monitor metrics
pytest tests/integration --performance-monitor
```

### Performance Metrics

Tests monitor:
- Response times
- Memory usage
- CPU utilization
- Throughput
- Error rates

## Continuous Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r tests/requirements-test.txt
      - name: Run integration tests
        run: |
          python tests/integration/run_integration_tests.py --all-scenarios
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in PYTHONPATH
2. **Timeout Errors**: Increase timeout values in configuration
3. **Mock Server Issues**: Check mock server initialization
4. **Async Errors**: Ensure proper async/await usage

### Getting Help

1. Check test logs in `tests/integration/logs/`
2. Review mock server recordings
3. Enable debug mode for detailed output
4. Check fixture initialization

## Contributing

When adding new integration tests:

1. Follow existing patterns
2. Add appropriate fixtures
3. Document test purpose
4. Include error scenarios
5. Add performance checks
6. Update this README

## License

Same as the main project.