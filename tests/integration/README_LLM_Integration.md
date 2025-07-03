# LLM Integration Tests

This directory contains comprehensive integration tests for LLM provider connections and memory extraction workflows in the OWUI Adaptive Memory Plugin.

## Overview

The LLM integration tests validate the `query_llm_with_retry` method and related memory extraction functionality across multiple LLM providers, error scenarios, and real-world usage patterns.

## Test Structure

### Core Test Files

- **`test_llm_integration.py`** - Main test suite with 7 test classes
- **`fixtures/llm_fixtures.py`** - Test fixtures and helper utilities
- **`mocks/llm_api_mock.py`** - Mock LLM API implementations
- **`run_llm_integration_tests.py`** - Test runner script

### Test Classes

#### 1. TestLLMProviderConnections
Tests basic LLM provider connectivity and configuration:

- ✅ OpenAI-compatible API integration
- ✅ Ollama local API integration  
- ✅ Anthropic Claude API integration
- ✅ Google Gemini API integration
- ✅ Custom endpoint support
- ✅ API key validation
- ✅ Request/response formatting

#### 2. TestMemoryExtractionWorkflows
Tests memory extraction and analysis functionality:

- ✅ Conversation analysis for memory extraction
- ✅ Memory importance scoring (1-10 scale)
- ✅ Category classification (personal, professional, technical, etc.)
- ✅ Content filtering for sensitive information
- ✅ Keyword extraction and tagging

#### 3. TestErrorScenarios
Tests error handling and edge cases:

- ✅ API key validation errors (401)
- ✅ Rate limiting scenarios (429)
- ✅ Model availability errors (404)
- ✅ Server errors (500, 502, 503, 504)
- ✅ Response parsing errors
- ✅ Timeout handling
- ✅ JSON mode fallback handling

#### 4. TestCircuitBreakerFunctionality
Tests circuit breaker protection:

- ✅ Circuit breaker opening after failures
- ✅ Circuit breaker recovery after success
- ✅ Endpoint health monitoring
- ✅ Failure threshold management

#### 5. TestStreamingAndFunctionCalling
Tests advanced LLM features:

- ✅ Streaming response handling (NDJSON)
- ✅ Function calling responses
- ✅ Chunk aggregation
- ✅ Partial response handling

#### 6. TestEndToEndMemoryExtraction
Tests complete memory extraction workflows:

- ✅ Complete conversation processing
- ✅ Memory importance filtering by threshold
- ✅ Sensitive content filtering workflow
- ✅ Multi-provider memory extraction
- ✅ Cross-provider consistency validation

#### 7. TestRealWorldScenarios
Tests realistic usage scenarios:

- ✅ Long conversation memory extraction
- ✅ Incremental memory updates
- ✅ Progressive memory refinement
- ✅ Complex professional/personal information handling

## Running the Tests

### Quick Start

```bash
# Run all LLM integration tests
python tests/integration/run_llm_integration_tests.py

# Run specific test category
python tests/integration/run_llm_integration_tests.py provider_connections
python tests/integration/run_llm_integration_tests.py memory_extraction
python tests/integration/run_llm_integration_tests.py error_handling

# Show available categories
python tests/integration/run_llm_integration_tests.py help
```

### Using pytest directly

```bash
# Run all tests with verbose output
pytest tests/integration/test_llm_integration.py -v --asyncio-mode=auto

# Run specific test class
pytest tests/integration/test_llm_integration.py::TestLLMProviderConnections -v

# Run specific test method
pytest tests/integration/test_llm_integration.py::TestMemoryExtractionWorkflows::test_conversation_analysis -v

# Run with coverage
pytest tests/integration/test_llm_integration.py --cov=adaptive_memory_v4.0 --cov-report=html
```

## Test Configuration

### Mock LLM Providers

The tests use mock implementations that simulate real LLM provider APIs:

```python
# OpenAI-compatible mock
openai_mock = OpenAIAPIMock(
    enable_streaming=True,
    enable_rate_limiting=False,
    enable_random_errors=False
)

# Ollama mock with local API simulation
ollama_mock = OllamaAPIMock(
    response_delay_ms=100,
    max_tokens=2048
)

# Custom response patterns
mock.set_response_pattern("memory", "This is about memory extraction")
```

### Test Data

Test fixtures provide realistic conversation data:

```python
sample_conversation = {
    "messages": [
        {"role": "user", "content": "I'm John, a software engineer at TechCorp"},
        {"role": "assistant", "content": "Nice to meet you John!"},
        {"role": "user", "content": "I work on ML infrastructure using Python"}
    ],
    "expected_memories": [
        {"importance": 9, "category": "professional", "content": "Works at TechCorp"},
        {"importance": 8, "category": "technical", "content": "Uses Python for ML"}
    ]
}
```

## Memory Extraction Testing

### Importance Scoring

Tests validate memory importance scoring on a 1-10 scale:

- **1-3**: Trivial (greetings, weather)
- **4-6**: Moderate (preferences, interests)  
- **7-8**: Important (professional info)
- **9-10**: Critical (identity, health, security)

### Category Classification

Tests cover all memory categories:

- `personal` - Basic personal information
- `professional` - Work and career info
- `technical_preferences` - Technology choices
- `food_preferences` - Dietary preferences
- `hobbies` - Recreational activities
- `education` - Learning and academic background
- `location` - Geographic information
- `goals` - Future plans and aspirations
- `health` - Medical and fitness info
- `relationships` - Family and social connections

### Content Filtering

Tests validate sensitive information filtering:

```python
# Sensitive content (should be filtered)
"My social security number is 123-45-6789"
"My password is mySecretPass123" 
"My credit card is 4111-1111-1111-1111"

# Safe content (should be stored)
"I enjoy reading science fiction books"
"My favorite programming language is Python"
"I work in software development"
```

## Error Scenario Testing

### Retry Logic

Tests verify proper retry behavior:

- ✅ Exponential backoff with jitter
- ✅ Different retry strategies for different errors
- ✅ Maximum retry limits
- ✅ Circuit breaker integration

### Provider-Specific Errors

Tests handle provider-specific error responses:

```python
# OpenAI API errors
{"error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}}

# Ollama errors  
{"error": "Model not found"}

# Custom endpoint errors
{"error": "json_object format not supported"}
```

## Streaming Response Testing

### NDJSON Handling

Tests validate streaming response parsing:

```python
# Ollama NDJSON stream
'{"choices": [{"delta": {"role": "assistant"}}]}\n'
'{"choices": [{"delta": {"content": "The"}}]}\n'
'{"choices": [{"delta": {"content": " memory"}}]}\n'
'{"choices": [{"finish_reason": "stop"}]}\n'
```

### Function Calling

Tests validate function call responses:

```python
{
    "choices": [{
        "message": {
            "function_call": {
                "name": "extract_memory",
                "arguments": '{"importance": 8, "category": "technical"}'
            }
        },
        "finish_reason": "function_call"
    }]
}
```

## Real-World Scenario Testing

### Complex Conversations

Tests handle realistic conversation patterns:

- Long multi-turn conversations
- Professional discussions with technical details
- Personal information mixed with preferences
- Incremental information updates
- Context switches and topic changes

### Memory Quality Validation

Tests ensure extracted memories are:

- ✅ Accurate and complete
- ✅ Properly categorized
- ✅ Appropriately scored for importance
- ✅ Free of sensitive information
- ✅ Useful for future interactions

## Debugging and Development

### Test Output

Tests provide detailed logging for debugging:

```bash
# Run with verbose output and no capture
pytest tests/integration/test_llm_integration.py -v -s

# Show test durations
pytest tests/integration/test_llm_integration.py --durations=10

# Stop on first failure
pytest tests/integration/test_llm_integration.py -x
```

### Mock Recording

Enable request/response recording for debugging:

```python
mock.enable_recording = True
# ... run tests ...
recordings = mock.get_recording()
```

### Custom Test Scenarios

Add new test scenarios by extending fixtures:

```python
@pytest.fixture
def custom_scenario():
    return {
        "input": "Custom test input",
        "expected_output": {"importance": 5, "category": "custom"}
    }
```

## Dependencies

Required packages for running tests:

```bash
pip install pytest pytest-asyncio aiohttp
```

Optional packages for enhanced testing:

```bash
pip install pytest-cov pytest-html pytest-xdist
```

## Integration with CI/CD

### GitHub Actions

```yaml
- name: Run LLM Integration Tests
  run: |
    python tests/integration/run_llm_integration_tests.py
```

### Docker Testing

```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "tests/integration/run_llm_integration_tests.py"]
```

## Contributing

When adding new LLM integration tests:

1. Follow the existing test class structure
2. Add appropriate fixtures for test data
3. Include both success and failure scenarios
4. Test with multiple LLM providers when applicable
5. Add documentation for new test categories
6. Ensure tests are isolated and repeatable

## Performance Considerations

- Tests use async/await for efficient execution
- Mock responses avoid actual API calls
- Parallel test execution supported with pytest-xdist
- Memory usage monitored for long-running tests
- Circuit breaker prevents cascade failures

## Security Testing

Tests validate security aspects:

- ✅ API key handling and validation
- ✅ Sensitive information filtering
- ✅ Input sanitization
- ✅ Output validation
- ✅ Error message sanitization

This comprehensive test suite ensures the OWUI Adaptive Memory Plugin's LLM integration is robust, secure, and reliable across different providers and usage scenarios.