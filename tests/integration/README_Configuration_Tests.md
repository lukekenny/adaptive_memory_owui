# Configuration Integration Tests

This directory contains comprehensive integration tests for configuration handling, error scenarios, and system resilience in the OWUI Adaptive Memory Plugin.

## Overview

The configuration integration tests validate:

1. **Configuration Management** - Valves validation, persistence, recovery
2. **Error Handling** - Network failures, API errors, authentication issues
3. **System Resilience** - Graceful degradation, circuit breakers, recovery
4. **Stress Testing** - Concurrency, memory pressure, long-running stability

## Test Structure

### Core Test Files

- `test_configuration_integration.py` - Main configuration and resilience tests
- `test_valves_configuration.py` - Specific Valves class testing
- `config_test_settings.py` - Test configuration and utilities
- `run_configuration_tests.py` - Test runner script

### Test Classes

#### TestConfigurationIntegration
Tests configuration validation, persistence, and recovery:
- Valid/invalid configuration validation
- Configuration persistence across instances
- Recovery from corrupted configuration
- Environment variable loading
- Default value fallbacks

#### TestErrorScenarios  
Tests error handling and recovery mechanisms:
- Network failure recovery
- API endpoint failures
- Authentication error handling
- JSON parsing error recovery
- Resource exhaustion handling
- Concurrent operation failures

#### TestSystemResilience
Tests system resilience and circuit breaker functionality:
- Graceful degradation when services fail
- Circuit breaker-like behavior
- Automatic recovery mechanisms
- Health check functionality
- Error logging and monitoring

#### TestStressAndConcurrency
Tests stress scenarios and concurrent operations:
- High concurrency memory operations
- Memory pressure handling
- Rapid configuration changes
- Long-running stability

#### TestValvesValidation
Tests Valves field validation and constraints:
- Provider type validation
- Threshold value constraints  
- Positive integer constraints
- Boolean field validation
- Optional string fields
- URL field validation

## Running Tests

### Quick Start

```bash
# Run all configuration tests
python run_configuration_tests.py

# Run specific test suite
python run_configuration_tests.py --suite config
python run_configuration_tests.py --suite error
python run_configuration_tests.py --suite resilience
python run_configuration_tests.py --suite valves

# Run with stress tests (takes longer)
python run_configuration_tests.py --suite stress --stress

# Generate reports
python run_configuration_tests.py --report --coverage
```

### Test Suites

- `config` - Configuration validation and management
- `error` - Error handling and recovery scenarios
- `resilience` - System resilience and circuit breakers
- `stress` - Stress testing and concurrency
- `valves` - Valves-specific configuration tests
- `all` - All test suites (default)

### Options

- `--stress` - Include resource-intensive stress tests
- `--verbose` - Enable verbose test output
- `--report` - Generate HTML test report
- `--coverage` - Generate code coverage report
- `--quick` - Run quick tests only (skip slow tests)

### Using pytest directly

```bash
# Run specific test file
pytest test_configuration_integration.py -v

# Run specific test class
pytest test_configuration_integration.py::TestConfigurationIntegration -v

# Run with markers
pytest -m "not stress" -v  # Skip stress tests
pytest -m "stress" -v      # Only stress tests

# Run with coverage
pytest --cov=adaptive_memory_v4 --cov-report=html test_configuration_integration.py
```

## Test Configuration

### Environment Variables

Tests respect these environment variables for configuration:

- `OWUI_LLM_API_KEY` - API key for LLM testing
- `OWUI_EMBEDDING_API_KEY` - API key for embedding testing  
- `OWUI_MAX_MEMORIES` - Memory limit override
- `OWUI_RELEVANCE_THRESHOLD` - Relevance threshold override

### Test Settings

Key test configuration in `config_test_settings.py`:

```python
TEST_CONFIG = {
    "DEFAULT_TIMEOUT": 30.0,
    "STRESS_TEST_TIMEOUT": 120.0,
    "MAX_CONCURRENT_OPERATIONS": 50,
    "MAX_RESPONSE_TIME": 10.0,
    "MIN_SUCCESS_RATE": 0.8,
    # ... more settings
}
```

## Test Scenarios

### Configuration Scenarios

1. **Valid Configurations**
   - Ollama + local embeddings
   - OpenAI compatible setup
   - Google Gemini configuration
   - High performance config
   - Strict quality config

2. **Invalid Configurations**
   - Invalid provider types
   - Out-of-range thresholds
   - Negative integer values
   - Missing API keys
   - Malformed URLs

### Error Scenarios

1. **Network Errors**
   - Connection failures
   - Timeout errors
   - DNS resolution failures
   - SSL certificate errors

2. **API Errors**
   - Server errors (500, 502, 503, 504)
   - Client errors (400, 404)
   - Authentication errors (401, 403)
   - Rate limiting (429)

3. **Data Errors**
   - JSON parsing failures
   - Malformed responses
   - Empty responses
   - Encoding errors

### Resilience Scenarios

1. **Graceful Degradation**
   - LLM service failures → fallback to regex parsing
   - Embedding service failures → text similarity fallback
   - API unavailable → cached responses

2. **Circuit Breaker Patterns**
   - Failure threshold detection
   - Service recovery detection
   - Automatic retry with backoff

3. **Resource Management**
   - Memory pressure handling
   - Connection pool management
   - Task cancellation and cleanup

## Test Data and Mocks

### Mock Services

Tests use comprehensive mocks for external dependencies:

- `LLMAPIMock` - Mock LLM API responses
- `EmbeddingAPIMock` - Mock embedding API responses  
- `OpenWebUIMemoryAPIMock` - Mock OpenWebUI memory API

### Test Data Generation

- `generate_test_user()` - Generate test user data
- `generate_test_message()` - Generate test message data
- `generate_test_memory()` - Generate test memory data
- `ConfigurationFactory` - Generate test configurations

### Error Injection

Tests support controlled error injection:

```python
# Simulate network failures
error_simulation = {
    "type": "network",
    "errors": ["connection_error", "timeout"],
    "frequency": 0.5
}

# Simulate API errors
error_simulation = {
    "type": "api", 
    "status_codes": [500, 502, 503],
    "frequency": 0.3
}
```

## Performance Testing

### Concurrency Tests

- Up to 50 concurrent operations
- Measures success rate and response times
- Validates thread safety

### Memory Pressure Tests

- Large message processing
- Memory usage monitoring  
- Leak detection
- Garbage collection verification

### Stability Tests

- Long-running operations (5+ minutes)
- Resource cleanup verification
- Performance degradation detection

## Reporting and Analysis

### Test Reports

HTML test reports include:
- Test execution summary
- Pass/fail status per test
- Execution times
- Error details and stack traces

### Coverage Reports

Code coverage reports show:
- Line coverage percentages
- Branch coverage analysis
- Uncovered code sections
- Coverage trends

### Performance Metrics

Tests collect performance metrics:
- Response times
- Memory usage
- Concurrency throughput
- Error rates

## Common Issues and Troubleshooting

### Test Failures

1. **Configuration Validation Errors**
   - Check Pydantic model constraints
   - Verify field types and ranges
   - Review default values

2. **Network/API Errors**
   - Verify mock configuration
   - Check timeout settings
   - Review error simulation parameters

3. **Resource Exhaustion**
   - Increase test timeouts
   - Reduce concurrency levels
   - Check system resources

### Performance Issues

1. **Slow Test Execution**
   - Use `--quick` flag
   - Skip stress tests with `-m "not stress"`
   - Reduce test data size

2. **Memory Issues**
   - Monitor system memory
   - Increase swap space
   - Use smaller test datasets

### CI/CD Integration

Tests are designed for CI/CD environments:

```yaml
# Example GitHub Actions
- name: Run Configuration Tests
  run: |
    python tests/integration/run_configuration_tests.py --quick --report
    
- name: Run Stress Tests  
  run: |
    python tests/integration/run_configuration_tests.py --suite stress --stress
  timeout-minutes: 30
```

## Contributing

### Adding New Tests

1. **Configuration Tests** - Add to `TestConfigurationIntegration`
2. **Error Scenarios** - Add to `TestErrorScenarios`  
3. **Resilience Tests** - Add to `TestSystemResilience`
4. **Stress Tests** - Add to `TestStressAndConcurrency`

### Test Naming Conventions

- Test methods: `test_<functionality>_<scenario>`
- Test classes: `Test<ComponentName><TestType>`
- Test files: `test_<component>_<test_type>.py`

### Mock Guidelines

- Use existing mock factories when possible
- Add new mocks to `mocks/` directory
- Document mock behavior and limitations
- Ensure mocks match real API behavior

## Dependencies

Required packages for configuration tests:

```txt
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-html>=3.1.0
aiohttp>=3.8.0
psutil>=5.9.0
pydantic>=2.0.0
```

Install with:
```bash
pip install -r requirements-test.txt
```

## See Also

- [Integration Tests Overview](README.md)
- [LLM Integration Tests](README_LLM_Integration.md)
- [E2E Tests Documentation](README_E2E_Tests.md)
- [Filter Documentation](../../FILTER_ORCHESTRATION_SYSTEM.md)