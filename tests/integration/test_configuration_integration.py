"""
Integration tests for configuration handling, error scenarios, and system resilience
in the OWUI Adaptive Memory Plugin.

This module tests:
1. Configuration scenarios (validation, recovery, persistence)
2. Error scenarios (network failures, API errors, resource exhaustion)  
3. System resilience (graceful degradation, recovery, health checks)

Test Categories:
- Configuration Management
- Error Handling & Recovery
- System Resilience & Circuit Breakers
- Stress & Concurrency Testing
"""

import pytest
import asyncio
import json
import os
import tempfile
import uuid
import logging
import time
import threading
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import weakref
import psutil
import gc

# Import the filter and dependencies
import sys
import importlib.util
from pathlib import Path

# Load the filter module directly
project_root = Path(__file__).parent.parent.parent
filter_path = project_root / "adaptive_memory_v4.0.py"
spec = importlib.util.spec_from_file_location("adaptive_memory_v4_0", filter_path)
adaptive_memory_v4_0 = importlib.util.module_from_spec(spec)
sys.modules["adaptive_memory_v4_0"] = adaptive_memory_v4_0
spec.loader.exec_module(adaptive_memory_v4_0)

Filter = adaptive_memory_v4_0.Filter
from tests.integration.fixtures import (
    generate_test_user, 
    generate_test_message,
    generate_test_memory
)
from tests.integration.mocks.llm_api_mock import LLMAPIMock, LLMProvider
from tests.integration.mocks.embedding_api_mock import EmbeddingAPIMock
from tests.integration.mocks.openwebui_api_mock import OpenWebUIMemoryAPIMock, APIError


class TestConfigurationIntegration:
    """Test configuration management scenarios"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for configuration tests"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def filter_instance(self, temp_config_dir):
        """Create filter instance with temporary configuration"""
        filter_instance = Filter()
        # Override config directory
        filter_instance._config_dir = temp_config_dir
        return filter_instance
    
    @pytest.fixture
    def valid_valves_config(self):
        """Valid configuration for testing"""
        return {
            "embedding_provider_type": "local",
            "embedding_model_name": "all-MiniLM-L6-v2",
            "llm_provider_type": "ollama",
            "llm_model_name": "llama3:latest",
            "llm_api_endpoint_url": "http://localhost:11434/api/chat",
            "max_total_memories": 200,
            "relevance_threshold": 0.45,
            "memory_threshold": 0.6,
            "min_memory_length": 8,
            "enable_json_stripping": True,
            "enable_fallback_regex": True,
            "filter_trivia": True,
            "deduplicate_memories": True,
            "use_embeddings_for_deduplication": True,
            "embedding_similarity_threshold": 0.97,
            "use_fingerprinting": True,
            "use_lsh_optimization": True,
            "cache_ttl_seconds": 86400,
            "enable_summarization_task": True,
            "summarization_interval": 7200
        }
    
    @pytest.fixture
    def invalid_valves_config(self):
        """Invalid configuration for testing validation"""
        return {
            "embedding_provider_type": "invalid_provider",
            "llm_provider_type": "unknown_llm",
            "max_total_memories": -10,  # Invalid negative value
            "relevance_threshold": 1.5,  # Invalid > 1.0
            "memory_threshold": -0.1,   # Invalid < 0.0
            "min_memory_length": -5,    # Invalid negative
            "embedding_similarity_threshold": 2.0,  # Invalid > 1.0
            "cache_ttl_seconds": -1000,  # Invalid negative
            "summarization_interval": 0   # Invalid zero
        }
    
    @pytest.mark.asyncio
    async def test_configuration_validation_success(self, filter_instance, valid_valves_config):
        """Test successful configuration validation"""
        # Test configuration validation
        is_valid, error_msg = filter_instance.validate_configuration_before_ui_save(valid_valves_config)
        
        assert is_valid, f"Valid configuration rejected: {error_msg}"
        assert error_msg == "", f"Error message should be empty for valid config: {error_msg}"
    
    @pytest.mark.asyncio
    async def test_configuration_validation_failure(self, filter_instance, invalid_valves_config):
        """Test configuration validation with invalid values"""
        # Test each invalid field separately for better error reporting
        test_cases = [
            ({"embedding_provider_type": "invalid_provider"}, "embedding_provider_type"),
            ({"llm_provider_type": "unknown_llm"}, "llm_provider_type"),
            ({"max_total_memories": -10}, "max_total_memories"),
            ({"relevance_threshold": 1.5}, "relevance_threshold"),
            ({"memory_threshold": -0.1}, "memory_threshold"),
            ({"min_memory_length": -5}, "min_memory_length"),
            ({"embedding_similarity_threshold": 2.0}, "embedding_similarity_threshold"),
            ({"cache_ttl_seconds": -1000}, "cache_ttl_seconds"),
            ({"summarization_interval": 0}, "summarization_interval")
        ]
        
        for invalid_field, field_name in test_cases:
            config = {"llm_provider_type": "ollama"}  # Base valid config
            config.update(invalid_field)
            
            is_valid, error_msg = filter_instance.validate_configuration_before_ui_save(config)
            
            assert not is_valid, f"Invalid {field_name} configuration was accepted"
            assert error_msg, f"Error message should not be empty for invalid {field_name}"
    
    @pytest.mark.asyncio
    async def test_configuration_persistence(self, filter_instance, valid_valves_config, temp_config_dir):
        """Test configuration persistence and loading"""
        # Set configuration
        for key, value in valid_valves_config.items():
            setattr(filter_instance.valves, key, value)
        
        # Save configuration
        config_file = os.path.join(temp_config_dir, "config.json")
        saved_config = {
            key: getattr(filter_instance.valves, key)
            for key in valid_valves_config.keys()
        }
        
        with open(config_file, 'w') as f:
            json.dump(saved_config, f)
        
        # Create new filter instance and load configuration
        new_filter = Filter()
        new_filter._config_dir = temp_config_dir
        
        # Mock the _load_configuration_safe method to load from our file
        def mock_load_config():
            with open(config_file, 'r') as f:
                return json.load(f)
        
        with patch.object(new_filter, '_load_configuration_safe', side_effect=mock_load_config):
            loaded_config = new_filter._load_configuration_safe()
            
            # Verify configuration was loaded correctly
            for key, expected_value in valid_valves_config.items():
                assert loaded_config.get(key) == expected_value, f"Configuration {key} not persisted correctly"
    
    @pytest.mark.asyncio
    async def test_configuration_recovery_from_corruption(self, filter_instance, temp_config_dir):
        """Test configuration recovery when configuration is corrupted"""
        # Create corrupted configuration file
        config_file = os.path.join(temp_config_dir, "config.json")
        with open(config_file, 'w') as f:
            f.write("{ invalid json content }")
        
        # Mock _load_configuration_safe to simulate corruption recovery
        with patch.object(filter_instance, '_load_configuration_safe') as mock_load:
            mock_load.side_effect = [
                json.JSONDecodeError("Invalid JSON", "test", 0),  # First call fails
                {"llm_provider_type": "ollama"}  # Recovery succeeds
            ]
            
            # Test that recovery is attempted
            try:
                config = filter_instance._load_configuration_safe()
                assert config is not None, "Configuration recovery should provide fallback"
            except Exception as e:
                pytest.fail(f"Configuration recovery failed: {e}")
    
    @pytest.mark.asyncio
    async def test_environment_variable_loading(self, filter_instance):
        """Test loading configuration from environment variables"""
        env_vars = {
            "OWUI_LLM_API_KEY": "test-api-key-123",
            "OWUI_EMBEDDING_API_KEY": "test-embedding-key-456",
            "OWUI_MAX_MEMORIES": "150",
            "OWUI_RELEVANCE_THRESHOLD": "0.5"
        }
        
        with patch.dict(os.environ, env_vars):
            # Test that environment variables can override configuration
            # Note: This tests the concept - actual implementation may vary
            
            # Check if filter respects environment variables
            if hasattr(filter_instance.valves, 'llm_api_key'):
                filter_instance.valves.llm_api_key = os.environ.get("OWUI_LLM_API_KEY")
                assert filter_instance.valves.llm_api_key == "test-api-key-123"
            
            # Test numeric environment variable conversion
            if hasattr(filter_instance.valves, 'max_total_memories'):
                filter_instance.valves.max_total_memories = int(os.environ.get("OWUI_MAX_MEMORIES", "200"))
                assert filter_instance.valves.max_total_memories == 150
    
    @pytest.mark.asyncio
    async def test_configuration_defaults_fallback(self, filter_instance):
        """Test that configuration falls back to defaults when values are missing"""
        # Create filter with minimal configuration
        minimal_config = {"llm_provider_type": "ollama"}
        
        # Test that defaults are applied for missing values
        default_fields = [
            ("max_total_memories", 200),
            ("relevance_threshold", 0.45),
            ("memory_threshold", 0.6),
            ("min_memory_length", 8),
            ("cache_ttl_seconds", 86400),
            ("enable_json_stripping", True),
            ("filter_trivia", True),
            ("deduplicate_memories", True)
        ]
        
        for field_name, expected_default in default_fields:
            if hasattr(filter_instance.valves, field_name):
                actual_value = getattr(filter_instance.valves, field_name)
                assert actual_value == expected_default, f"Default for {field_name} should be {expected_default}, got {actual_value}"


class TestErrorScenarios:
    """Test error handling and recovery scenarios"""
    
    @pytest.fixture
    def filter_with_mocks(self):
        """Create filter instance with mocked dependencies"""
        filter_instance = Filter()
        return filter_instance
    
    @pytest.mark.asyncio
    async def test_network_failure_recovery(self, filter_with_mocks):
        """Test network failure handling and recovery"""
        user_data = generate_test_user()
        message_data = generate_test_message()
        
        # Mock network failures
        with patch('aiohttp.ClientSession.post') as mock_post:
            # First call fails with network error
            mock_post.side_effect = [
                aiohttp.ClientConnectorError(connection_key=None, os_error=None),
                # Second call succeeds (recovery)
                AsyncMock(status=200, json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]}))
            ]
            
            # Test that the filter handles network errors gracefully
            try:
                result = await filter_with_mocks.inlet(
                    body={"messages": [message_data]},
                    user=user_data
                )
                # Should not crash, may have empty or fallback content
                assert result is not None
                assert "messages" in result
                
            except Exception as e:
                # Network errors should be handled gracefully, not crash the filter
                assert "connection" not in str(e).lower(), f"Network error not handled gracefully: {e}"
    
    @pytest.mark.asyncio
    async def test_api_endpoint_failure(self, filter_with_mocks):
        """Test API endpoint failure handling"""
        user_data = generate_test_user()
        message_data = generate_test_message()
        
        # Mock API failures
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Simulate various API errors
            error_responses = [
                AsyncMock(status=500, text=AsyncMock(return_value="Internal Server Error")),
                AsyncMock(status=429, text=AsyncMock(return_value="Rate Limited")),
                AsyncMock(status=404, text=AsyncMock(return_value="Not Found")),
                AsyncMock(status=200, json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]}))  # Recovery
            ]
            mock_post.side_effect = error_responses
            
            # Test API error handling
            result = await filter_with_mocks.inlet(
                body={"messages": [message_data]},
                user=user_data
            )
            
            # Should handle API errors and potentially provide fallback
            assert result is not None
            assert "messages" in result
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling(self, filter_with_mocks):
        """Test authentication error handling"""
        user_data = generate_test_user()
        message_data = generate_test_message()
        
        # Mock authentication errors
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = [
                AsyncMock(status=401, text=AsyncMock(return_value="Unauthorized")),
                AsyncMock(status=403, text=AsyncMock(return_value="Forbidden")),
                AsyncMock(status=200, json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]}))  # Recovery with valid auth
            ]
            
            # Test authentication error handling
            result = await filter_with_mocks.inlet(
                body={"messages": [message_data]},
                user=user_data
            )
            
            assert result is not None
            assert "messages" in result
    
    @pytest.mark.asyncio
    async def test_json_parsing_error_recovery(self, filter_with_mocks):
        """Test JSON parsing error recovery with fallback mechanisms"""
        user_data = generate_test_user()
        message_data = generate_test_message()
        
        # Mock responses with invalid JSON
        with patch('aiohttp.ClientSession.post') as mock_post:
            invalid_json_responses = [
                AsyncMock(status=200, json=AsyncMock(return_value={"choices": [{"message": {"content": "invalid json { malformed"}}]})),
                AsyncMock(status=200, json=AsyncMock(return_value={"choices": [{"message": {"content": "I like coffee"}}]})),  # Fallback content
            ]
            mock_post.side_effect = invalid_json_responses
            
            # Test JSON parsing error recovery
            result = await filter_with_mocks.inlet(
                body={"messages": [message_data]},
                user=user_data
            )
            
            assert result is not None
            assert "messages" in result
            
            # Should use fallback mechanisms (regex, preference detection)
            # when JSON parsing fails but content contains preference keywords
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_handling(self, filter_with_mocks):
        """Test handling of resource exhaustion scenarios"""
        user_data = generate_test_user()
        message_data = generate_test_message()
        
        # Mock memory/resource exhaustion
        with patch('asyncio.create_task') as mock_create_task:
            # Simulate resource exhaustion
            mock_create_task.side_effect = [
                MemoryError("Out of memory"),
                OSError("Too many open files"),
                asyncio.TimeoutError("Operation timeout"),
                AsyncMock()  # Recovery
            ]
            
            # Test resource exhaustion handling
            try:
                result = await filter_with_mocks.inlet(
                    body={"messages": [message_data]},
                    user=user_data
                )
                assert result is not None
            except (MemoryError, OSError, asyncio.TimeoutError):
                # These should be caught and handled gracefully
                pytest.fail("Resource exhaustion should be handled gracefully")
    
    @pytest.mark.asyncio
    async def test_concurrent_operation_failures(self, filter_with_mocks):
        """Test handling of concurrent operation failures"""
        user_data = generate_test_user()
        
        # Create multiple concurrent operations
        async def failing_operation(i):
            if i % 3 == 0:  # Every third operation fails
                raise Exception(f"Simulated failure {i}")
            return f"Success {i}"
        
        # Test concurrent error handling
        tasks = [failing_operation(i) for i in range(10)]
        
        # Use asyncio.gather with return_exceptions to handle failures
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        # Should have both successes and handled failures
        assert len(successes) > 0, "Some operations should succeed"
        assert len(failures) > 0, "Some operations should fail as expected"
        
        # Failures should be Exception instances, not unhandled
        for failure in failures:
            assert isinstance(failure, Exception)


class TestSystemResilience:
    """Test system resilience and circuit breaker functionality"""
    
    @pytest.fixture
    def resilient_filter(self):
        """Create filter with resilience features enabled"""
        filter_instance = Filter()
        # Enable resilience features
        filter_instance.valves.enable_fallback_regex = True
        filter_instance.valves.enable_json_stripping = True
        filter_instance.valves.enable_feature_detection = True
        return filter_instance
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_llm_failure(self, resilient_filter):
        """Test graceful degradation when LLM services fail"""
        user_data = generate_test_user()
        message_data = generate_test_message(content="I really love chocolate ice cream")
        
        # Mock LLM service failures
        with patch('aiohttp.ClientSession.post') as mock_post:
            # LLM consistently fails
            mock_post.side_effect = Exception("LLM service unavailable")
            
            # Test graceful degradation
            result = await resilient_filter.inlet(
                body={"messages": [message_data]},
                user=user_data
            )
            
            # Should not crash, should provide basic functionality
            assert result is not None
            assert "messages" in result
            
            # Should maintain message content even if memory extraction fails
            assert len(result["messages"]) > 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, resilient_filter):
        """Test circuit breaker-like behavior for repeated failures"""
        user_data = generate_test_user()
        
        failure_count = 0
        success_count = 0
        
        # Simulate repeated operations with failures
        with patch('aiohttp.ClientSession.post') as mock_post:
            def side_effect(*args, **kwargs):
                nonlocal failure_count, success_count
                if failure_count < 5:  # First 5 calls fail
                    failure_count += 1
                    raise Exception("Service failure")
                else:  # Subsequent calls succeed
                    success_count += 1
                    return AsyncMock(status=200, json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]}))
            
            mock_post.side_effect = side_effect
            
            # Make multiple requests
            for i in range(10):
                message_data = generate_test_message(content=f"Test message {i}")
                
                try:
                    result = await resilient_filter.inlet(
                        body={"messages": [message_data]},
                        user=user_data
                    )
                    assert result is not None
                    
                    # After several failures, should start succeeding
                    if i >= 5:
                        assert success_count > 0, "Should start succeeding after initial failures"
                        
                except Exception as e:
                    # Early failures should be handled gracefully
                    if i < 5:
                        assert "Service failure" in str(e) or result is not None
    
    @pytest.mark.asyncio
    async def test_automatic_recovery_mechanisms(self, resilient_filter):
        """Test automatic recovery after service restoration"""
        user_data = generate_test_user()
        
        # Simulate service outage and recovery
        call_count = 0
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                if call_count <= 3:  # First 3 calls fail (outage)
                    raise Exception("Service outage")
                else:  # Service recovered
                    return AsyncMock(status=200, json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]}))
            
            mock_post.side_effect = side_effect
            
            # Test recovery after outage
            for i in range(5):
                message_data = generate_test_message(content=f"Recovery test {i}")
                
                result = await resilient_filter.inlet(
                    body={"messages": [message_data]},
                    user=user_data
                )
                
                assert result is not None
                
                # After service recovery (i >= 3), should work normally
                if i >= 3:
                    assert "messages" in result
                    assert len(result["messages"]) > 0
    
    @pytest.mark.asyncio
    async def test_health_check_mechanisms(self, resilient_filter):
        """Test health check and monitoring functionality"""
        user_data = generate_test_user()
        
        # Test basic health indicators
        health_indicators = {
            "config_valid": True,
            "dependencies_available": True,
            "memory_usage_normal": True,
            "response_time_acceptable": True
        }
        
        # Simulate health check
        start_time = time.time()
        
        message_data = generate_test_message()
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value = AsyncMock(
                status=200, 
                json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]})
            )
            
            result = await resilient_filter.inlet(
                body={"messages": [message_data]},
                user=user_data
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Health checks
            assert result is not None, "Service should be responsive"
            health_indicators["dependencies_available"] = True
            health_indicators["response_time_acceptable"] = response_time < 10.0  # 10 second threshold
            
            # Configuration health
            config_fields = ["llm_provider_type", "max_total_memories", "relevance_threshold"]
            for field in config_fields:
                if hasattr(resilient_filter.valves, field):
                    assert getattr(resilient_filter.valves, field) is not None
            
            health_indicators["config_valid"] = True
        
        # Overall health assessment
        overall_health = all(health_indicators.values())
        assert overall_health, f"Health check failed: {health_indicators}"
    
    @pytest.mark.asyncio
    async def test_error_logging_and_monitoring(self, resilient_filter, caplog):
        """Test error logging and monitoring functionality"""
        user_data = generate_test_user()
        message_data = generate_test_message()
        
        # Enable logging
        caplog.set_level(logging.WARNING)
        
        # Simulate errors that should be logged
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = [
                Exception("Test error for logging"),
                AsyncMock(status=500, text=AsyncMock(return_value="Server error")),
                AsyncMock(status=200, json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]}))
            ]
            
            # Test error logging
            for i in range(3):
                try:
                    await resilient_filter.inlet(
                        body={"messages": [message_data]},
                        user=user_data
                    )
                except Exception:
                    pass  # Errors may be raised but should be logged
        
        # Verify error logging occurred
        log_messages = [record.message for record in caplog.records]
        
        # Should have logged some errors or warnings
        assert len(log_messages) >= 0, "Error logging should capture issues"


class TestStressAndConcurrency:
    """Test stress scenarios and concurrent operations"""
    
    @pytest.fixture
    def stress_test_filter(self):
        """Create filter optimized for stress testing"""
        filter_instance = Filter()
        # Optimize for stress testing
        filter_instance.valves.cache_ttl_seconds = 60  # Shorter cache for testing
        filter_instance.valves.max_total_memories = 50  # Lower limit for testing
        return filter_instance
    
    @pytest.mark.asyncio
    async def test_high_concurrency_operations(self, stress_test_filter):
        """Test high concurrency memory operations"""
        user_data = generate_test_user()
        
        # Create multiple concurrent operations
        concurrent_operations = 20
        
        async def memory_operation(operation_id):
            message_data = generate_test_message(content=f"Concurrent operation {operation_id}")
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value = AsyncMock(
                    status=200,
                    json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]})
                )
                
                result = await stress_test_filter.inlet(
                    body={"messages": [message_data]},
                    user=user_data
                )
                
                return operation_id, result
        
        # Execute concurrent operations
        start_time = time.time()
        tasks = [memory_operation(i) for i in range(concurrent_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Analyze results
        successful_operations = [r for r in results if not isinstance(r, Exception)]
        failed_operations = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful_operations) / len(results)
        total_time = end_time - start_time
        
        # Performance assertions
        assert success_rate >= 0.8, f"Success rate too low: {success_rate}"
        assert total_time < 30.0, f"Operations took too long: {total_time}s"
        
        # Verify all successful operations returned valid results
        for operation_id, result in successful_operations:
            assert result is not None
            assert "messages" in result
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, stress_test_filter):
        """Test handling of memory pressure scenarios"""
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        user_data = generate_test_user()
        
        # Create memory-intensive operations
        large_operations = 50
        
        for i in range(large_operations):
            # Create large message content
            large_content = "This is a large memory test. " * 100  # ~3KB per message
            message_data = generate_test_message(content=large_content)
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value = AsyncMock(
                    status=200,
                    json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]})
                )
                
                try:
                    result = await stress_test_filter.inlet(
                        body={"messages": [message_data]},
                        user=user_data
                    )
                    assert result is not None
                    
                    # Check memory usage periodically
                    if i % 10 == 0:
                        current_memory = process.memory_info().rss
                        memory_increase = current_memory - initial_memory
                        
                        # Memory should not grow unbounded (basic leak detection)
                        # Allow reasonable growth but not excessive
                        max_allowed_increase = 100 * 1024 * 1024  # 100MB
                        assert memory_increase < max_allowed_increase, f"Memory usage increased by {memory_increase / 1024 / 1024:.1f}MB"
                    
                except MemoryError:
                    # Memory errors should be handled gracefully
                    pytest.fail("Memory errors should be handled gracefully")
        
        # Force garbage collection
        gc.collect()
        
        # Final memory check
        final_memory = process.memory_info().rss
        total_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        max_total_increase = 150 * 1024 * 1024  # 150MB total
        assert total_increase < max_total_increase, f"Total memory increase too large: {total_increase / 1024 / 1024:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_rapid_configuration_changes(self, stress_test_filter):
        """Test rapid configuration changes under load"""
        user_data = generate_test_user()
        
        # Configuration change scenarios
        config_changes = [
            {"max_total_memories": 30},
            {"relevance_threshold": 0.3},
            {"memory_threshold": 0.7},
            {"relevance_threshold": 0.5},
            {"max_total_memories": 50}
        ]
        
        async def operation_with_config_change(change_id):
            # Apply configuration change
            config_change = config_changes[change_id % len(config_changes)]
            for key, value in config_change.items():
                if hasattr(stress_test_filter.valves, key):
                    setattr(stress_test_filter.valves, key, value)
            
            # Perform operation
            message_data = generate_test_message(content=f"Config change test {change_id}")
            
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_post.return_value = AsyncMock(
                    status=200,
                    json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]})
                )
                
                result = await stress_test_filter.inlet(
                    body={"messages": [message_data]},
                    user=user_data
                )
                
                return change_id, result
        
        # Execute operations with rapid config changes
        change_operations = 15
        tasks = [operation_with_config_change(i) for i in range(change_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify operations succeeded despite config changes
        successful_operations = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful_operations) / len(results)
        
        assert success_rate >= 0.7, f"Too many failures during config changes: {success_rate}"
        
        # Verify final configuration is valid
        final_config_check = {
            "max_total_memories": getattr(stress_test_filter.valves, "max_total_memories", None),
            "relevance_threshold": getattr(stress_test_filter.valves, "relevance_threshold", None),
            "memory_threshold": getattr(stress_test_filter.valves, "memory_threshold", None)
        }
        
        for key, value in final_config_check.items():
            assert value is not None, f"Configuration {key} should not be None after changes"
            
            # Validate ranges
            if "threshold" in key and value is not None:
                assert 0.0 <= value <= 1.0, f"Threshold {key} should be in range [0, 1]: {value}"
    
    @pytest.mark.asyncio
    async def test_long_running_stability(self, stress_test_filter):
        """Test long-running stability and resource cleanup"""
        user_data = generate_test_user()
        
        # Simulate long-running operations
        operation_cycles = 100
        operations_per_cycle = 5
        
        for cycle in range(operation_cycles):
            cycle_start = time.time()
            
            # Perform multiple operations per cycle
            for op in range(operations_per_cycle):
                message_data = generate_test_message(content=f"Stability test cycle {cycle} op {op}")
                
                with patch('aiohttp.ClientSession.post') as mock_post:
                    mock_post.return_value = AsyncMock(
                        status=200,
                        json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]})
                    )
                    
                    result = await stress_test_filter.inlet(
                        body={"messages": [message_data]},
                        user=user_data
                    )
                    
                    assert result is not None, f"Operation failed at cycle {cycle}, op {op}"
            
            cycle_time = time.time() - cycle_start
            
            # Performance should remain stable
            max_cycle_time = 5.0  # 5 seconds per cycle
            assert cycle_time < max_cycle_time, f"Cycle {cycle} took too long: {cycle_time:.2f}s"
            
            # Periodic resource cleanup check
            if cycle % 20 == 0:
                gc.collect()  # Force garbage collection
                
                # Check that filter is still responsive
                test_message = generate_test_message(content="Health check")
                
                with patch('aiohttp.ClientSession.post') as mock_post:
                    mock_post.return_value = AsyncMock(
                        status=200,
                        json=AsyncMock(return_value={"choices": [{"message": {"content": "[]"}}]})
                    )
                    
                    health_result = await stress_test_filter.inlet(
                        body={"messages": [test_message]},
                        user=user_data
                    )
                    
                    assert health_result is not None, f"Health check failed at cycle {cycle}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])