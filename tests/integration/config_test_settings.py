"""
Configuration settings and utilities for configuration integration tests.

This module provides test configuration, mock factories, and helper utilities
specifically for testing configuration handling, error scenarios, and system resilience.
"""

import os
import json
import tempfile
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

# Test configuration constants
TEST_CONFIG = {
    # Test timeouts
    "DEFAULT_TIMEOUT": 30.0,
    "STRESS_TEST_TIMEOUT": 120.0,
    "NETWORK_TIMEOUT": 5.0,
    
    # Test data limits
    "MAX_CONCURRENT_OPERATIONS": 50,
    "STRESS_TEST_CYCLES": 100,
    "MEMORY_PRESSURE_OPERATIONS": 50,
    
    # Performance thresholds
    "MAX_RESPONSE_TIME": 10.0,
    "MAX_MEMORY_INCREASE_MB": 150,
    "MIN_SUCCESS_RATE": 0.8,
    
    # Configuration test values
    "TEST_LLM_ENDPOINT": "http://localhost:11434/api/chat",
    "TEST_EMBEDDING_MODEL": "all-MiniLM-L6-v2",
    "TEST_API_KEY": "test-api-key-12345",
    
    # Error simulation settings
    "ERROR_INJECTION_RATE": 0.3,
    "NETWORK_ERROR_TYPES": ["connection", "timeout", "dns", "ssl"],
    "API_ERROR_CODES": [400, 401, 403, 404, 429, 500, 502, 503, 504],
}


@dataclass
class TestScenario:
    """Test scenario configuration"""
    name: str
    description: str
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    error_simulation: Optional[Dict[str, Any]] = None
    expected_behavior: str = "success"
    timeout: float = TEST_CONFIG["DEFAULT_TIMEOUT"]
    tags: List[str] = field(default_factory=list)


class ConfigurationTestScenarios:
    """Predefined test scenarios for configuration testing"""
    
    @staticmethod
    def get_valid_configuration_scenarios() -> List[TestScenario]:
        """Get test scenarios for valid configurations"""
        return [
            TestScenario(
                name="ollama_local_embedding",
                description="Ollama LLM with local embeddings",
                config_overrides={
                    "llm_provider_type": "ollama",
                    "llm_api_endpoint_url": "http://localhost:11434/api/chat",
                    "embedding_provider_type": "local",
                    "embedding_model_name": "all-MiniLM-L6-v2"
                },
                tags=["valid", "local", "ollama"]
            ),
            TestScenario(
                name="openai_compatible_setup",
                description="OpenAI compatible setup with API key",
                config_overrides={
                    "llm_provider_type": "openai_compatible",
                    "llm_api_endpoint_url": "https://api.openai.com/v1/chat/completions",
                    "llm_api_key": TEST_CONFIG["TEST_API_KEY"],
                    "embedding_provider_type": "openai_compatible",
                    "embedding_api_url": "https://api.openai.com/v1/embeddings",
                    "embedding_api_key": TEST_CONFIG["TEST_API_KEY"]
                },
                tags=["valid", "api", "openai"]
            ),
            TestScenario(
                name="gemini_configuration",
                description="Google Gemini configuration",
                config_overrides={
                    "llm_provider_type": "gemini",
                    "llm_api_endpoint_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                    "llm_api_key": TEST_CONFIG["TEST_API_KEY"]
                },
                tags=["valid", "api", "gemini"]
            ),
            TestScenario(
                name="high_performance_config",
                description="Configuration optimized for high performance",
                config_overrides={
                    "max_total_memories": 500,
                    "cache_ttl_seconds": 7200,
                    "use_llm_for_relevance": False,
                    "use_lsh_optimization": True,
                    "use_fingerprinting": True,
                    "top_n_memories": 5
                },
                tags=["valid", "performance", "optimization"]
            ),
            TestScenario(
                name="strict_quality_config",
                description="Configuration with strict quality thresholds",
                config_overrides={
                    "relevance_threshold": 0.8,
                    "memory_threshold": 0.9,
                    "min_confidence_threshold": 0.7,
                    "embedding_similarity_threshold": 0.95,
                    "filter_trivia": True,
                    "deduplicate_memories": True
                },
                tags=["valid", "quality", "strict"]
            )
        ]
    
    @staticmethod
    def get_invalid_configuration_scenarios() -> List[TestScenario]:
        """Get test scenarios for invalid configurations"""
        return [
            TestScenario(
                name="invalid_provider_types",
                description="Invalid provider type values",
                config_overrides={
                    "llm_provider_type": "invalid_llm",
                    "embedding_provider_type": "invalid_embedding"
                },
                expected_behavior="validation_error",
                tags=["invalid", "provider"]
            ),
            TestScenario(
                name="invalid_threshold_ranges",
                description="Threshold values outside valid ranges",
                config_overrides={
                    "relevance_threshold": 1.5,
                    "memory_threshold": -0.1,
                    "embedding_similarity_threshold": 2.0
                },
                expected_behavior="validation_error",
                tags=["invalid", "threshold"]
            ),
            TestScenario(
                name="invalid_integer_values",
                description="Invalid integer field values",
                config_overrides={
                    "max_total_memories": -10,
                    "min_memory_length": -5,
                    "cache_ttl_seconds": -1000
                },
                expected_behavior="validation_error",
                tags=["invalid", "integer"]
            ),
            TestScenario(
                name="missing_required_api_keys",
                description="API provider without required API key",
                config_overrides={
                    "llm_provider_type": "openai_compatible",
                    "llm_api_key": None,
                    "embedding_provider_type": "openai_compatible",
                    "embedding_api_key": None
                },
                expected_behavior="runtime_error",
                tags=["invalid", "auth"]
            ),
            TestScenario(
                name="invalid_url_formats",
                description="Invalid URL formats",
                config_overrides={
                    "llm_api_endpoint_url": "not-a-valid-url",
                    "embedding_api_url": "invalid://malformed"
                },
                expected_behavior="validation_error",
                tags=["invalid", "url"]
            )
        ]
    
    @staticmethod
    def get_error_simulation_scenarios() -> List[TestScenario]:
        """Get test scenarios for error simulation"""
        return [
            TestScenario(
                name="network_connection_failure",
                description="Simulate network connection failures",
                error_simulation={
                    "type": "network",
                    "errors": ["connection_error", "timeout", "dns_failure"],
                    "frequency": 0.5
                },
                expected_behavior="graceful_degradation",
                tags=["error", "network"]
            ),
            TestScenario(
                name="api_server_errors",
                description="Simulate API server error responses",
                error_simulation={
                    "type": "api",
                    "status_codes": [500, 502, 503, 504],
                    "frequency": 0.3
                },
                expected_behavior="retry_with_fallback",
                tags=["error", "api"]
            ),
            TestScenario(
                name="authentication_failures",
                description="Simulate authentication and authorization errors",
                error_simulation={
                    "type": "auth",
                    "status_codes": [401, 403],
                    "frequency": 0.8
                },
                expected_behavior="auth_error_handling",
                tags=["error", "auth"]
            ),
            TestScenario(
                name="rate_limiting",
                description="Simulate rate limiting responses",
                error_simulation={
                    "type": "rate_limit",
                    "status_codes": [429],
                    "frequency": 0.4,
                    "retry_after": 5
                },
                expected_behavior="backoff_retry",
                tags=["error", "rate_limit"]
            ),
            TestScenario(
                name="json_parsing_errors",
                description="Simulate malformed JSON responses",
                error_simulation={
                    "type": "json",
                    "responses": [
                        "{ invalid json",
                        "incomplete json {",
                        "null",
                        ""
                    ],
                    "frequency": 0.6
                },
                expected_behavior="fallback_parsing",
                tags=["error", "parsing"]
            ),
            TestScenario(
                name="resource_exhaustion",
                description="Simulate resource exhaustion scenarios",
                error_simulation={
                    "type": "resource",
                    "errors": ["memory_error", "file_descriptor_limit", "timeout"],
                    "frequency": 0.2
                },
                expected_behavior="resource_management",
                tags=["error", "resource"]
            )
        ]
    
    @staticmethod
    def get_stress_test_scenarios() -> List[TestScenario]:
        """Get test scenarios for stress testing"""
        return [
            TestScenario(
                name="high_concurrency_load",
                description="High concurrency memory operations",
                config_overrides={
                    "max_total_memories": 100,
                    "cache_ttl_seconds": 300
                },
                timeout=TEST_CONFIG["STRESS_TEST_TIMEOUT"],
                tags=["stress", "concurrency"]
            ),
            TestScenario(
                name="memory_pressure_test",
                description="Test under memory pressure conditions",
                config_overrides={
                    "max_total_memories": 1000,
                    "use_lsh_optimization": False  # Force more memory usage
                },
                timeout=TEST_CONFIG["STRESS_TEST_TIMEOUT"],
                tags=["stress", "memory"]
            ),
            TestScenario(
                name="rapid_config_changes",
                description="Rapid configuration changes under load",
                config_overrides={},
                timeout=TEST_CONFIG["STRESS_TEST_TIMEOUT"],
                tags=["stress", "config"]
            ),
            TestScenario(
                name="long_running_stability",
                description="Long-running stability test",
                config_overrides={
                    "cache_ttl_seconds": 60,  # Shorter cache for testing
                    "summarization_interval": 300  # More frequent summarization
                },
                timeout=300.0,  # 5 minutes
                tags=["stress", "stability"]
            )
        ]


class ConfigurationFactory:
    """Factory for creating test configurations"""
    
    @staticmethod
    def create_minimal_config() -> Dict[str, Any]:
        """Create minimal valid configuration"""
        return {
            "llm_provider_type": "ollama",
            "embedding_provider_type": "local"
        }
    
    @staticmethod
    def create_full_config() -> Dict[str, Any]:
        """Create full configuration with all fields"""
        return {
            # Provider configuration
            "embedding_provider_type": "local",
            "embedding_model_name": "all-MiniLM-L6-v2",
            "embedding_api_url": None,
            "embedding_api_key": None,
            "llm_provider_type": "ollama",
            "llm_model_name": "llama3:latest",
            "llm_api_endpoint_url": "http://localhost:11434/api/chat",
            "llm_api_key": None,
            
            # Memory management
            "max_total_memories": 200,
            "min_memory_length": 8,
            "pruning_strategy": "fifo",
            "min_confidence_threshold": 0.5,
            
            # Thresholds
            "relevance_threshold": 0.45,
            "memory_threshold": 0.6,
            "vector_similarity_threshold": 0.45,
            "llm_skip_relevance_threshold": 0.93,
            "embedding_similarity_threshold": 0.97,
            "similarity_threshold": 0.95,
            "fingerprint_similarity_threshold": 0.8,
            
            # Processing settings
            "recent_messages_n": 5,
            "related_memories_n": 5,
            "top_n_memories": 3,
            "cache_ttl_seconds": 86400,
            "max_injected_memory_length": 300,
            
            # Feature flags
            "enable_json_stripping": True,
            "enable_fallback_regex": True,
            "enable_short_preference_shortcut": True,
            "enable_feature_detection": True,
            "filter_trivia": True,
            "deduplicate_memories": True,
            "use_embeddings_for_deduplication": True,
            "use_fingerprinting": True,
            "use_lsh_optimization": True,
            "use_enhanced_confidence_scoring": True,
            "use_llm_for_relevance": False,
            
            # Background tasks
            "enable_summarization_task": True,
            "summarization_interval": 7200,
            "enable_error_logging_task": True,
            "error_logging_interval": 1800,
            "enable_date_update_task": True,
            "date_update_interval": 3600,
            "enable_model_discovery_task": True,
            "model_discovery_interval": 7200,
            
            # Summarization
            "summarization_min_cluster_size": 3,
            "summarization_similarity_threshold": 0.7,
            "summarization_max_cluster_size": 8,
            "summarization_min_memory_age_days": 7,
            "summarization_strategy": "hybrid",
            
            # Advanced features
            "fingerprint_num_hashes": 128,
            "fingerprint_shingle_size": 3,
            "lsh_threshold_for_activation": 100,
            "short_preference_no_dedupe_length": 100,
            
            # Optional fields
            "blacklist_topics": None,
            "whitelist_keywords": None,
            "preference_keywords_no_dedupe": "favorite,love,like,prefer,enjoy"
        }
    
    @staticmethod
    def create_performance_config() -> Dict[str, Any]:
        """Create performance-optimized configuration"""
        config = ConfigurationFactory.create_full_config()
        config.update({
            "max_total_memories": 500,
            "cache_ttl_seconds": 7200,
            "use_llm_for_relevance": False,
            "use_lsh_optimization": True,
            "use_fingerprinting": True,
            "top_n_memories": 5,
            "summarization_interval": 3600
        })
        return config
    
    @staticmethod
    def create_quality_config() -> Dict[str, Any]:
        """Create quality-focused configuration"""
        config = ConfigurationFactory.create_full_config()
        config.update({
            "relevance_threshold": 0.8,
            "memory_threshold": 0.9,
            "min_confidence_threshold": 0.7,
            "embedding_similarity_threshold": 0.98,
            "filter_trivia": True,
            "deduplicate_memories": True,
            "use_llm_for_relevance": True
        })
        return config
    
    @staticmethod
    def create_test_config_file(config: Dict[str, Any], temp_dir: Optional[str] = None) -> str:
        """Create temporary configuration file for testing"""
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        config_file = os.path.join(temp_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_file


class TestDataGenerator:
    """Generator for test data and scenarios"""
    
    @staticmethod
    def generate_configuration_mutations(base_config: Dict[str, Any], num_mutations: int = 10) -> List[Dict[str, Any]]:
        """Generate configuration mutations for testing"""
        mutations = []
        
        # Common mutation patterns
        mutation_patterns = [
            # Threshold adjustments
            lambda c: {**c, "relevance_threshold": max(0.0, min(1.0, c.get("relevance_threshold", 0.5) + 0.1))},
            lambda c: {**c, "memory_threshold": max(0.0, min(1.0, c.get("memory_threshold", 0.6) + 0.1))},
            
            # Memory limit adjustments
            lambda c: {**c, "max_total_memories": max(10, c.get("max_total_memories", 200) + 50)},
            lambda c: {**c, "max_total_memories": max(10, c.get("max_total_memories", 200) - 50)},
            
            # Feature flag toggles
            lambda c: {**c, "filter_trivia": not c.get("filter_trivia", True)},
            lambda c: {**c, "deduplicate_memories": not c.get("deduplicate_memories", True)},
            lambda c: {**c, "use_llm_for_relevance": not c.get("use_llm_for_relevance", False)},
            
            # Performance adjustments
            lambda c: {**c, "cache_ttl_seconds": max(60, c.get("cache_ttl_seconds", 86400) // 2)},
            lambda c: {**c, "top_n_memories": max(1, min(10, c.get("top_n_memories", 3) + 1))},
            
            # Provider switches
            lambda c: {**c, "llm_provider_type": "openai_compatible" if c.get("llm_provider_type") == "ollama" else "ollama"},
        ]
        
        for i in range(min(num_mutations, len(mutation_patterns))):
            pattern = mutation_patterns[i % len(mutation_patterns)]
            mutations.append(pattern(base_config))
        
        return mutations
    
    @staticmethod
    def generate_error_conditions() -> List[Dict[str, Any]]:
        """Generate error condition specifications"""
        return [
            {
                "name": "network_timeout",
                "type": "network",
                "simulation": "timeout",
                "duration": 5.0
            },
            {
                "name": "api_server_error", 
                "type": "api",
                "status_code": 500,
                "response": "Internal Server Error"
            },
            {
                "name": "authentication_error",
                "type": "auth",
                "status_code": 401,
                "response": "Unauthorized"
            },
            {
                "name": "rate_limit_error",
                "type": "rate_limit",
                "status_code": 429,
                "response": "Too Many Requests",
                "retry_after": 60
            },
            {
                "name": "json_parse_error",
                "type": "parsing",
                "response": "{ invalid json content"
            },
            {
                "name": "resource_exhaustion",
                "type": "resource",
                "error_type": "MemoryError",
                "message": "Out of memory"
            }
        ]


# Export test configuration for easy access
__all__ = [
    'TEST_CONFIG',
    'TestScenario', 
    'ConfigurationTestScenarios',
    'ConfigurationFactory',
    'TestDataGenerator'
]