"""
Test configuration for OpenWebUI integration tests.

This module provides configuration options and test scenarios
for comprehensive integration testing.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class TestScenarioConfig:
    """Configuration for a test scenario"""
    name: str
    description: str
    api_configs: Dict[str, Any] = field(default_factory=dict)
    error_injection: Dict[str, Any] = field(default_factory=dict)
    performance_limits: Dict[str, float] = field(default_factory=dict)
    data_volumes: Dict[str, int] = field(default_factory=dict)


class IntegrationTestConfig:
    """Main configuration for integration tests"""
    
    # API endpoint configurations
    OPENWEBUI_API_CONFIGS = {
        "v1": {
            "base_url": "http://localhost:8080/api/v1",
            "memory_endpoint": "/memories",
            "headers": {"Content-Type": "application/json"},
            "timeout": 30.0
        },
        "v2": {
            "base_url": "http://localhost:8080/api/v2",
            "memory_endpoint": "/memories",
            "headers": {
                "Content-Type": "application/json",
                "X-API-Version": "2.0"
            },
            "timeout": 30.0
        }
    }
    
    # Test data configurations
    TEST_DATA_SIZES = {
        "small": {
            "num_users": 5,
            "memories_per_user": 10,
            "message_count": 20
        },
        "medium": {
            "num_users": 50,
            "memories_per_user": 100,
            "message_count": 200
        },
        "large": {
            "num_users": 500,
            "memories_per_user": 1000,
            "message_count": 2000
        }
    }
    
    # Error injection configurations
    ERROR_SCENARIOS = {
        "network_issues": {
            "enable_random_disconnects": True,
            "disconnect_rate": 0.1,
            "enable_delays": True,
            "delay_range_ms": (100, 5000),
            "enable_timeouts": True,
            "timeout_rate": 0.05
        },
        "api_errors": {
            "enable_rate_limiting": True,
            "rate_limit_threshold": 100,
            "enable_server_errors": True,
            "server_error_rate": 0.1,
            "enable_invalid_responses": True,
            "invalid_response_rate": 0.05
        },
        "data_corruption": {
            "enable_partial_responses": True,
            "enable_encoding_errors": True,
            "enable_schema_violations": True,
            "corruption_rate": 0.02
        }
    }
    
    # Performance thresholds
    PERFORMANCE_THRESHOLDS = {
        "memory_extraction": {
            "max_duration_ms": 500,
            "max_memory_mb": 50
        },
        "memory_injection": {
            "max_duration_ms": 200,
            "max_memories_injected": 10
        },
        "api_call": {
            "max_duration_ms": 1000,
            "max_retries": 3
        },
        "batch_operation": {
            "max_duration_per_item_ms": 10,
            "max_total_duration_s": 30
        }
    }
    
    # Test scenarios
    SCENARIOS = [
        TestScenarioConfig(
            name="happy_path",
            description="Normal operation with minimal errors",
            api_configs={"version": "v1", "enable_cache": True},
            error_injection={"error_rate": 0.01},
            performance_limits=PERFORMANCE_THRESHOLDS,
            data_volumes=TEST_DATA_SIZES["small"]
        ),
        TestScenarioConfig(
            name="high_load",
            description="High volume concurrent operations",
            api_configs={"version": "v1", "enable_cache": True},
            error_injection={"error_rate": 0.05},
            performance_limits={
                **PERFORMANCE_THRESHOLDS,
                "batch_operation": {
                    "max_duration_per_item_ms": 20,
                    "max_total_duration_s": 60
                }
            },
            data_volumes=TEST_DATA_SIZES["large"]
        ),
        TestScenarioConfig(
            name="unreliable_network",
            description="Simulate unreliable network conditions",
            api_configs={"version": "v1", "enable_cache": False},
            error_injection=ERROR_SCENARIOS["network_issues"],
            performance_limits={
                **PERFORMANCE_THRESHOLDS,
                "api_call": {
                    "max_duration_ms": 5000,
                    "max_retries": 5
                }
            },
            data_volumes=TEST_DATA_SIZES["medium"]
        ),
        TestScenarioConfig(
            name="api_migration",
            description="Test migration between API versions",
            api_configs={"version": "v1_to_v2", "enable_cache": False},
            error_injection={"error_rate": 0.02},
            performance_limits=PERFORMANCE_THRESHOLDS,
            data_volumes=TEST_DATA_SIZES["small"]
        )
    ]
    
    @classmethod
    def get_scenario(cls, name: str) -> TestScenarioConfig:
        """Get a specific test scenario by name"""
        for scenario in cls.SCENARIOS:
            if scenario.name == name:
                return scenario
        raise ValueError(f"Unknown scenario: {name}")
    
    @classmethod
    def get_api_config(cls, version: str) -> Dict[str, Any]:
        """Get API configuration for a specific version"""
        if version not in cls.OPENWEBUI_API_CONFIGS:
            raise ValueError(f"Unknown API version: {version}")
        return cls.OPENWEBUI_API_CONFIGS[version]
    
    @classmethod
    def get_performance_threshold(cls, operation: str) -> Dict[str, Any]:
        """Get performance thresholds for a specific operation"""
        if operation not in cls.PERFORMANCE_THRESHOLDS:
            raise ValueError(f"Unknown operation: {operation}")
        return cls.PERFORMANCE_THRESHOLDS[operation]


# Environment-based configuration
class EnvironmentConfig:
    """Configuration based on environment variables"""
    
    # Test environment
    TEST_ENV = os.getenv("TEST_ENV", "local")
    
    # API endpoints (can be overridden by environment)
    OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "http://localhost:8080")
    OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY", "test_api_key")
    
    # Test data location
    TEST_DATA_DIR = os.getenv("TEST_DATA_DIR", "./tests/integration/test_data")
    
    # Performance settings
    ENABLE_PERFORMANCE_TESTS = os.getenv("ENABLE_PERFORMANCE_TESTS", "true").lower() == "true"
    PERFORMANCE_TEST_ITERATIONS = int(os.getenv("PERFORMANCE_TEST_ITERATIONS", "3"))
    
    # Debugging
    ENABLE_DEBUG_LOGGING = os.getenv("ENABLE_DEBUG_LOGGING", "false").lower() == "true"
    SAVE_API_RECORDINGS = os.getenv("SAVE_API_RECORDINGS", "false").lower() == "true"
    
    # Timeout settings
    DEFAULT_TIMEOUT_SECONDS = int(os.getenv("DEFAULT_TIMEOUT_SECONDS", "30"))
    LONG_RUNNING_TIMEOUT_SECONDS = int(os.getenv("LONG_RUNNING_TIMEOUT_SECONDS", "300"))


# Test data generators configuration
class TestDataConfig:
    """Configuration for test data generation"""
    
    # User data patterns
    USER_NAME_PATTERNS = [
        "test_user_{index}",
        "user_{index}_{timestamp}",
        "{adjective}_{noun}_{index}"
    ]
    
    # Memory content templates
    MEMORY_TEMPLATES = [
        "User prefers {preference} over {alternative}",
        "User has {years} years of experience in {field}",
        "User's favorite {category} is {item}",
        "User is learning {skill} and practices {frequency}",
        "User works as a {job_title} at {company}",
        "User lives in {city} and enjoys {activity}"
    ]
    
    # Conversation templates
    CONVERSATION_STARTERS = [
        "Hi, I need help with {topic}",
        "Can you tell me about {subject}?",
        "I'm working on {project} and need advice",
        "What do you think about {question}?",
        "I've been thinking about {idea}"
    ]
    
    # Metadata categories
    METADATA_CATEGORIES = [
        "preferences",
        "knowledge",
        "experience",
        "personal",
        "professional",
        "goals",
        "interests",
        "skills"
    ]
    
    # Importance distribution
    IMPORTANCE_WEIGHTS = {
        "low": (0.1, 0.3),
        "medium": (0.3, 0.7),
        "high": (0.7, 0.9),
        "critical": (0.9, 1.0)
    }


# Validation configuration
class ValidationConfig:
    """Configuration for test validation"""
    
    # Memory validation rules
    MEMORY_VALIDATION = {
        "min_content_length": 10,
        "max_content_length": 1000,
        "required_fields": ["id", "user_id", "content", "timestamp"],
        "optional_fields": ["metadata", "importance", "context", "embedding"],
        "importance_range": (0.0, 1.0)
    }
    
    # API response validation
    RESPONSE_VALIDATION = {
        "required_status_codes": [200, 201, 204],
        "error_status_codes": [400, 401, 403, 404, 429, 500, 502, 503],
        "max_response_time_ms": 5000,
        "required_headers": ["content-type"],
        "json_schema_validation": True
    }
    
    # Performance validation
    PERFORMANCE_VALIDATION = {
        "max_memory_growth_mb": 100,
        "max_cpu_usage_percent": 80,
        "max_concurrent_operations": 100,
        "min_throughput_ops_per_second": 10
    }


# Export all configurations
__all__ = [
    "IntegrationTestConfig",
    "EnvironmentConfig",
    "TestDataConfig",
    "ValidationConfig",
    "TestScenarioConfig"
]