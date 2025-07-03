"""
Pytest configuration and fixtures for OWUI Adaptive Memory Plugin testing.

This module provides shared fixtures and configuration for all test modules.
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the main module with proper mocking for missing dependencies
try:
    from adaptive_memory_v4_0 import Filter, Valves, UserValves
except ImportError:
    # Fallback import method with dependency mocking
    import importlib.util
    
    # Mock missing dependencies before importing
    import types
    
    # Mock pydantic first (many things depend on it)
    if 'pydantic' not in sys.modules:
        pydantic_mock = types.ModuleType('pydantic')
        
        # Create a proper BaseModel mock
        class MockBaseModel:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
            
            def dict(self):
                return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
            
            def json(self):
                import json
                return json.dumps(self.dict())
        
        pydantic_mock.BaseModel = MockBaseModel
        pydantic_mock.Field = lambda **kwargs: None
        pydantic_mock.create_model = lambda name, **fields: type(name, (MockBaseModel,), {})
        pydantic_mock.validator = lambda *args, **kwargs: lambda f: f
        pydantic_mock.model_validator = lambda *args, **kwargs: lambda f: f
        pydantic_mock.field_validator = lambda *args, **kwargs: lambda f: f
        pydantic_mock.root_validator = lambda *args, **kwargs: lambda f: f
        pydantic_mock.ValidationError = Exception
        sys.modules['pydantic'] = pydantic_mock
    
    # Mock sentence_transformers
    sentence_transformers_mock = types.ModuleType('sentence_transformers')
    sentence_transformers_mock.SentenceTransformer = Mock
    sys.modules['sentence_transformers'] = sentence_transformers_mock
    
    # Mock httpx
    httpx_mock = types.ModuleType('httpx')
    httpx_mock.AsyncClient = AsyncMock
    sys.modules['httpx'] = httpx_mock
    
    # Mock numpy
    numpy_mock = types.ModuleType('numpy')
    numpy_mock.array = list
    numpy_mock.dot = lambda a, b: sum(x*y for x,y in zip(a,b))
    numpy_linalg = types.ModuleType('linalg')
    numpy_linalg.norm = lambda x: sum(i**2 for i in x)**0.5
    numpy_mock.linalg = numpy_linalg
    sys.modules['numpy'] = numpy_mock
    
    # Mock fastapi if needed
    if 'fastapi' not in sys.modules:
        fastapi_mock = types.ModuleType('fastapi')
        fastapi_mock.Response = Mock
        fastapi_mock.FastAPI = Mock
        sys.modules['fastapi'] = fastapi_mock
    
    # Mock pytz if needed
    if 'pytz' not in sys.modules:
        pytz_mock = types.ModuleType('pytz')
        pytz_mock.timezone = lambda x: timezone.utc
        sys.modules['pytz'] = pytz_mock
    
    # Mock open_webui
    if 'open_webui' not in sys.modules:
        open_webui_mock = types.ModuleType('open_webui')
        routers_mock = types.ModuleType('routers')
        memories_mock = types.ModuleType('memories')
        
        # Mock the memory functions
        memories_mock.create_memory = Mock(return_value={"id": "test_memory"})
        memories_mock.get_memories = Mock(return_value=[])
        memories_mock.update_memory_by_id = Mock(return_value={"id": "test_memory"})
        memories_mock.delete_memory_by_id = Mock(return_value=True)
        memories_mock.add_memory = Mock(return_value={"id": "test_memory"})
        memories_mock.query_memory = Mock(return_value=[])
        memories_mock.AddMemoryForm = Mock
        memories_mock.QueryMemoryForm = Mock
        memories_mock.Memories = Mock
        
        routers_mock.memories = memories_mock
        open_webui_mock.routers = routers_mock
        
        # Mock users module
        models_mock = types.ModuleType('models')
        users_mock = types.ModuleType('users')
        users_mock.Users = Mock
        models_mock.users = users_mock
        open_webui_mock.models = models_mock
        
        # Mock main app
        main_mock = types.ModuleType('main')
        main_mock.app = Mock()
        open_webui_mock.main = main_mock
        
        sys.modules['open_webui'] = open_webui_mock
        sys.modules['open_webui.routers'] = routers_mock
        sys.modules['open_webui.routers.memories'] = memories_mock
        sys.modules['open_webui.models'] = models_mock
        sys.modules['open_webui.models.users'] = users_mock
        sys.modules['open_webui.main'] = main_mock
    
    spec = importlib.util.spec_from_file_location(
        "adaptive_memory_v4_0", 
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "adaptive_memory_v4.0.py")
    )
    adaptive_memory_v4_0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adaptive_memory_v4_0)
    Filter = adaptive_memory_v4_0.Filter
    Valves = getattr(adaptive_memory_v4_0, 'Valves', type('MockValves', (), {}))
    UserValves = getattr(adaptive_memory_v4_0, 'UserValves', type('MockUserValves', (), {}))


@pytest.fixture
def sample_user_id():
    """Generate a sample user ID for testing."""
    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_chat_id():
    """Generate a sample chat ID for testing."""
    return f"test_chat_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_message_id():
    """Generate a sample message ID for testing."""
    return f"test_msg_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def basic_message_body(sample_user_id, sample_chat_id, sample_message_id):
    """Create a basic OpenWebUI message body for testing."""
    return {
        "messages": [
            {
                "id": sample_message_id,
                "role": "user",
                "content": "Hello, this is a test message.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ],
        "chat_id": sample_chat_id,
        "user": {
            "id": sample_user_id,
            "name": "Test User",
            "email": "test@example.com"
        },
        "model": "test-model",
        "stream": False
    }


@pytest.fixture
def complex_message_body(sample_user_id, sample_chat_id):
    """Create a complex OpenWebUI message body with multiple messages."""
    return {
        "messages": [
            {
                "id": "msg1",
                "role": "user",
                "content": "What is machine learning?",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "msg2",
                "role": "assistant",
                "content": "Machine learning is a subset of artificial intelligence...",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "msg3",
                "role": "user",
                "content": "Can you give me an example?",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ],
        "chat_id": sample_chat_id,
        "user": {
            "id": sample_user_id,
            "name": "Test User",
            "email": "test@example.com"
        },
        "model": "gpt-4",
        "stream": False
    }


@pytest.fixture
def default_valves():
    """Create default valves configuration for testing."""
    # Try to create a real Valves instance if available
    if hasattr(adaptive_memory_v4_0, 'Filter') and hasattr(adaptive_memory_v4_0.Filter, 'Valves'):
        # Get the actual Valves class from the Filter
        ValvesClass = adaptive_memory_v4_0.Filter.Valves
        # Create instance with defaults
        return ValvesClass()
    else:
        # Fallback to a mock with required attributes
        class MockValves:
            # Essential attributes that the Filter expects
            enable_filter_orchestration = True
            filter_execution_timeout_ms = 10000
            enable_conflict_detection = True
            enable_performance_monitoring = True
            filter_priority = "normal"
            enable_rollback_mechanism = True
            max_concurrent_filters = 5
            coordination_overhead_threshold_ms = 100.0
            enable_shared_state = False
            filter_isolation_level = "partial"
            
            # Memory settings
            memory_extraction_enabled = True
            memory_injection_enabled = True
            memory_relevance_scoring_with_llm_enabled = False
            memory_relevance_score_threshold = 0.7
            memory_max_count = 20
            
            # Other common attributes
            priority = 100
            embedding_model_name = "all-MiniLM-L6-v2"
            embedding_provider_type = "local"
            llm_provider_type = "ollama"
            
            def dict(self):
                return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
                
        return MockValves()


@pytest.fixture
def default_user_valves(sample_user_id):
    """Create default user valves configuration for testing."""
    return UserValves()


@pytest.fixture
def filter_instance(default_valves):
    """Create a Filter instance with default configuration."""
    # Mock the async task creation to avoid event loop issues
    with patch('asyncio.create_task') as mock_create_task:
        mock_create_task.return_value = Mock()
        
        # Create filter instance
        filter_obj = Filter()
        
        # Ensure valves is properly set
        if hasattr(filter_obj, 'valves'):
            # Update with our test valves if it already exists
            for key, value in default_valves.__dict__.items():
                if not key.startswith('_'):
                    setattr(filter_obj.valves, key, value)
        else:
            # Set our valves if not already set
            filter_obj.valves = default_valves
        
        # Initialize some attributes that might be missing
        filter_obj._error_log_task = None
        filter_obj._api_version_info = {}
        filter_obj._filter_metadata = None
        filter_obj.orchestration_context = None
        
        # Mock embedding model to avoid loading real models
        filter_obj._embedding_model = Mock()
        filter_obj._embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock the orchestration manager if it exists
        if hasattr(adaptive_memory_v4_0, '_orchestration_manager'):
            adaptive_memory_v4_0._orchestration_manager.register_filter = Mock(return_value=True)
        
        return filter_obj


@pytest.fixture
def mock_openwebui_env():
    """Mock OpenWebUI environment variables and context."""
    with patch.dict(os.environ, {
        'OPENWEBUI_HOST': 'http://localhost:8080',
        'OPENWEBUI_API_KEY': 'test-api-key',
        'OPENWEBUI_USER_ID': 'test-user',
    }):
        yield


@pytest.fixture
def mock_sentence_transformers():
    """Mock sentence transformers for testing without requiring the dependency."""
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3] for _ in range(5)]
        mock_st.return_value = mock_model
        yield mock_st


@pytest.fixture
def mock_embedding_response():
    """Mock embedding API response."""
    return {
        "data": [
            {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "index": 0
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 10
        }
    }


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for API calls."""
    with patch('httpx.AsyncClient') as mock_client:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        yield mock_client


@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    return {
        "memories": [
            {
                "id": "mem1",
                "content": "User prefers concise explanations",
                "importance": 0.8,
                "recency": 0.9,
                "context": "conversation_style",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "mem2",
                "content": "User is interested in machine learning",
                "importance": 0.7,
                "recency": 0.8,
                "context": "interests",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
    }


@pytest.fixture
def empty_body():
    """Empty request body for testing edge cases."""
    return {}


@pytest.fixture
def invalid_body():
    """Invalid request body for testing error handling."""
    return {
        "invalid_field": "invalid_value",
        "malformed_data": None
    }


@pytest.fixture
def stream_event():
    """Sample stream event for testing stream method."""
    return {
        "type": "message",
        "data": {
            "content": "This is a streaming message",
            "role": "assistant"
        }
    }


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_logger():
    """Mock logger for testing logging functionality."""
    with patch('logging.getLogger') as mock_get_logger:
        mock_log = Mock()
        mock_get_logger.return_value = mock_log
        yield mock_log


# Test data fixtures
@pytest.fixture
def test_memory_operations():
    """Test data for memory operations."""
    return [
        {
            "operation": "store",
            "content": "User likes detailed explanations",
            "importance": 0.8
        },
        {
            "operation": "retrieve",
            "query": "user preferences",
            "limit": 5
        },
        {
            "operation": "update",
            "memory_id": "mem1",
            "content": "Updated memory content"
        },
        {
            "operation": "delete",
            "memory_id": "mem2"
        }
    ]


# Performance testing fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        "max_execution_time": 1.0,  # seconds
        "memory_limit": 100,  # MB
        "concurrent_requests": 10
    }


# Integration testing fixtures
@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "openwebui_mock_port": 8081,
        "test_timeout": 30,
        "retry_attempts": 3
    }