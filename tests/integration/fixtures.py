"""
Integration test fixtures for OWUI Adaptive Memory Plugin.

This module provides reusable fixtures for integration testing,
including mock servers, test data, and helper utilities.
"""

import pytest
import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from unittest.mock import patch, Mock, AsyncMock
import tempfile
import uuid

# Import mock servers
from .mocks.openwebui_api_mock import (
    OpenWebUIMemoryAPIMock, 
    MockOpenWebUIClient,
    APIError
)
from .mocks.llm_api_mock import (
    LLMAPIMock,
    OpenAIAPIMock,
    OllamaAPIMock,
    AnthropicAPIMock,
    MockLLMClient,
    LLMProvider
)
from .mocks.embedding_api_mock import (
    EmbeddingAPIMock,
    LocalEmbeddingModelMock,
    MockEmbeddingClient,
    EmbeddingProvider
)
from .mocks.websocket_mock import (
    WebSocketServerMock,
    MockWebSocketClient,
    EventType
)


# Test data generators
def generate_test_user(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Generate test user data"""
    user_id = user_id or f"test_user_{uuid.uuid4().hex[:8]}"
    return {
        "id": user_id,
        "name": f"Test User {user_id[-4:]}",
        "email": f"{user_id}@test.example.com",
        "role": "user",
        "created_at": datetime.now(timezone.utc).isoformat()
    }


def generate_test_chat(chat_id: Optional[str] = None, 
                      user_id: Optional[str] = None) -> Dict[str, Any]:
    """Generate test chat data"""
    chat_id = chat_id or f"test_chat_{uuid.uuid4().hex[:8]}"
    user_id = user_id or f"test_user_{uuid.uuid4().hex[:8]}"
    
    return {
        "id": chat_id,
        "user_id": user_id,
        "title": f"Test Chat {chat_id[-4:]}",
        "model": "gpt-3.5-turbo",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }


def generate_test_messages(count: int = 5, 
                         chat_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate test message data"""
    chat_id = chat_id or f"test_chat_{uuid.uuid4().hex[:8]}"
    messages = []
    
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "role": role,
            "content": f"Test message {i+1} from {role}",
            "chat_id": chat_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    return messages


def generate_test_memories(count: int = 10, 
                         user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Generate test memory data"""
    user_id = user_id or f"test_user_{uuid.uuid4().hex[:8]}"
    memories = []
    
    contexts = ["preferences", "knowledge", "experience", "relationships", "goals"]
    
    for i in range(count):
        memories.append({
            "id": f"mem_{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
            "content": f"Test memory {i+1}: User information about {contexts[i % len(contexts)]}",
            "importance": 0.5 + (i % 5) * 0.1,
            "context": contexts[i % len(contexts)],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "source": "test",
                "confidence": 0.8,
                "tags": [f"tag_{i}", "test"]
            }
        })
    
    return memories


# Mock server fixtures
@pytest.fixture
async def openwebui_memory_api_mock():
    """Create OpenWebUI Memory API mock"""
    mock = OpenWebUIMemoryAPIMock(
        enable_rate_limiting=False,
        enable_random_errors=False,
        response_delay_ms=10
    )
    
    # Load some test data
    test_memories = generate_test_memories(5, "default_user")
    mock.load_test_data(test_memories)
    
    yield mock
    
    # Cleanup
    mock.reset()


@pytest.fixture
async def openwebui_memory_api_mock_with_errors():
    """Create OpenWebUI Memory API mock with errors enabled"""
    mock = OpenWebUIMemoryAPIMock(
        enable_rate_limiting=True,
        enable_random_errors=True,
        error_rate=0.3,
        response_delay_ms=50
    )
    
    yield mock
    
    # Cleanup
    mock.reset()


@pytest.fixture
async def openwebui_client(openwebui_memory_api_mock):
    """Create mock OpenWebUI client"""
    client = MockOpenWebUIClient(openwebui_memory_api_mock)
    return client


@pytest.fixture
async def openai_api_mock():
    """Create OpenAI API mock"""
    mock = OpenAIAPIMock(
        enable_streaming=True,
        enable_rate_limiting=False,
        enable_random_errors=False,
        response_delay_ms=10
    )
    
    # Set up some response patterns
    mock.set_response_pattern("hello", "Hello! How can I help you today?")
    mock.set_response_pattern("memory", "I'll help you with memory-related tasks.")
    
    yield mock
    
    # Cleanup
    mock.reset()


@pytest.fixture
async def ollama_api_mock():
    """Create Ollama API mock"""
    mock = OllamaAPIMock(
        enable_streaming=True,
        enable_rate_limiting=False,
        enable_random_errors=False,
        response_delay_ms=5
    )
    
    yield mock
    
    # Cleanup
    mock.reset()


@pytest.fixture
async def anthropic_api_mock():
    """Create Anthropic API mock"""
    mock = AnthropicAPIMock(
        enable_streaming=True,
        enable_rate_limiting=False,
        enable_random_errors=False,
        response_delay_ms=10
    )
    
    yield mock
    
    # Cleanup
    mock.reset()


@pytest.fixture
async def llm_client(request, openai_api_mock, ollama_api_mock, anthropic_api_mock):
    """Create LLM client based on provider parameter"""
    provider = getattr(request, "param", LLMProvider.OPENAI)
    
    if provider == LLMProvider.OPENAI:
        client = MockLLMClient(provider, openai_api_mock)
    elif provider == LLMProvider.OLLAMA:
        client = MockLLMClient(provider, ollama_api_mock)
    elif provider == LLMProvider.ANTHROPIC:
        client = MockLLMClient(provider, anthropic_api_mock)
    else:
        client = MockLLMClient(LLMProvider.CUSTOM, openai_api_mock)
    
    return client


@pytest.fixture
async def embedding_api_mock():
    """Create embedding API mock"""
    mock = EmbeddingAPIMock(
        default_model="text-embedding-ada-002",
        enable_rate_limiting=False,
        enable_random_errors=False,
        response_delay_ms=5,
        deterministic=True
    )
    
    # Set up some custom embeddings for testing
    mock.set_custom_embedding("test query", [0.1, 0.2, 0.3] * 512)  # 1536 dims
    mock.set_custom_embedding("similar text", [0.1, 0.2, 0.35] * 512)
    mock.set_custom_embedding("different text", [0.9, 0.1, 0.1] * 512)
    
    yield mock
    
    # Cleanup
    mock.reset()


@pytest.fixture
async def local_embedding_model():
    """Create local embedding model mock"""
    model = LocalEmbeddingModelMock(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        enable_errors=False
    )
    
    return model


@pytest.fixture
async def embedding_client(embedding_api_mock):
    """Create embedding client"""
    client = MockEmbeddingClient(EmbeddingProvider.OPENAI, embedding_api_mock)
    return client


@pytest.fixture
async def websocket_server():
    """Create WebSocket server mock"""
    server = WebSocketServerMock(
        enable_auto_ping=True,
        ping_interval=30,
        enable_random_disconnects=False,
        enable_message_delays=False,
        message_delay_ms=0
    )
    
    await server.start()
    
    yield server
    
    # Cleanup
    await server.stop()
    server.reset()


@pytest.fixture
async def websocket_server_with_issues():
    """Create WebSocket server mock with network issues"""
    server = WebSocketServerMock(
        enable_auto_ping=True,
        ping_interval=10,
        enable_random_disconnects=True,
        disconnect_rate=0.2,
        enable_message_delays=True,
        message_delay_ms=100
    )
    
    await server.start()
    
    yield server
    
    # Cleanup
    await server.stop()
    server.reset()


@pytest.fixture
async def websocket_client(websocket_server):
    """Create WebSocket client"""
    client = MockWebSocketClient(websocket_server, "test_user")
    
    # Connect automatically
    await client.connect()
    
    yield client
    
    # Cleanup
    if client.is_connected:
        await client.disconnect()


# HTTP client mocking fixtures
@pytest.fixture
def mock_httpx_client(openwebui_client, llm_client, embedding_client):
    """Mock httpx.AsyncClient for all API calls"""
    with patch('httpx.AsyncClient') as mock_class:
        # Create a mock instance that routes to appropriate mocks
        mock_instance = AsyncMock()
        
        async def route_request(method: str, url: str, **kwargs):
            """Route requests to appropriate mock based on URL"""
            if "memories" in url or "openwebui" in url:
                # Route to OpenWebUI mock
                if method == "post":
                    return await openwebui_client.post(url, **kwargs)
                elif method == "get":
                    return await openwebui_client.get(url, **kwargs)
                elif method == "put":
                    return await openwebui_client.put(url, **kwargs)
                elif method == "delete":
                    return await openwebui_client.delete(url, **kwargs)
            
            elif "chat/completions" in url or "v1/chat" in url:
                # Route to LLM mock
                return await llm_client.post(url, **kwargs)
            
            elif "embeddings" in url or "embed" in url:
                # Route to embedding mock
                return await embedding_client.post(url, **kwargs)
            
            else:
                # Default response
                response = Mock()
                response.status_code = 404
                response.json = Mock(return_value={"error": "Not found"})
                return response
        
        # Set up routing
        mock_instance.post = AsyncMock(side_effect=lambda url, **kwargs: route_request("post", url, **kwargs))
        mock_instance.get = AsyncMock(side_effect=lambda url, **kwargs: route_request("get", url, **kwargs))
        mock_instance.put = AsyncMock(side_effect=lambda url, **kwargs: route_request("put", url, **kwargs))
        mock_instance.delete = AsyncMock(side_effect=lambda url, **kwargs: route_request("delete", url, **kwargs))
        
        # Make the context manager work
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        
        mock_class.return_value = mock_instance
        
        yield mock_instance


# Temporary directory fixtures
@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create subdirectories
        os.makedirs(os.path.join(temp_dir, "memories"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(temp_dir, "logs"), exist_ok=True)
        
        yield temp_dir


# Recording fixtures
@pytest.fixture
def request_recorder(openwebui_memory_api_mock, 
                    openai_api_mock,
                    embedding_api_mock):
    """Enable request/response recording on all mocks"""
    # Enable recording
    openwebui_memory_api_mock.enable_recording = True
    openai_api_mock.enable_recording = True
    embedding_api_mock.enable_recording = True
    
    yield {
        "openwebui": openwebui_memory_api_mock,
        "llm": openai_api_mock,
        "embedding": embedding_api_mock
    }
    
    # Get recordings
    recordings = {
        "openwebui": openwebui_memory_api_mock.get_recording(),
        "llm": openai_api_mock.recorded_requests,
        "embedding": embedding_api_mock.get_recording()
    }
    
    # Save to file if test passed
    if hasattr(pytest, "_test_passed") and pytest._test_passed:
        with open("test_recordings.json", "w") as f:
            json.dump(recordings, f, indent=2)


# Scenario fixtures
@pytest.fixture
async def memory_search_scenario(openwebui_memory_api_mock):
    """Set up a memory search scenario"""
    user_id = "scenario_user"
    
    # Create diverse memories
    memories = [
        {
            "user_id": user_id,
            "content": "User prefers Python over JavaScript for backend development",
            "metadata": {"category": "preferences", "topic": "programming"}
        },
        {
            "user_id": user_id,
            "content": "User has 5 years of experience with machine learning",
            "metadata": {"category": "experience", "topic": "ml"}
        },
        {
            "user_id": user_id,
            "content": "User's favorite color is blue",
            "metadata": {"category": "personal", "topic": "preferences"}
        },
        {
            "user_id": user_id,
            "content": "User works at a tech startup in San Francisco",
            "metadata": {"category": "professional", "topic": "work"}
        },
        {
            "user_id": user_id,
            "content": "User is interested in learning Rust programming language",
            "metadata": {"category": "goals", "topic": "learning"}
        }
    ]
    
    # Load memories
    for mem_data in memories:
        await openwebui_memory_api_mock.create_memory(**mem_data)
    
    return {
        "user_id": user_id,
        "memories": memories,
        "api_mock": openwebui_memory_api_mock
    }


@pytest.fixture
async def conversation_scenario(websocket_server):
    """Set up a conversation scenario with WebSocket"""
    user_id = "conv_user"
    chat_id = "conv_chat"
    
    # Create connection
    conn_id = await websocket_server.create_connection(user_id)
    
    # Simulate conversation
    messages = [
        ("user", "Hello, I'm learning about machine learning"),
        ("assistant", "That's great! What aspect of ML interests you?"),
        ("user", "I'm particularly interested in neural networks"),
        ("assistant", "Neural networks are fascinating! Would you like to start with basics?"),
        ("user", "Yes, please explain perceptrons")
    ]
    
    for role, content in messages:
        event_type = EventType.USER_MESSAGE if role == "user" else EventType.ASSISTANT_MESSAGE
        await websocket_server.send_message(conn_id, event_type, {
            "chat_id": chat_id,
            "content": content,
            "role": role
        }, chat_id=chat_id)
    
    return {
        "user_id": user_id,
        "chat_id": chat_id,
        "connection_id": conn_id,
        "messages": messages,
        "server": websocket_server
    }


@pytest.fixture
def error_scenarios():
    """Collection of error scenarios for testing"""
    return {
        "rate_limit": {
            "errors": [APIError.RATE_LIMIT],
            "expected_status": 429,
            "retry_after": 60
        },
        "timeout": {
            "errors": [APIError.TIMEOUT],
            "expected_status": 504,
            "retry_after": None
        },
        "server_error": {
            "errors": [APIError.SERVER_ERROR],
            "expected_status": 500,
            "retry_after": None
        },
        "intermittent": {
            "errors": [APIError.SERVER_ERROR, None, APIError.RATE_LIMIT, None],
            "expected_status": None,  # Varies
            "retry_after": None
        }
    }


# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests"""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.process = psutil.Process(os.getpid())
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def stop(self):
            self.end_time = time.time()
            self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def get_metrics(self):
            if not all([self.start_time, self.end_time, 
                       self.start_memory, self.end_memory]):
                return None
            
            return {
                "duration_seconds": self.end_time - self.start_time,
                "memory_used_mb": self.end_memory - self.start_memory,
                "cpu_percent": self.process.cpu_percent(interval=0.1)
            }
    
    monitor = PerformanceMonitor()
    monitor.start()
    
    yield monitor
    
    monitor.stop()
    metrics = monitor.get_metrics()
    if metrics:
        print(f"\nPerformance metrics: {metrics}")


# Batch operation fixtures
@pytest.fixture
async def batch_memory_scenario(openwebui_memory_api_mock):
    """Set up scenario for batch memory operations"""
    user_id = "batch_user"
    
    # Prepare batch data
    batch_size = 100
    memories = []
    
    for i in range(batch_size):
        memories.append({
            "content": f"Batch memory {i}: Random fact about topic {i % 10}",
            "metadata": {
                "batch_id": "test_batch",
                "index": i,
                "category": f"category_{i % 5}"
            }
        })
    
    return {
        "user_id": user_id,
        "memories": memories,
        "api_mock": openwebui_memory_api_mock,
        "batch_size": batch_size
    }


# Helper fixture for filter testing
@pytest.fixture
async def mock_filter_with_apis(mock_httpx_client):
    """Create filter instance with mocked APIs"""
    from adaptive_memory_v4_0 import Filter
    
    # Create filter
    filter_instance = Filter()
    
    # Configure for testing
    filter_instance.valves.memory_extraction_enabled = True
    filter_instance.valves.memory_injection_enabled = True
    filter_instance.valves.enable_filter_orchestration = True
    
    return filter_instance