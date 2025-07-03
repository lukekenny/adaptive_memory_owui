"""
Mock servers and utilities for integration testing.

This package provides mock implementations of external APIs
used by the OWUI Adaptive Memory Plugin.
"""

from .openwebui_api_mock import (
    OpenWebUIMemoryAPIMock,
    MockOpenWebUIClient,
    APIError,
    MockMemory,
    RateLimiter
)

from .llm_api_mock import (
    LLMAPIMock,
    OpenAIAPIMock,
    OllamaAPIMock,
    AnthropicAPIMock,
    MockLLMClient,
    LLMProvider,
    MockChatMessage,
    MockChatResponse
)

from .embedding_api_mock import (
    EmbeddingAPIMock,
    LocalEmbeddingModelMock,
    MockEmbeddingClient,
    EmbeddingProvider,
    EmbeddingModel
)

from .websocket_mock import (
    WebSocketServerMock,
    MockWebSocketClient,
    WebSocketState,
    EventType,
    WebSocketMessage,
    WebSocketConnection
)

__all__ = [
    # OpenWebUI mocks
    'OpenWebUIMemoryAPIMock',
    'MockOpenWebUIClient',
    'APIError',
    'MockMemory',
    'RateLimiter',
    
    # LLM mocks
    'LLMAPIMock',
    'OpenAIAPIMock',
    'OllamaAPIMock',
    'AnthropicAPIMock',
    'MockLLMClient',
    'LLMProvider',
    'MockChatMessage',
    'MockChatResponse',
    
    # Embedding mocks
    'EmbeddingAPIMock',
    'LocalEmbeddingModelMock',
    'MockEmbeddingClient',
    'EmbeddingProvider',
    'EmbeddingModel',
    
    # WebSocket mocks
    'WebSocketServerMock',
    'MockWebSocketClient',
    'WebSocketState',
    'EventType',
    'WebSocketMessage',
    'WebSocketConnection'
]