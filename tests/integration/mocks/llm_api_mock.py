"""
Mock servers for LLM API endpoints.

This module provides mock implementations of various LLM provider APIs
(OpenAI, Ollama, Anthropic) for integration testing.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from enum import Enum
import random
from unittest.mock import Mock
import threading


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


@dataclass
class MockChatMessage:
    """Mock chat message structure"""
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class MockChatResponse:
    """Mock chat completion response"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    system_fingerprint: Optional[str] = None


class LLMAPIMock:
    """Base mock implementation for LLM APIs"""
    
    def __init__(self,
                 provider: LLMProvider = LLMProvider.OPENAI,
                 enable_streaming: bool = True,
                 enable_rate_limiting: bool = False,
                 enable_random_errors: bool = False,
                 error_rate: float = 0.1,
                 response_delay_ms: int = 0,
                 max_tokens: int = 4096):
        self.provider = provider
        self.enable_streaming = enable_streaming
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_random_errors = enable_random_errors
        self.error_rate = error_rate
        self.response_delay_ms = response_delay_ms
        self.max_tokens = max_tokens
        
        self.request_count = 0
        self.total_tokens_used = 0
        self._lock = threading.Lock()
        
        # Request/response recording
        self.recorded_requests: List[Dict[str, Any]] = []
        self.recorded_responses: List[Dict[str, Any]] = []
        self.enable_recording = False
        
        # Custom response patterns
        self.response_patterns: Dict[str, str] = {}
        self.function_responses: Dict[str, Any] = {}
        
        # Rate limiting
        self.rate_limiter = self._create_rate_limiter() if enable_rate_limiting else None
        
    def _create_rate_limiter(self):
        """Create provider-specific rate limiter"""
        limits = {
            LLMProvider.OPENAI: {"requests": 60, "tokens": 150000, "window": 60},
            LLMProvider.ANTHROPIC: {"requests": 50, "tokens": 100000, "window": 60},
            LLMProvider.OLLAMA: {"requests": 100, "tokens": 200000, "window": 60},
            LLMProvider.CUSTOM: {"requests": 100, "tokens": 200000, "window": 60}
        }
        
        config = limits.get(self.provider, limits[LLMProvider.CUSTOM])
        
        class RateLimiter:
            def __init__(self, max_requests, max_tokens, window_seconds):
                self.max_requests = max_requests
                self.max_tokens = max_tokens
                self.window_seconds = window_seconds
                self.requests = []
                self.tokens = []
                self._lock = threading.Lock()
                
            def is_allowed(self, tokens_requested: int) -> tuple[bool, Optional[str]]:
                with self._lock:
                    now = time.time()
                    
                    # Clean old entries
                    self.requests = [t for t in self.requests if now - t < self.window_seconds]
                    self.tokens = [(t, tok) for t, tok in self.tokens if now - t < self.window_seconds]
                    
                    # Check request limit
                    if len(self.requests) >= self.max_requests:
                        return False, "Request rate limit exceeded"
                    
                    # Check token limit
                    total_tokens = sum(tok for _, tok in self.tokens)
                    if total_tokens + tokens_requested > self.max_tokens:
                        return False, "Token rate limit exceeded"
                    
                    # Record request
                    self.requests.append(now)
                    self.tokens.append((now, tokens_requested))
                    
                    return True, None
        
        return RateLimiter(**config)
    
    async def _simulate_delay(self):
        """Simulate network delay"""
        if self.response_delay_ms > 0:
            await asyncio.sleep(self.response_delay_ms / 1000.0)
    
    def _should_error(self) -> Optional[Dict[str, Any]]:
        """Determine if request should error"""
        if self.enable_random_errors and random.random() < self.error_rate:
            errors = [
                {"type": "rate_limit_error", "message": "Rate limit exceeded"},
                {"type": "timeout_error", "message": "Request timeout"},
                {"type": "api_error", "message": "Internal server error"},
                {"type": "invalid_request_error", "message": "Invalid request format"},
                {"type": "authentication_error", "message": "Invalid API key"}
            ]
            return random.choice(errors)
        return None
    
    def _record_request(self, endpoint: str, data: Any):
        """Record request for testing"""
        if self.enable_recording:
            self.recorded_requests.append({
                "timestamp": datetime.utcnow().isoformat(),
                "endpoint": endpoint,
                "data": data,
                "request_count": self.request_count
            })
    
    def _record_response(self, data: Any):
        """Record response for testing"""
        if self.enable_recording:
            self.recorded_responses.append({
                "timestamp": datetime.utcnow().isoformat(),
                "data": data,
                "request_count": self.request_count
            })
    
    def _count_tokens(self, text: str) -> int:
        """Simple token counting (approximate)"""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _generate_response(self, messages: List[Dict[str, Any]], 
                         model: str, **kwargs) -> str:
        """Generate mock response based on messages"""
        # Check for custom patterns
        last_message = messages[-1]["content"] if messages else ""
        
        for pattern, response in self.response_patterns.items():
            if pattern.lower() in last_message.lower():
                return response
        
        # Default responses based on content
        if "test" in last_message.lower():
            return "This is a test response from the mock LLM."
        elif "memory" in last_message.lower():
            return "I understand you're asking about memory. Memory is a fascinating topic."
        elif "hello" in last_message.lower():
            return "Hello! How can I assist you today?"
        else:
            return f"Mock response to: {last_message[:50]}..."
    
    async def create_chat_completion(self, 
                                   messages: List[Dict[str, Any]],
                                   model: str,
                                   temperature: float = 0.7,
                                   max_tokens: Optional[int] = None,
                                   stream: bool = False,
                                   **kwargs) -> Union[Dict[str, Any], AsyncGenerator]:
        """Mock chat completion endpoint"""
        self.request_count += 1
        request_data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        self._record_request("/chat/completions", request_data)
        
        await self._simulate_delay()
        
        # Check rate limiting
        if self.rate_limiter:
            token_estimate = sum(self._count_tokens(m["content"]) for m in messages)
            allowed, error_msg = self.rate_limiter.is_allowed(token_estimate)
            if not allowed:
                error_response = {
                    "error": {
                        "type": "rate_limit_error",
                        "message": error_msg
                    }
                }
                self._record_response(error_response)
                return error_response
        
        # Check for errors
        error = self._should_error()
        if error:
            self._record_response({"error": error})
            return {"error": error}
        
        # Generate response
        response_content = self._generate_response(messages, model, **kwargs)
        
        # Handle function calls if present
        function_call = None
        if "functions" in kwargs and random.random() < 0.3:  # 30% chance of function call
            func = random.choice(kwargs["functions"])
            function_call = {
                "name": func["name"],
                "arguments": json.dumps({"test": "value"})
            }
        
        if stream and self.enable_streaming:
            return self._create_stream_response(response_content, model, function_call)
        else:
            return self._create_completion_response(response_content, model, function_call)
    
    def _create_completion_response(self, content: str, model: str, 
                                  function_call: Optional[Dict] = None) -> Dict[str, Any]:
        """Create non-streaming completion response"""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        # Count tokens
        prompt_tokens = self.request_count * 10  # Simplified
        completion_tokens = self._count_tokens(content)
        total_tokens = prompt_tokens + completion_tokens
        
        self.total_tokens_used += total_tokens
        
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop" if not function_call else "function_call"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
        
        if function_call:
            response["choices"][0]["message"]["function_call"] = function_call
        
        self._record_response(response)
        return response
    
    async def _create_stream_response(self, content: str, model: str,
                                    function_call: Optional[Dict] = None) -> AsyncGenerator:
        """Create streaming completion response"""
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        # Initial chunk
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        }
        
        # Content chunks
        words = content.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.01)  # Simulate streaming delay
            
            chunk_content = word + (" " if i < len(words) - 1 else "")
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk_content},
                    "finish_reason": None
                }]
            }
        
        # Function call chunk if present
        if function_call:
            yield {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"function_call": function_call},
                    "finish_reason": "function_call"
                }]
            }
        
        # Final chunk
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop" if not function_call else "function_call"
            }]
        }
    
    def set_response_pattern(self, pattern: str, response: str):
        """Set custom response for specific patterns"""
        self.response_patterns[pattern] = response
    
    def set_function_response(self, function_name: str, response: Any):
        """Set custom response for function calls"""
        self.function_responses[function_name] = response
    
    def reset(self):
        """Reset mock to initial state"""
        with self._lock:
            self.request_count = 0
            self.total_tokens_used = 0
            self.recorded_requests.clear()
            self.recorded_responses.clear()
            self.response_patterns.clear()
            self.function_responses.clear()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_requests": self.request_count,
            "total_tokens": self.total_tokens_used,
            "average_tokens_per_request": (
                self.total_tokens_used / self.request_count 
                if self.request_count > 0 else 0
            )
        }


class OpenAIAPIMock(LLMAPIMock):
    """Mock implementation specifically for OpenAI API"""
    
    def __init__(self, **kwargs):
        super().__init__(provider=LLMProvider.OPENAI, **kwargs)
        self.available_models = [
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
            "gpt-4-32k", "gpt-3.5-turbo-16k"
        ]
    
    async def list_models(self) -> Dict[str, Any]:
        """Mock list models endpoint"""
        self._record_request("/models", {})
        
        models = []
        for model_id in self.available_models:
            models.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()) - 86400,
                "owned_by": "openai"
            })
        
        response = {
            "object": "list",
            "data": models
        }
        self._record_response(response)
        return response


class OllamaAPIMock(LLMAPIMock):
    """Mock implementation specifically for Ollama API"""
    
    def __init__(self, **kwargs):
        super().__init__(provider=LLMProvider.OLLAMA, **kwargs)
        self.available_models = [
            "llama2", "mistral", "codellama",
            "neural-chat", "starling-lm"
        ]
    
    async def generate(self, model: str, prompt: str, 
                      stream: bool = False, **kwargs) -> Union[Dict[str, Any], AsyncGenerator]:
        """Mock Ollama generate endpoint"""
        messages = [{"role": "user", "content": prompt}]
        
        # Convert to chat completion format
        response = await self.create_chat_completion(
            messages=messages,
            model=model,
            stream=stream,
            **kwargs
        )
        
        # Convert response to Ollama format
        if stream:
            async def ollama_stream():
                async for chunk in response:
                    yield {
                        "model": model,
                        "created_at": datetime.utcnow().isoformat(),
                        "response": chunk.get("choices", [{}])[0].get("delta", {}).get("content", ""),
                        "done": chunk.get("choices", [{}])[0].get("finish_reason") is not None
                    }
            return ollama_stream()
        else:
            return {
                "model": model,
                "created_at": datetime.utcnow().isoformat(),
                "response": response["choices"][0]["message"]["content"],
                "done": True,
                "context": [],
                "total_duration": 1000000000,  # 1 second in nanoseconds
                "load_duration": 100000000,
                "prompt_eval_duration": 200000000,
                "eval_duration": 700000000
            }
    
    async def list_models(self) -> Dict[str, Any]:
        """Mock Ollama list models endpoint"""
        models = []
        for model_name in self.available_models:
            models.append({
                "name": model_name,
                "modified_at": datetime.utcnow().isoformat(),
                "size": random.randint(1000000000, 5000000000),  # 1-5 GB
                "digest": f"sha256:{uuid.uuid4().hex}"
            })
        
        return {"models": models}


class AnthropicAPIMock(LLMAPIMock):
    """Mock implementation specifically for Anthropic API"""
    
    def __init__(self, **kwargs):
        super().__init__(provider=LLMProvider.ANTHROPIC, **kwargs)
        self.available_models = [
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
            "claude-2.1", "claude-2"
        ]
    
    async def create_message(self, model: str, messages: List[Dict[str, Any]],
                           max_tokens: int = 1024, **kwargs) -> Dict[str, Any]:
        """Mock Anthropic messages endpoint"""
        # Convert to internal format
        response = await self.create_chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stream=kwargs.get("stream", False),
            **kwargs
        )
        
        # Convert to Anthropic format
        if "error" in response:
            return response
        
        return {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": response["choices"][0]["message"]["content"]
            }],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": response["usage"]["prompt_tokens"],
                "output_tokens": response["usage"]["completion_tokens"]
            }
        }


class MockLLMClient:
    """Mock client that can simulate different LLM provider interfaces"""
    
    def __init__(self, provider: LLMProvider, api_mock: LLMAPIMock):
        self.provider = provider
        self.api_mock = api_mock
        self.base_url = self._get_base_url()
        self.headers = {}
        self.api_key = "mock-api-key"
    
    def _get_base_url(self) -> str:
        """Get provider-specific base URL"""
        urls = {
            LLMProvider.OPENAI: "https://api.openai.com/v1",
            LLMProvider.OLLAMA: "http://localhost:11434/api",
            LLMProvider.ANTHROPIC: "https://api.anthropic.com/v1",
            LLMProvider.CUSTOM: "http://localhost:8000/v1"
        }
        return urls.get(self.provider, urls[LLMProvider.CUSTOM])
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def post(self, url: str, json: Optional[Dict[str, Any]] = None,
                   headers: Optional[Dict[str, str]] = None) -> Mock:
        """Mock POST request to LLM API"""
        if self.provider == LLMProvider.OPENAI:
            if "/chat/completions" in url:
                result = await self.api_mock.create_chat_completion(**json)
            elif "/embeddings" in url:
                # Handled by embedding mock
                result = {"error": "Use embedding mock for embeddings"}
            else:
                result = {"error": "Unknown endpoint"}
                
        elif self.provider == LLMProvider.OLLAMA:
            if "/generate" in url:
                result = await self.api_mock.generate(**json)
            elif "/chat" in url:
                result = await self.api_mock.create_chat_completion(**json)
            else:
                result = {"error": "Unknown endpoint"}
                
        elif self.provider == LLMProvider.ANTHROPIC:
            if "/messages" in url:
                result = await self.api_mock.create_message(**json)
            else:
                result = {"error": "Unknown endpoint"}
        else:
            result = {"error": "Unknown provider"}
        
        response = Mock()
        response.status_code = 200 if "error" not in result else 400
        response.json = Mock(return_value=result)
        response.text = json.dumps(result) if isinstance(result, dict) else ""
        response.headers = {"content-type": "application/json"}
        
        # Handle streaming responses
        if isinstance(result, AsyncGenerator):
            response.aiter_lines = lambda: self._stream_lines(result)
            response.aiter_bytes = lambda: self._stream_bytes(result)
        
        return response
    
    async def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Mock:
        """Mock GET request to LLM API"""
        if self.provider == LLMProvider.OPENAI:
            if "/models" in url:
                result = await self.api_mock.list_models()
            else:
                result = {"error": "Unknown endpoint"}
        elif self.provider == LLMProvider.OLLAMA:
            if "/tags" in url:
                result = await self.api_mock.list_models()
            else:
                result = {"error": "Unknown endpoint"}
        else:
            result = {"error": "GET not supported for this provider"}
        
        response = Mock()
        response.status_code = 200 if "error" not in result else 400
        response.json = Mock(return_value=result)
        response.text = json.dumps(result)
        response.headers = {"content-type": "application/json"}
        
        return response
    
    async def _stream_lines(self, generator: AsyncGenerator):
        """Convert generator to line stream"""
        async for chunk in generator:
            if isinstance(chunk, dict):
                yield f"data: {json.dumps(chunk)}\n\n"
            else:
                yield chunk
    
    async def _stream_bytes(self, generator: AsyncGenerator):
        """Convert generator to byte stream"""
        async for line in self._stream_lines(generator):
            yield line.encode("utf-8")