"""
Mock server for OpenWebUI Memory API endpoints.

This module provides mock implementations of OpenWebUI API endpoints
for integration testing of the adaptive memory plugin.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
from unittest.mock import Mock
import threading


class APIError(Enum):
    """Common API error types for testing"""
    RATE_LIMIT = "rate_limit_exceeded"
    TIMEOUT = "request_timeout"
    NOT_FOUND = "resource_not_found"
    UNAUTHORIZED = "unauthorized"
    SERVER_ERROR = "internal_server_error"
    INVALID_REQUEST = "invalid_request"


@dataclass
class MockMemory:
    """Mock memory object structure"""
    id: str
    user_id: str
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    context: str = "general"
    embedding: Optional[List[float]] = None


class RateLimiter:
    """Simple rate limiter for mock API testing"""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def is_allowed(self, user_id: str) -> bool:
        """Check if request is allowed for user"""
        with self._lock:
            now = time.time()
            if user_id not in self.requests:
                self.requests[user_id] = []
            
            # Remove old requests outside window
            self.requests[user_id] = [
                req_time for req_time in self.requests[user_id]
                if now - req_time < self.window_seconds
            ]
            
            if len(self.requests[user_id]) >= self.max_requests:
                return False
            
            self.requests[user_id].append(now)
            return True


class OpenWebUIMemoryAPIMock:
    """Mock implementation of OpenWebUI Memory API"""
    
    def __init__(self, 
                 enable_rate_limiting: bool = False,
                 enable_random_errors: bool = False,
                 error_rate: float = 0.1,
                 response_delay_ms: int = 0):
        self.memories: Dict[str, List[MockMemory]] = {}
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_random_errors = enable_random_errors
        self.error_rate = error_rate
        self.response_delay_ms = response_delay_ms
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        self.request_count = 0
        self.error_sequence: List[APIError] = []
        self._lock = threading.Lock()
        
        # Request/response recording
        self.recorded_requests: List[Dict[str, Any]] = []
        self.recorded_responses: List[Dict[str, Any]] = []
        self.enable_recording = False
        
    async def _simulate_delay(self):
        """Simulate network delay"""
        if self.response_delay_ms > 0:
            await asyncio.sleep(self.response_delay_ms / 1000.0)
    
    def _should_error(self) -> Optional[APIError]:
        """Determine if request should error"""
        if self.error_sequence:
            return self.error_sequence.pop(0)
        
        if self.enable_random_errors and random.random() < self.error_rate:
            return random.choice(list(APIError))
        
        return None
    
    def _record_request(self, endpoint: str, method: str, data: Any):
        """Record request for testing"""
        if self.enable_recording:
            self.recorded_requests.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "endpoint": endpoint,
                "method": method,
                "data": data,
                "request_count": self.request_count
            })
    
    def _record_response(self, status: int, data: Any):
        """Record response for testing"""
        if self.enable_recording:
            self.recorded_responses.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "data": data,
                "request_count": self.request_count
            })
    
    def set_error_sequence(self, errors: List[APIError]):
        """Set specific error sequence for testing"""
        self.error_sequence = errors.copy()
    
    async def create_memory(self, user_id: str, content: str, 
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock create_memory endpoint"""
        self.request_count += 1
        self._record_request("/memories/create", "POST", {
            "user_id": user_id,
            "content": content,
            "metadata": metadata
        })
        
        await self._simulate_delay()
        
        # Check rate limiting
        if self.enable_rate_limiting and not self.rate_limiter.is_allowed(user_id):
            response = {
                "error": APIError.RATE_LIMIT.value,
                "message": "Rate limit exceeded",
                "retry_after": 60
            }
            self._record_response(429, response)
            return response
        
        # Check for errors
        error = self._should_error()
        if error:
            status_map = {
                APIError.RATE_LIMIT: 429,
                APIError.TIMEOUT: 504,
                APIError.NOT_FOUND: 404,
                APIError.UNAUTHORIZED: 401,
                APIError.SERVER_ERROR: 500,
                APIError.INVALID_REQUEST: 400
            }
            response = {
                "error": error.value,
                "message": f"Mock error: {error.value}"
            }
            self._record_response(status_map.get(error, 500), response)
            return response
        
        # Create memory
        with self._lock:
            memory = MockMemory(
                id=f"mem_{uuid.uuid4().hex[:8]}",
                user_id=user_id,
                content=content,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata=metadata or {}
            )
            
            if user_id not in self.memories:
                self.memories[user_id] = []
            self.memories[user_id].append(memory)
            
            response = {
                "id": memory.id,
                "user_id": memory.user_id,
                "content": memory.content,
                "timestamp": memory.timestamp,
                "metadata": memory.metadata
            }
            self._record_response(200, response)
            return response
    
    async def get_memories(self, user_id: str, limit: int = 20, 
                          offset: int = 0) -> Dict[str, Any]:
        """Mock get_memories endpoint"""
        self.request_count += 1
        self._record_request("/memories/list", "GET", {
            "user_id": user_id,
            "limit": limit,
            "offset": offset
        })
        
        await self._simulate_delay()
        
        # Check for errors
        error = self._should_error()
        if error:
            response = {"error": error.value}
            self._record_response(500, response)
            return response
        
        with self._lock:
            user_memories = self.memories.get(user_id, [])
            # Sort by timestamp (newest first)
            sorted_memories = sorted(
                user_memories, 
                key=lambda m: m.timestamp, 
                reverse=True
            )
            
            # Apply pagination
            paginated = sorted_memories[offset:offset + limit]
            
            response = {
                "memories": [
                    {
                        "id": m.id,
                        "content": m.content,
                        "timestamp": m.timestamp,
                        "metadata": m.metadata,
                        "importance": m.importance,
                        "context": m.context
                    }
                    for m in paginated
                ],
                "total": len(user_memories),
                "limit": limit,
                "offset": offset
            }
            self._record_response(200, response)
            return response
    
    async def query_memory(self, user_id: str, query: str, 
                          limit: int = 10) -> Dict[str, Any]:
        """Mock query_memory endpoint"""
        self.request_count += 1
        self._record_request("/memories/query", "POST", {
            "user_id": user_id,
            "query": query,
            "limit": limit
        })
        
        await self._simulate_delay()
        
        # Check for errors
        error = self._should_error()
        if error:
            response = {"error": error.value}
            self._record_response(500, response)
            return response
        
        with self._lock:
            user_memories = self.memories.get(user_id, [])
            
            # Simple query matching (in real implementation would use embeddings)
            query_lower = query.lower()
            matched_memories = []
            
            for memory in user_memories:
                if query_lower in memory.content.lower():
                    score = len(query_lower) / len(memory.content)
                    matched_memories.append((memory, score))
            
            # Sort by relevance score
            matched_memories.sort(key=lambda x: x[1], reverse=True)
            matched_memories = matched_memories[:limit]
            
            response = {
                "results": [
                    {
                        "id": m.id,
                        "content": m.content,
                        "timestamp": m.timestamp,
                        "relevance_score": score,
                        "metadata": m.metadata
                    }
                    for m, score in matched_memories
                ],
                "query": query,
                "total_results": len(matched_memories)
            }
            self._record_response(200, response)
            return response
    
    async def update_memory_by_id(self, memory_id: str, user_id: str,
                                 content: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock update_memory_by_id endpoint"""
        self.request_count += 1
        self._record_request("/memories/update", "PUT", {
            "memory_id": memory_id,
            "user_id": user_id,
            "content": content,
            "metadata": metadata
        })
        
        await self._simulate_delay()
        
        with self._lock:
            user_memories = self.memories.get(user_id, [])
            
            for memory in user_memories:
                if memory.id == memory_id:
                    if content is not None:
                        memory.content = content
                    if metadata is not None:
                        memory.metadata.update(metadata)
                    memory.timestamp = datetime.now(timezone.utc).isoformat()
                    
                    response = {
                        "id": memory.id,
                        "content": memory.content,
                        "timestamp": memory.timestamp,
                        "metadata": memory.metadata
                    }
                    self._record_response(200, response)
                    return response
            
            # Memory not found
            response = {
                "error": APIError.NOT_FOUND.value,
                "message": f"Memory {memory_id} not found"
            }
            self._record_response(404, response)
            return response
    
    async def delete_memory_by_id(self, memory_id: str, user_id: str) -> Dict[str, Any]:
        """Mock delete_memory_by_id endpoint"""
        self.request_count += 1
        self._record_request("/memories/delete", "DELETE", {
            "memory_id": memory_id,
            "user_id": user_id
        })
        
        await self._simulate_delay()
        
        with self._lock:
            user_memories = self.memories.get(user_id, [])
            
            for i, memory in enumerate(user_memories):
                if memory.id == memory_id:
                    del user_memories[i]
                    response = {"success": True, "message": "Memory deleted"}
                    self._record_response(200, response)
                    return response
            
            # Memory not found
            response = {
                "error": APIError.NOT_FOUND.value,
                "message": f"Memory {memory_id} not found"
            }
            self._record_response(404, response)
            return response
    
    async def bulk_create_memories(self, user_id: str, 
                                 memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock bulk memory creation"""
        self.request_count += 1
        self._record_request("/memories/bulk_create", "POST", {
            "user_id": user_id,
            "memories": memories
        })
        
        await self._simulate_delay()
        
        created_memories = []
        errors = []
        
        for mem_data in memories:
            try:
                result = await self.create_memory(
                    user_id=user_id,
                    content=mem_data.get("content", ""),
                    metadata=mem_data.get("metadata", {})
                )
                if "error" in result:
                    errors.append(result)
                else:
                    created_memories.append(result)
            except Exception as e:
                errors.append({
                    "error": "processing_error",
                    "message": str(e)
                })
        
        response = {
            "created": created_memories,
            "errors": errors,
            "total_requested": len(memories),
            "total_created": len(created_memories)
        }
        self._record_response(200, response)
        return response
    
    def reset(self):
        """Reset mock to initial state"""
        with self._lock:
            self.memories.clear()
            self.request_count = 0
            self.recorded_requests.clear()
            self.recorded_responses.clear()
            self.error_sequence.clear()
    
    def get_recording(self) -> Dict[str, Any]:
        """Get recorded requests and responses"""
        return {
            "requests": self.recorded_requests.copy(),
            "responses": self.recorded_responses.copy(),
            "total_requests": self.request_count
        }
    
    def load_test_data(self, test_memories: List[Dict[str, Any]]):
        """Load test data into mock"""
        with self._lock:
            for mem_data in test_memories:
                user_id = mem_data.get("user_id", "test_user")
                memory = MockMemory(
                    id=mem_data.get("id", f"mem_{uuid.uuid4().hex[:8]}"),
                    user_id=user_id,
                    content=mem_data.get("content", ""),
                    timestamp=mem_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    metadata=mem_data.get("metadata", {}),
                    importance=mem_data.get("importance", 0.5),
                    context=mem_data.get("context", "general")
                )
                
                if user_id not in self.memories:
                    self.memories[user_id] = []
                self.memories[user_id].append(memory)


class MockOpenWebUIClient:
    """Mock client that mimics httpx.AsyncClient interface for OpenWebUI"""
    
    def __init__(self, memory_api_mock: OpenWebUIMemoryAPIMock):
        self.memory_api = memory_api_mock
        self.headers = {}
        self.base_url = "http://mock-openwebui:8080"
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def _parse_endpoint(self, url: str) -> tuple[str, str]:
        """Parse URL to determine endpoint and method"""
        # Remove base URL if present
        if url.startswith(self.base_url):
            url = url[len(self.base_url):]
        
        # Determine endpoint
        if "/memories" in url:
            if url.endswith("/create"):
                return "create_memory", "POST"
            elif url.endswith("/list"):
                return "get_memories", "GET"
            elif url.endswith("/query"):
                return "query_memory", "POST"
            elif "/update/" in url:
                return "update_memory", "PUT"
            elif "/delete/" in url:
                return "delete_memory", "DELETE"
            elif url.endswith("/bulk_create"):
                return "bulk_create", "POST"
        
        return "unknown", "GET"
    
    async def post(self, url: str, json: Optional[Dict[str, Any]] = None,
                   headers: Optional[Dict[str, str]] = None) -> Mock:
        """Mock POST request"""
        endpoint, _ = self._parse_endpoint(url)
        
        if endpoint == "create_memory":
            result = await self.memory_api.create_memory(
                user_id=json.get("user_id", ""),
                content=json.get("content", ""),
                metadata=json.get("metadata")
            )
        elif endpoint == "query_memory":
            result = await self.memory_api.query_memory(
                user_id=json.get("user_id", ""),
                query=json.get("query", ""),
                limit=json.get("limit", 10)
            )
        elif endpoint == "bulk_create":
            result = await self.memory_api.bulk_create_memories(
                user_id=json.get("user_id", ""),
                memories=json.get("memories", [])
            )
        else:
            result = {"error": "unknown_endpoint"}
        
        response = Mock()
        response.status_code = 200 if "error" not in result else 400
        response.json = Mock(return_value=result)
        response.text = json.dumps(result)
        response.headers = {"content-type": "application/json"}
        
        return response
    
    async def get(self, url: str, params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None) -> Mock:
        """Mock GET request"""
        endpoint, _ = self._parse_endpoint(url)
        
        if endpoint == "get_memories":
            result = await self.memory_api.get_memories(
                user_id=params.get("user_id", ""),
                limit=int(params.get("limit", 20)),
                offset=int(params.get("offset", 0))
            )
        else:
            result = {"error": "unknown_endpoint"}
        
        response = Mock()
        response.status_code = 200 if "error" not in result else 400
        response.json = Mock(return_value=result)
        response.text = json.dumps(result)
        response.headers = {"content-type": "application/json"}
        
        return response
    
    async def put(self, url: str, json: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None) -> Mock:
        """Mock PUT request"""
        # Extract memory_id from URL
        memory_id = url.split("/")[-1] if "/" in url else ""
        
        result = await self.memory_api.update_memory_by_id(
            memory_id=memory_id,
            user_id=json.get("user_id", ""),
            content=json.get("content"),
            metadata=json.get("metadata")
        )
        
        response = Mock()
        response.status_code = 200 if "error" not in result else 404
        response.json = Mock(return_value=result)
        response.text = json.dumps(result)
        response.headers = {"content-type": "application/json"}
        
        return response
    
    async def delete(self, url: str, headers: Optional[Dict[str, str]] = None) -> Mock:
        """Mock DELETE request"""
        # Extract memory_id from URL
        parts = url.split("/")
        memory_id = parts[-1] if parts else ""
        user_id = headers.get("X-User-Id", "") if headers else ""
        
        result = await self.memory_api.delete_memory_by_id(
            memory_id=memory_id,
            user_id=user_id
        )
        
        response = Mock()
        response.status_code = 200 if "error" not in result else 404
        response.json = Mock(return_value=result)
        response.text = json.dumps(result)
        response.headers = {"content-type": "application/json"}
        
        return response