"""
Mock servers for Embedding API endpoints.

This module provides mock implementations of embedding APIs
for various providers (OpenAI, local models, etc.).
"""

import asyncio
import json
import numpy as np
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import random
from unittest.mock import Mock
import threading


class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    LOCAL = "local"
    COHERE = "cohere"
    VOYAGE = "voyage"
    CUSTOM = "custom"


@dataclass
class EmbeddingModel:
    """Embedding model configuration"""
    name: str
    dimensions: int
    max_tokens: int
    provider: EmbeddingProvider


class EmbeddingAPIMock:
    """Mock implementation for embedding APIs"""
    
    # Predefined models with their configurations
    MODELS = {
        "text-embedding-ada-002": EmbeddingModel(
            name="text-embedding-ada-002",
            dimensions=1536,
            max_tokens=8191,
            provider=EmbeddingProvider.OPENAI
        ),
        "text-embedding-3-small": EmbeddingModel(
            name="text-embedding-3-small",
            dimensions=1536,
            max_tokens=8191,
            provider=EmbeddingProvider.OPENAI
        ),
        "text-embedding-3-large": EmbeddingModel(
            name="text-embedding-3-large",
            dimensions=3072,
            max_tokens=8191,
            provider=EmbeddingProvider.OPENAI
        ),
        "all-MiniLM-L6-v2": EmbeddingModel(
            name="all-MiniLM-L6-v2",
            dimensions=384,
            max_tokens=512,
            provider=EmbeddingProvider.LOCAL
        ),
        "all-mpnet-base-v2": EmbeddingModel(
            name="all-mpnet-base-v2",
            dimensions=768,
            max_tokens=512,
            provider=EmbeddingProvider.LOCAL
        ),
        "embed-english-v3.0": EmbeddingModel(
            name="embed-english-v3.0",
            dimensions=1024,
            max_tokens=512,
            provider=EmbeddingProvider.COHERE
        )
    }
    
    def __init__(self,
                 default_model: str = "text-embedding-ada-002",
                 enable_rate_limiting: bool = False,
                 enable_random_errors: bool = False,
                 error_rate: float = 0.1,
                 response_delay_ms: int = 0,
                 deterministic: bool = False):
        self.default_model = default_model
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_random_errors = enable_random_errors
        self.error_rate = error_rate
        self.response_delay_ms = response_delay_ms
        self.deterministic = deterministic
        
        self.request_count = 0
        self.total_tokens_used = 0
        self._lock = threading.Lock()
        
        # Request/response recording
        self.recorded_requests: List[Dict[str, Any]] = []
        self.recorded_responses: List[Dict[str, Any]] = []
        self.enable_recording = False
        
        # Cache for deterministic embeddings
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Custom embeddings for specific texts
        self.custom_embeddings: Dict[str, List[float]] = {}
        
        # Rate limiting
        self.rate_limiter = self._create_rate_limiter() if enable_rate_limiting else None
    
    def _create_rate_limiter(self):
        """Create rate limiter for embeddings"""
        class RateLimiter:
            def __init__(self, max_requests=60, max_tokens=1000000, window_seconds=60):
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
                    
                    # Check limits
                    if len(self.requests) >= self.max_requests:
                        return False, "Request rate limit exceeded"
                    
                    total_tokens = sum(tok for _, tok in self.tokens)
                    if total_tokens + tokens_requested > self.max_tokens:
                        return False, "Token rate limit exceeded"
                    
                    # Record request
                    self.requests.append(now)
                    self.tokens.append((now, tokens_requested))
                    
                    return True, None
        
        return RateLimiter()
    
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
                {"type": "model_not_found", "message": "Model not found"}
            ]
            return random.choice(errors)
        return None
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count"""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def _generate_embedding(self, text: str, model_config: EmbeddingModel) -> List[float]:
        """Generate mock embedding vector"""
        if self.deterministic:
            # Use cached embedding if available
            cache_key = f"{model_config.name}:{text}"
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]
        
        # Check for custom embeddings
        if text in self.custom_embeddings:
            embedding = self.custom_embeddings[text]
            # Resize if needed
            if len(embedding) != model_config.dimensions:
                embedding = self._resize_embedding(embedding, model_config.dimensions)
            return embedding
        
        if self.deterministic:
            # Generate deterministic embedding based on text hash
            import hashlib
            
            # Create seed from text
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            seed = int(text_hash[:8], 16)
            np.random.seed(seed)
            
            # Generate embedding
            embedding = np.random.randn(model_config.dimensions).tolist()
            
            # Normalize
            norm = np.linalg.norm(embedding)
            embedding = [x / norm for x in embedding]
            
            # Cache it
            cache_key = f"{model_config.name}:{text}"
            self._embedding_cache[cache_key] = embedding
            
            return embedding
        else:
            # Generate random embedding
            embedding = np.random.randn(model_config.dimensions)
            # Normalize to unit vector
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
    
    def _resize_embedding(self, embedding: List[float], target_dim: int) -> List[float]:
        """Resize embedding to target dimensions"""
        current_dim = len(embedding)
        
        if current_dim == target_dim:
            return embedding
        elif current_dim < target_dim:
            # Pad with zeros
            return embedding + [0.0] * (target_dim - current_dim)
        else:
            # Truncate
            return embedding[:target_dim]
    
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
    
    async def create_embeddings(self,
                              input: Union[str, List[str]],
                              model: Optional[str] = None,
                              encoding_format: str = "float",
                              user: Optional[str] = None) -> Dict[str, Any]:
        """Mock embedding creation endpoint"""
        self.request_count += 1
        
        # Normalize input to list
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input
        
        model = model or self.default_model
        
        request_data = {
            "input": input,
            "model": model,
            "encoding_format": encoding_format,
            "user": user
        }
        self._record_request("/embeddings", request_data)
        
        await self._simulate_delay()
        
        # Get model configuration
        model_config = self.MODELS.get(model)
        if not model_config:
            error = {
                "error": {
                    "type": "model_not_found",
                    "message": f"Model {model} not found"
                }
            }
            self._record_response(error)
            return error
        
        # Count tokens
        total_tokens = sum(self._count_tokens(text) for text in texts)
        
        # Check rate limiting
        if self.rate_limiter:
            allowed, error_msg = self.rate_limiter.is_allowed(total_tokens)
            if not allowed:
                error = {
                    "error": {
                        "type": "rate_limit_error",
                        "message": error_msg
                    }
                }
                self._record_response(error)
                return error
        
        # Check for errors
        error = self._should_error()
        if error:
            response = {"error": error}
            self._record_response(response)
            return response
        
        # Generate embeddings
        embeddings = []
        for i, text in enumerate(texts):
            embedding = self._generate_embedding(text, model_config)
            
            if encoding_format == "base64":
                # Convert to base64 (simplified)
                import base64
                embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
                embedding_b64 = base64.b64encode(embedding_bytes).decode('utf-8')
                embeddings.append({
                    "object": "embedding",
                    "embedding": embedding_b64,
                    "index": i
                })
            else:
                embeddings.append({
                    "object": "embedding",
                    "embedding": embedding,
                    "index": i
                })
        
        self.total_tokens_used += total_tokens
        
        response = {
            "object": "list",
            "data": embeddings,
            "model": model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
        
        self._record_response(response)
        return response
    
    def set_custom_embedding(self, text: str, embedding: List[float]):
        """Set custom embedding for specific text"""
        self.custom_embeddings[text] = embedding
    
    def compute_similarity(self, embedding1: List[float], 
                         embedding2: List[float]) -> float:
        """Compute cosine similarity between embeddings"""
        # Ensure same dimensions
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have same dimensions")
        
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def batch_compute_similarities(self, query_embedding: List[float],
                                 embeddings: List[List[float]]) -> List[float]:
        """Compute similarities between query and multiple embeddings"""
        return [
            self.compute_similarity(query_embedding, emb)
            for emb in embeddings
        ]
    
    def reset(self):
        """Reset mock to initial state"""
        with self._lock:
            self.request_count = 0
            self.total_tokens_used = 0
            self.recorded_requests.clear()
            self.recorded_responses.clear()
            self._embedding_cache.clear()
            self.custom_embeddings.clear()
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_requests": self.request_count,
            "total_tokens": self.total_tokens_used,
            "average_tokens_per_request": (
                self.total_tokens_used / self.request_count 
                if self.request_count > 0 else 0
            ),
            "cached_embeddings": len(self._embedding_cache),
            "custom_embeddings": len(self.custom_embeddings)
        }


class LocalEmbeddingModelMock:
    """Mock for local embedding models (sentence-transformers style)"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 device: str = "cpu",
                 enable_errors: bool = False):
        self.model_name = model_name
        self.device = device
        self.enable_errors = enable_errors
        
        # Get model config
        self.model_config = EmbeddingAPIMock.MODELS.get(
            model_name,
            EmbeddingModel(
                name=model_name,
                dimensions=384,
                max_tokens=512,
                provider=EmbeddingProvider.LOCAL
            )
        )
        
        self.encode_count = 0
        self._embedding_mock = EmbeddingAPIMock(
            default_model=model_name,
            deterministic=True
        )
    
    def encode(self, sentences: Union[str, List[str]], 
              batch_size: int = 32,
              show_progress_bar: bool = False,
              convert_to_numpy: bool = True,
              normalize_embeddings: bool = True):
        """Mock encode method for sentence-transformers"""
        self.encode_count += 1
        
        if self.enable_errors and random.random() < 0.1:
            raise RuntimeError("Mock embedding model error")
        
        # Normalize input
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # Generate embeddings
        embeddings = []
        for sentence in sentences:
            embedding = self._embedding_mock._generate_embedding(
                sentence, 
                self.model_config
            )
            embeddings.append(embedding)
        
        # Normalize if requested
        if normalize_embeddings:
            embeddings = [
                (np.array(emb) / np.linalg.norm(emb)).tolist()
                for emb in embeddings
            ]
        
        # Convert to numpy if requested
        if convert_to_numpy:
            return np.array(embeddings)
        else:
            return embeddings
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model_config.dimensions
    
    def get_max_seq_length(self) -> int:
        """Get maximum sequence length"""
        return self.model_config.max_tokens


class MockEmbeddingClient:
    """Mock client for embedding APIs"""
    
    def __init__(self, provider: EmbeddingProvider, api_mock: EmbeddingAPIMock):
        self.provider = provider
        self.api_mock = api_mock
        self.base_url = self._get_base_url()
        self.headers = {}
    
    def _get_base_url(self) -> str:
        """Get provider-specific base URL"""
        urls = {
            EmbeddingProvider.OPENAI: "https://api.openai.com/v1",
            EmbeddingProvider.COHERE: "https://api.cohere.ai/v1",
            EmbeddingProvider.VOYAGE: "https://api.voyageai.com/v1",
            EmbeddingProvider.CUSTOM: "http://localhost:8000/v1"
        }
        return urls.get(self.provider, urls[EmbeddingProvider.CUSTOM])
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def post(self, url: str, json: Optional[Dict[str, Any]] = None,
                   headers: Optional[Dict[str, str]] = None) -> Mock:
        """Mock POST request to embedding API"""
        if "/embeddings" in url or "/embed" in url:
            result = await self.api_mock.create_embeddings(**json)
        else:
            result = {"error": "Unknown endpoint"}
        
        response = Mock()
        response.status_code = 200 if "error" not in result else 400
        response.json = Mock(return_value=result)
        response.text = json.dumps(result)
        response.headers = {"content-type": "application/json"}
        
        return response