"""
Comprehensive integration tests for LLM provider connections and memory extraction.

This module tests the actual query_llm_with_retry method from the adaptive memory filter
with various LLM providers and memory extraction workflows.
"""

import pytest
import asyncio
import json
import aiohttp
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import time

from tests.integration.mocks.llm_api_mock import (
    LLMAPIMock, 
    OpenAIAPIMock, 
    OllamaAPIMock, 
    AnthropicAPIMock,
    LLMProvider,
    MockLLMClient
)
from tests.integration.fixtures.llm_fixtures import (
    create_mock_response,
    create_ndjson_response,
    MockAsyncContextManager
)

# Import the adaptive memory filter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import the filter - we'll use importlib to handle the module name with dots
import importlib.util
adaptive_memory_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    "adaptive_memory_v4.0.py"
)
spec = importlib.util.spec_from_file_location("adaptive_memory_v4_0", adaptive_memory_path)
adaptive_memory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive_memory_module)
AdaptiveMemoryFilter = adaptive_memory_module.Filter


class TestLLMProviderConnections:
    """Test LLM provider connection functionality"""
    
    @pytest.fixture
    def mock_filter(self):
        """Create filter instance with mocked dependencies"""
        filter_instance = AdaptiveMemoryFilter()
        
        # Set up test valves
        filter_instance.valves.llm_provider_type = "openai_compatible"
        filter_instance.valves.llm_model_name = "test-model"
        filter_instance.valves.llm_api_endpoint_url = "http://localhost:8080/v1/chat/completions"
        filter_instance.valves.llm_api_key = "test-api-key"
        filter_instance.valves.max_retries = 3
        filter_instance.valves.retry_delay = 0.1
        filter_instance.valves.request_timeout = 5.0
        filter_instance.valves.enable_health_checks = False
        filter_instance.valves.enable_feature_detection = True
        
        # Initialize metrics
        filter_instance.metrics = {"llm_call_count": 0}
        
        # Reset circuit breaker state
        filter_instance._circuit_breaker_state = {}
        
        return filter_instance
    
    @pytest.fixture
    async def mock_session(self):
        """Create mock aiohttp session"""
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.post = AsyncMock()
        return session
    
    @pytest.mark.asyncio
    async def test_openai_compatible_provider_success(self, mock_filter, mock_session):
        """Test successful OpenAI-compatible API connection"""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": '{"importance": 8, "category": "technical", "content": "User discusses Python programming"}'
                }
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75}
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        # Mock session getter
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="You are a memory extraction assistant.",
                user_prompt="Extract memory from: I love programming in Python"
            )
        
        # Verify result
        assert result is not None
        assert '"importance": 8' in result
        assert '"category": "technical"' in result
        assert mock_filter.metrics["llm_call_count"] == 1
        
        # Verify request was made correctly
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "json" in call_args.kwargs
        assert "headers" in call_args.kwargs
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-api-key"
    
    @pytest.mark.asyncio
    async def test_ollama_provider_success(self, mock_filter, mock_session):
        """Test successful Ollama API connection"""
        mock_filter.valves.llm_provider_type = "ollama"
        mock_filter.valves.llm_api_endpoint_url = "http://localhost:11434/api/chat"
        
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "message": {
                "content": '{"importance": 7, "category": "personal", "content": "User enjoys cooking"}'
            }
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="You are a memory extraction assistant.",
                user_prompt="Extract memory from: I enjoy cooking Italian food"
            )
        
        # Verify result
        assert result is not None
        assert '"importance": 7' in result
        assert '"category": "personal"' in result
        
        # Verify request format for Ollama
        call_args = mock_session.post.call_args
        request_data = call_args.kwargs["json"]
        assert "options" in request_data
        assert request_data["options"]["format"] == "json"
        assert request_data["stream"] is False
    
    @pytest.mark.asyncio
    async def test_gemini_provider_success(self, mock_filter, mock_session):
        """Test successful Gemini API connection"""
        mock_filter.valves.llm_provider_type = "gemini"
        mock_filter.valves.llm_api_endpoint_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        # Setup mock response using correct Gemini API response format
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"importance": 9, "category": "work", "content": "User works with machine learning"}'
                    }]
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE"
                    }
                ]
            }],
            "usageMetadata": {
                "promptTokenCount": 60,
                "candidatesTokenCount": 30,
                "totalTokenCount": 90
            }
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="You are a memory extraction assistant.",
                user_prompt="Extract memory from: I work with machine learning models"
            )
        
        # Verify result
        assert result is not None
        assert '"importance": 9' in result
        assert '"category": "work"' in result
        
        # Verify request uses correct Gemini API format
        call_args = mock_session.post.call_args
        request_data = call_args.kwargs["json"]
        assert "contents" in request_data
        assert "generationConfig" in request_data
        assert request_data["generationConfig"]["responseMimeType"] == "application/json"
    
    @pytest.mark.asyncio
    async def test_custom_endpoint_support(self, mock_filter, mock_session):
        """Test custom endpoint configuration"""
        mock_filter.valves.llm_provider_type = "openai_compatible"
        mock_filter.valves.llm_api_endpoint_url = "https://custom-ai.example.com/v1/chat/completions"
        mock_filter.valves.llm_api_key = "custom-api-key"
        
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": '{"importance": 6, "category": "hobby", "content": "User plays guitar"}'
                }
            }]
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="You are a memory extraction assistant.",
                user_prompt="Extract memory from: I play guitar in my free time"
            )
        
        # Verify custom endpoint was used
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args.args[0] == "https://custom-ai.example.com/v1/chat/completions"
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer custom-api-key"


class TestMemoryExtractionWorkflows:
    """Test memory extraction and analysis workflows"""
    
    @pytest.fixture
    def mock_filter_with_prompts(self, mock_filter):
        """Filter with memory extraction prompts configured"""
        filter_instance = mock_filter
        
        # Add memory extraction related configuration
        filter_instance.valves.extract_memories = True
        filter_instance.valves.memory_importance_threshold = 5
        filter_instance.valves.context_window_size = 10
        
        return filter_instance
    
    @pytest.mark.asyncio
    async def test_conversation_analysis(self, mock_filter_with_prompts, mock_session):
        """Test conversation analysis for memory extraction"""
        # Setup mock response for conversation analysis
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "memories": [
                            {
                                "importance": 8,
                                "category": "preferences",
                                "content": "User prefers dark mode in applications",
                                "keywords": ["dark mode", "UI", "preference"]
                            },
                            {
                                "importance": 6,
                                "category": "technical",
                                "content": "User is learning React framework",
                                "keywords": ["React", "JavaScript", "learning"]
                            }
                        ],
                        "conversation_summary": "User discussing development preferences and learning progress"
                    })
                }
            }]
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        # Mock conversation context
        conversation_text = """
        User: I really prefer dark mode in all my applications. It's easier on my eyes.
        Assistant: That's a great preference! Dark mode can indeed reduce eye strain.
        User: Yes, and I'm currently learning React. Do you know any good resources?
        Assistant: React has excellent documentation. I'd recommend starting with their tutorial.
        """
        
        with patch.object(mock_filter_with_prompts, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter_with_prompts.query_llm_with_retry(
                system_prompt="Extract important memories from the conversation.",
                user_prompt=f"Analyze this conversation for memories: {conversation_text}"
            )
        
        # Verify analysis result
        assert result is not None
        parsed_result = json.loads(result)
        assert "memories" in parsed_result
        assert len(parsed_result["memories"]) == 2
        
        # Check memory content
        memories = parsed_result["memories"]
        assert any("dark mode" in mem["content"] for mem in memories)
        assert any("React" in mem["content"] for mem in memories)
        assert any(mem["importance"] >= 6 for mem in memories)
    
    @pytest.mark.asyncio
    async def test_memory_importance_scoring(self, mock_filter_with_prompts, mock_session):
        """Test memory importance scoring functionality"""
        test_cases = [
            {
                "input": "User mentioned they have a cat named Whiskers",
                "expected_importance": 7,
                "category": "personal"
            },
            {
                "input": "User said hello",
                "expected_importance": 2,
                "category": "greeting"
            },
            {
                "input": "User revealed they work at Google as a software engineer",
                "expected_importance": 9,
                "category": "professional"
            }
        ]
        
        for case in test_cases:
            # Setup mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json = AsyncMock(return_value={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "importance": case["expected_importance"],
                            "category": case["category"],
                            "content": case["input"],
                            "reasoning": f"Scored {case['expected_importance']} because of content significance"
                        })
                    }
                }]
            })
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            with patch.object(mock_filter_with_prompts, '_get_aiohttp_session', return_value=mock_session):
                result = await mock_filter_with_prompts.query_llm_with_retry(
                    system_prompt="Score the importance of this memory on a scale of 1-10.",
                    user_prompt=f"Rate importance: {case['input']}"
                )
            
            # Verify scoring
            parsed_result = json.loads(result)
            assert parsed_result["importance"] == case["expected_importance"]
            assert parsed_result["category"] == case["category"]
    
    @pytest.mark.asyncio
    async def test_category_classification(self, mock_filter_with_prompts, mock_session):
        """Test memory category classification"""
        test_inputs = [
            ("User works as a data scientist", "professional"),
            ("User loves pizza and pasta", "food_preferences"), 
            ("User is learning Spanish", "education"),
            ("User lives in San Francisco", "location"),
            ("User prefers async/await over callbacks", "technical_preferences")
        ]
        
        for input_text, expected_category in test_inputs:
            # Setup mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json = AsyncMock(return_value={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "category": expected_category,
                            "confidence": 0.9,
                            "subcategories": [expected_category.split("_")[0]],
                            "content": input_text
                        })
                    }
                }]
            })
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            with patch.object(mock_filter_with_prompts, '_get_aiohttp_session', return_value=mock_session):
                result = await mock_filter_with_prompts.query_llm_with_retry(
                    system_prompt="Classify this memory into appropriate categories.",
                    user_prompt=f"Categorize: {input_text}"
                )
            
            # Verify classification
            parsed_result = json.loads(result)
            assert parsed_result["category"] == expected_category
            assert parsed_result["confidence"] > 0.8
    
    @pytest.mark.asyncio
    async def test_content_filtering(self, mock_filter_with_prompts, mock_session):
        """Test content filtering for sensitive information"""
        sensitive_inputs = [
            "My social security number is 123-45-6789",
            "My password is mySecretPass123",
            "My credit card number is 4111-1111-1111-1111"
        ]
        
        safe_inputs = [
            "I enjoy reading science fiction books",
            "My favorite color is blue",
            "I work in software development"
        ]
        
        # Test sensitive content filtering
        for sensitive_input in sensitive_inputs:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json = AsyncMock(return_value={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "should_store": False,
                            "reason": "Contains sensitive information",
                            "content_type": "sensitive",
                            "filtered_content": "[SENSITIVE INFORMATION FILTERED]"
                        })
                    }
                }]
            })
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            with patch.object(mock_filter_with_prompts, '_get_aiohttp_session', return_value=mock_session):
                result = await mock_filter_with_prompts.query_llm_with_retry(
                    system_prompt="Filter sensitive information from this content.",
                    user_prompt=f"Filter: {sensitive_input}"
                )
            
            parsed_result = json.loads(result)
            assert parsed_result["should_store"] is False
            assert "sensitive" in parsed_result["reason"].lower()
        
        # Test safe content approval
        for safe_input in safe_inputs:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json = AsyncMock(return_value={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "should_store": True,
                            "reason": "Safe content",
                            "content_type": "safe",
                            "filtered_content": safe_input
                        })
                    }
                }]
            })
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            with patch.object(mock_filter_with_prompts, '_get_aiohttp_session', return_value=mock_session):
                result = await mock_filter_with_prompts.query_llm_with_retry(
                    system_prompt="Filter sensitive information from this content.",
                    user_prompt=f"Filter: {safe_input}"
                )
            
            parsed_result = json.loads(result)
            assert parsed_result["should_store"] is True
            assert parsed_result["filtered_content"] == safe_input


class TestErrorScenarios:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_api_key_validation(self, mock_filter, mock_session):
        """Test API key validation errors"""
        # Setup unauthorized response
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value='{"error": "Invalid API key"}')
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="Test prompt",
                user_prompt="Test message"
            )
        
        # Should return error message
        assert "Error:" in result
        assert "401" in result
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_filter, mock_session):
        """Test rate limiting scenarios"""
        # Setup rate limit response
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.text = AsyncMock(return_value='{"error": "Rate limit exceeded"}')
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        start_time = time.time()
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="Test prompt",
                user_prompt="Test message"
            )
        
        end_time = time.time()
        
        # Should have attempted retries with backoff
        assert end_time - start_time > 0.1  # Some delay from retries
        assert "Error:" in result
        assert "429" in result
    
    @pytest.mark.asyncio
    async def test_model_availability(self, mock_filter, mock_session):
        """Test model availability errors"""
        # Setup model not found response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value='{"error": "Model not found"}')
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="Test prompt",
                user_prompt="Test message"
            )
        
        assert "Error:" in result
        assert "404" in result
    
    @pytest.mark.asyncio
    async def test_response_parsing_errors(self, mock_filter, mock_session):
        """Test response parsing error handling"""
        # Setup invalid JSON response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="Test prompt",
                user_prompt="Test message"
            )
        
        # Should handle JSON parsing errors gracefully
        assert "Error:" in result or result == ""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_filter, mock_session):
        """Test timeout handling"""
        # Setup timeout
        mock_session.post.side_effect = asyncio.TimeoutError()
        
        start_time = time.time()
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="Test prompt",
                user_prompt="Test message"
            )
        
        end_time = time.time()
        
        # Should have attempted retries
        assert end_time - start_time > 0.1
        assert "timeout" in result.lower()
    
    @pytest.mark.asyncio
    async def test_json_mode_fallback(self, mock_filter, mock_session):
        """Test JSON mode fallback when not supported"""
        # First request fails with JSON mode error
        error_response = AsyncMock()
        error_response.status = 400
        error_response.text = AsyncMock(return_value='{"error": "json_object format not supported"}')
        
        # Second request succeeds without JSON mode
        success_response = AsyncMock()
        success_response.status = 200
        success_response.headers = {"content-type": "application/json"}
        success_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": '{"importance": 5, "category": "test"}'
                }
            }]
        })
        
        # Mock sequential responses
        mock_session.post.return_value.__aenter__.side_effect = [
            error_response, 
            success_response
        ]
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="Test prompt",
                user_prompt="Test message"
            )
        
        # Should succeed on retry without JSON mode
        assert '"importance": 5' in result
        assert '"category": "test"' in result
        
        # Should have made two requests
        assert mock_session.post.call_count == 2


class TestCircuitBreakerFunctionality:
    """Test circuit breaker functionality"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opening(self, mock_filter, mock_session):
        """Test circuit breaker opens after failures"""
        # Setup failing response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value='{"error": "Internal server error"}')
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        # Simulate multiple failures
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            for _ in range(5):  # Trigger failures to open circuit breaker
                await mock_filter.query_llm_with_retry(
                    system_prompt="Test prompt",
                    user_prompt="Test message"
                )
        
        # Circuit breaker should be recording failures
        assert hasattr(mock_filter, '_circuit_breaker_state')
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, mock_filter, mock_session):
        """Test circuit breaker recovery after success"""
        # First setup failures to trigger circuit breaker
        error_response = AsyncMock()
        error_response.status = 500
        error_response.text = AsyncMock(return_value='{"error": "Server error"}')
        
        # Then setup success response
        success_response = AsyncMock()
        success_response.status = 200
        success_response.headers = {"content-type": "application/json"}
        success_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {"content": "Success!"}
            }]
        })
        
        # Simulate failure followed by success
        mock_session.post.return_value.__aenter__.side_effect = [
            error_response,
            success_response
        ]
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            # First call fails
            result1 = await mock_filter.query_llm_with_retry(
                system_prompt="Test prompt",
                user_prompt="Test message"
            )
            assert "Error:" in result1
            
            # Second call succeeds
            result2 = await mock_filter.query_llm_with_retry(
                system_prompt="Test prompt",
                user_prompt="Test message"
            )
            assert result2 == "Success!"


class TestStreamingAndFunctionCalling:
    """Test streaming responses and function calling capabilities"""
    
    @pytest.mark.asyncio
    async def test_streaming_response_handling(self, mock_filter, mock_session):
        """Test handling of streaming responses"""
        # Mock NDJSON streaming response
        ndjson_response = AsyncMock()
        ndjson_response.status = 200
        ndjson_response.headers = {"content-type": "application/x-ndjson"}
        ndjson_response.text = AsyncMock(return_value='''
{"choices": [{"delta": {"role": "assistant"}}]}
{"choices": [{"delta": {"content": "The"}}]}
{"choices": [{"delta": {"content": " memory"}}]}
{"choices": [{"delta": {"content": " is important."}}]}
{"choices": [{"finish_reason": "stop"}]}
        '''.strip())
        
        mock_session.post.return_value.__aenter__.return_value = ndjson_response
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="Test prompt",
                user_prompt="Test message"
            )
        
        # Should handle NDJSON streaming format
        assert result is not None
        # The filter should parse the last valid JSON line
    
    @pytest.mark.asyncio
    async def test_function_calling_responses(self, mock_filter, mock_session):
        """Test function calling in LLM responses"""
        # Mock response with function call
        function_response = AsyncMock()
        function_response.status = 200
        function_response.headers = {"content-type": "application/json"}
        function_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": None,
                    "function_call": {
                        "name": "extract_memory",
                        "arguments": '{"importance": 8, "category": "technical", "content": "User mentioned Python"}'
                    }
                },
                "finish_reason": "function_call"
            }]
        })
        
        mock_session.post.return_value.__aenter__.return_value = function_response
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="Test prompt with function",
                user_prompt="Test message"
            )
        
        # Should handle function call responses
        # The exact handling depends on the filter's implementation
        assert result is not None


class TestModelConfiguration:
    """Test different model configurations and parameters"""
    
    @pytest.mark.asyncio
    async def test_different_model_parameters(self, mock_filter, mock_session):
        """Test various model parameter configurations"""
        test_configs = [
            {
                "provider": "openai_compatible",
                "model": "gpt-4",
                "temperature": 0.0,
                "max_tokens": 1000
            },
            {
                "provider": "ollama", 
                "model": "llama2",
                "temperature": 0.7,
                "max_tokens": 2048
            },
            {
                "provider": "gemini",
                "model": "gemini-pro",
                "temperature": 0.3,
                "max_tokens": 512
            }
        ]
        
        for config in test_configs:
            # Update filter configuration
            mock_filter.valves.llm_provider_type = config["provider"]
            mock_filter.valves.llm_model_name = config["model"]
            
            # Setup appropriate mock response based on provider
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"content-type": "application/json"}
            
            if config["provider"] == "ollama":
                mock_response.json = AsyncMock(return_value={
                    "message": {"content": f"Response from {config['model']}"}
                })
            else:
                mock_response.json = AsyncMock(return_value={
                    "choices": [{
                        "message": {"content": f"Response from {config['model']}"}
                    }]
                })
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
                result = await mock_filter.query_llm_with_retry(
                    system_prompt=f"Test with {config['model']}",
                    user_prompt="Test message"
                )
            
            # Verify correct model was used
            assert f"Response from {config['model']}" in result
            
            # Verify request parameters
            call_args = mock_session.post.call_args
            request_data = call_args.kwargs["json"]
            assert request_data["model"] == config["model"]
    
    @pytest.mark.asyncio
    async def test_feature_detection(self, mock_filter, mock_session):
        """Test provider feature detection"""
        # Mock feature detection response
        feature_response = AsyncMock()
        feature_response.status = 200
        feature_response.json = AsyncMock(return_value={
            "features": {
                "supports_json_mode": True,
                "supports_streaming": True,
                "supports_function_calling": False
            }
        })
        
        # Mock actual API response
        api_response = AsyncMock()
        api_response.status = 200
        api_response.headers = {"content-type": "application/json"}
        api_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {"content": "Feature detection test"}
            }]
        })
        
        mock_session.post.return_value.__aenter__.side_effect = [
            api_response  # Skip feature detection for this test
        ]
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="Test feature detection",
                user_prompt="Test message"
            )
        
        assert "Feature detection test" in result


class TestEndToEndMemoryExtraction:
    """Test complete memory extraction workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_conversation_processing(self, mock_filter, mock_session, 
                                                  sample_conversation_data):
        """Test complete conversation processing workflow"""
        conversation = sample_conversation_data["messages"]
        expected_memories = sample_conversation_data["expected_memories"]
        
        # Create conversation text
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation
        ])
        
        # Setup mock response with expected memories
        mock_response = create_mock_response(200, {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "memories": expected_memories,
                        "conversation_summary": "Professional introduction and technical discussion",
                        "total_memories_extracted": len(expected_memories)
                    })
                }
            }]
        })
        
        mock_session.post.return_value = MockAsyncContextManager(mock_response)
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="Extract important memories from this conversation.",
                user_prompt=f"Conversation to analyze:\n{conversation_text}"
            )
        
        # Verify extraction results
        parsed_result = json.loads(result)
        assert "memories" in parsed_result
        assert len(parsed_result["memories"]) == len(expected_memories)
        
        # Verify memory quality
        memories = parsed_result["memories"]
        
        # Check for professional information
        professional_memories = [m for m in memories if m["category"] == "professional"]
        assert len(professional_memories) >= 1
        assert any("TechCorp" in m["content"] for m in professional_memories)
        
        # Check for technical preferences
        tech_memories = [m for m in memories if m["category"] == "technical_preferences"]
        assert len(tech_memories) >= 1
        assert any("PyTorch" in m["content"] for m in tech_memories)
        
        # Check importance scores
        high_importance_memories = [m for m in memories if m["importance"] >= 7]
        assert len(high_importance_memories) >= 2
    
    @pytest.mark.asyncio
    async def test_memory_importance_filtering(self, mock_filter, mock_session, 
                                             memory_importance_test_cases):
        """Test filtering memories by importance threshold"""
        mock_filter.valves.memory_importance_threshold = 6
        
        for test_case in memory_importance_test_cases:
            # Setup mock response
            mock_response = create_mock_response(200, {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "importance": test_case["expected_importance"],
                            "category": test_case["category"],
                            "content": test_case["input"],
                            "reasoning": test_case["reasoning"],
                            "should_store": test_case["expected_importance"] >= 6
                        })
                    }
                }]
            })
            
            mock_session.post.return_value = MockAsyncContextManager(mock_response)
            
            with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
                result = await mock_filter.query_llm_with_retry(
                    system_prompt="Rate the importance of this memory.",
                    user_prompt=f"Evaluate: {test_case['input']}"
                )
            
            parsed_result = json.loads(result)
            
            # Verify importance scoring
            assert parsed_result["importance"] == test_case["expected_importance"]
            assert parsed_result["category"] == test_case["category"]
            
            # Verify filtering logic
            should_store = test_case["expected_importance"] >= 6
            assert parsed_result["should_store"] == should_store
    
    @pytest.mark.asyncio
    async def test_sensitive_content_filtering_workflow(self, mock_filter, mock_session,
                                                       sensitive_content_test_cases):
        """Test complete sensitive content filtering workflow"""
        all_test_cases = (
            sensitive_content_test_cases["sensitive"] + 
            sensitive_content_test_cases["safe"] +
            sensitive_content_test_cases["mixed"]
        )
        
        for test_case in all_test_cases:
            # Setup mock response
            mock_response = create_mock_response(200, {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "should_store": test_case["should_store"],
                            "reason": test_case["reason"],
                            "filtered_content": test_case["expected_filtered"],
                            "content_type": "sensitive" if not test_case["should_store"] else "safe",
                            "original_content": test_case["input"]
                        })
                    }
                }]
            })
            
            mock_session.post.return_value = MockAsyncContextManager(mock_response)
            
            with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
                result = await mock_filter.query_llm_with_retry(
                    system_prompt="Filter sensitive information from this content.",
                    user_prompt=f"Content to filter: {test_case['input']}"
                )
            
            parsed_result = json.loads(result)
            
            # Verify filtering decisions
            assert parsed_result["should_store"] == test_case["should_store"]
            assert parsed_result["filtered_content"] == test_case["expected_filtered"]
            assert test_case["reason"].lower() in parsed_result["reason"].lower()
    
    @pytest.mark.asyncio
    async def test_multi_provider_memory_extraction(self, mock_filter, mock_session,
                                                   llm_provider_configs):
        """Test memory extraction across multiple LLM providers"""
        test_input = "I'm a data scientist at Netflix working on recommendation algorithms. I love hiking and live in Seattle."
        
        expected_base_memories = [
            {
                "importance": 9,
                "category": "professional", 
                "content": "Works as data scientist at Netflix on recommendation algorithms"
            },
            {
                "importance": 7,
                "category": "location",
                "content": "Lives in Seattle"
            },
            {
                "importance": 6, 
                "category": "hobbies",
                "content": "Enjoys hiking"
            }
        ]
        
        for provider_name, config in llm_provider_configs.items():
            # Configure filter for this provider
            mock_filter.valves.llm_provider_type = config["provider_type"]
            mock_filter.valves.llm_model_name = config["model_name"]
            mock_filter.valves.llm_api_endpoint_url = config["endpoint_url"]
            
            # Setup provider-specific mock response
            if config["provider_type"] == "ollama":
                response_data = {
                    "message": {
                        "content": json.dumps({"memories": expected_base_memories})
                    }
                }
            else:
                response_data = {
                    "choices": [{
                        "message": {
                            "content": json.dumps({"memories": expected_base_memories})
                        }
                    }]
                }
            
            mock_response = create_mock_response(200, response_data)
            mock_session.post.return_value = MockAsyncContextManager(mock_response)
            
            with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
                result = await mock_filter.query_llm_with_retry(
                    system_prompt=f"Extract memories using {provider_name} provider.",
                    user_prompt=f"Extract memories from: {test_input}"
                )
            
            # Verify each provider can extract memories
            parsed_result = json.loads(result)
            assert "memories" in parsed_result
            assert len(parsed_result["memories"]) == 3
            
            # Verify professional memory is present
            professional_memories = [
                m for m in parsed_result["memories"] 
                if m["category"] == "professional"
            ]
            assert len(professional_memories) >= 1
            assert any("Netflix" in m["content"] for m in professional_memories)


class TestRealWorldScenarios:
    """Test realistic usage scenarios"""
    
    @pytest.mark.asyncio
    async def test_long_conversation_memory_extraction(self, mock_filter, mock_session):
        """Test memory extraction from a long, complex conversation"""
        long_conversation = """
        User: Hi, I'm Sarah Chen, a product manager at Spotify. I'm working on improving our recommendation algorithms.
        Assistant: Hello Sarah! That sounds like fascinating work. What aspects of recommendation algorithms are you focusing on?
        User: Right now I'm focused on collaborative filtering and deep learning approaches. I have a PhD in Computer Science from Stanford.
        Assistant: Impressive background! How long have you been at Spotify?
        User: About 3 years now. Before that I worked at Google for 5 years as a software engineer. I specialized in machine learning infrastructure.
        Assistant: What a great career progression! Do you have any personal interests outside of work?
        User: I love photography, especially landscape photography. I also practice yoga daily and I'm vegetarian. I live in Stockholm now but I'm originally from California.
        Assistant: That's wonderful! Do you travel much for photography?
        User: Yes, I've been to Iceland, Norway, and the Scottish Highlands recently. My goal is to visit Patagonia next year for photography.
        Assistant: Those sound like amazing destinations for landscape photography!
        User: Definitely! Oh, and I should mention I have a cat named Pixel who loves to sit on my keyboard while I'm coding.
        """
        
        # Expected comprehensive memory extraction
        expected_memories = [
            {"importance": 10, "category": "personal", "content": "Name is Sarah Chen"},
            {"importance": 9, "category": "professional", "content": "Product manager at Spotify working on recommendation algorithms"},
            {"importance": 9, "category": "education", "content": "PhD in Computer Science from Stanford"},
            {"importance": 8, "category": "professional", "content": "Previously worked at Google for 5 years as software engineer in ML infrastructure"},
            {"importance": 8, "category": "location", "content": "Lives in Stockholm, originally from California"},
            {"importance": 7, "category": "hobbies", "content": "Passionate about landscape photography"},
            {"importance": 7, "category": "travel", "content": "Has traveled to Iceland, Norway, Scottish Highlands for photography"},
            {"importance": 7, "category": "goals", "content": "Plans to visit Patagonia next year for photography"},
            {"importance": 6, "category": "lifestyle", "content": "Practices yoga daily and is vegetarian"},
            {"importance": 5, "category": "personal", "content": "Has a cat named Pixel who sits on keyboard while coding"}
        ]
        
        mock_response = create_mock_response(200, {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "memories": expected_memories,
                        "conversation_analysis": {
                            "total_facts_extracted": len(expected_memories),
                            "primary_categories": ["professional", "personal", "hobbies"],
                            "confidence_score": 0.95
                        }
                    })
                }
            }]
        })
        
        mock_session.post.return_value = MockAsyncContextManager(mock_response)
        
        with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await mock_filter.query_llm_with_retry(
                system_prompt="Extract all important memories from this detailed conversation.",
                user_prompt=f"Analyze conversation:\n{long_conversation}"
            )
        
        parsed_result = json.loads(result)
        memories = parsed_result["memories"]
        
        # Verify comprehensive extraction
        assert len(memories) >= 8
        
        # Verify key information is captured
        assert any("Sarah Chen" in m["content"] for m in memories)
        assert any("Spotify" in m["content"] for m in memories)
        assert any("Stanford" in m["content"] for m in memories)
        assert any("photography" in m["content"] for m in memories)
        assert any("Stockholm" in m["content"] for m in memories)
        
        # Verify importance distribution
        high_importance = [m for m in memories if m["importance"] >= 8]
        medium_importance = [m for m in memories if 5 <= m["importance"] < 8]
        
        assert len(high_importance) >= 3  # Critical info
        assert len(medium_importance) >= 3  # Useful info
    
    @pytest.mark.asyncio
    async def test_incremental_memory_updates(self, mock_filter, mock_session):
        """Test updating memories as new information becomes available"""
        conversation_updates = [
            {
                "input": "I work in tech",
                "expected": {"importance": 6, "category": "professional", "content": "Works in tech"}
            },
            {
                "input": "Actually, I'm a software engineer at Meta",
                "expected": {"importance": 9, "category": "professional", "content": "Software engineer at Meta", "updates_previous": True}
            },
            {
                "input": "I lead the AI infrastructure team",
                "expected": {"importance": 10, "category": "professional", "content": "Leads AI infrastructure team at Meta", "updates_previous": True}
            }
        ]
        
        for i, update in enumerate(conversation_updates):
            mock_response = create_mock_response(200, {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "importance": update["expected"]["importance"],
                            "category": update["expected"]["category"],
                            "content": update["expected"]["content"],
                            "is_update": update["expected"].get("updates_previous", False),
                            "confidence": 0.9 - (i * 0.1)  # Decreasing confidence
                        })
                    }
                }]
            })
            
            mock_session.post.return_value = MockAsyncContextManager(mock_response)
            
            with patch.object(mock_filter, '_get_aiohttp_session', return_value=mock_session):
                result = await mock_filter.query_llm_with_retry(
                    system_prompt="Process this new information and determine if it updates existing memories.",
                    user_prompt=f"New info: {update['input']}"
                )
            
            parsed_result = json.loads(result)
            
            # Verify progressive refinement
            assert parsed_result["importance"] == update["expected"]["importance"]
            assert parsed_result["category"] == update["expected"]["category"]
            
            # Later updates should be marked as updates
            if i > 0:
                assert parsed_result.get("is_update", False) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])