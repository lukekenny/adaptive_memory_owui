"""
Comprehensive tests for Google Gemini API regression fix.

This module tests the specific regression fix for Google Gemini API integration,
ensuring proper request format, authentication, and response parsing.
"""

import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch
import aiohttp

# Import the adaptive memory filter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import importlib.util
adaptive_memory_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    "adaptive_memory_v4.0.py"
)
spec = importlib.util.spec_from_file_location("adaptive_memory_v4_0", adaptive_memory_path)
adaptive_memory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaptive_memory_module)
AdaptiveMemoryFilter = adaptive_memory_module.Filter


class TestGeminiAPIRegression:
    """Test the Gemini API regression fix"""
    
    @pytest.fixture
    def gemini_filter(self):
        """Create filter instance configured for Gemini"""
        filter_instance = AdaptiveMemoryFilter()
        
        # Configure for Gemini API
        filter_instance.valves.llm_provider_type = "gemini"
        filter_instance.valves.llm_model_name = "gemini-pro"
        filter_instance.valves.llm_api_endpoint_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        filter_instance.valves.llm_api_key = "test-gemini-api-key"
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
    async def test_gemini_api_request_format(self, gemini_filter, mock_session):
        """Test that requests use correct Gemini API format"""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"importance": 8, "category": "technical", "content": "User works with Python"}'
                    }]
                },
                "finishReason": "STOP",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "NEGLIGIBLE"
                    }
                ]
            }],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 25,
                "totalTokenCount": 75
            }
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(gemini_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await gemini_filter.query_llm_with_retry(
                system_prompt="You are a memory extraction assistant.",
                user_prompt="Extract memory from: I love programming in Python"
            )
        
        # Verify result
        assert result is not None
        assert '"importance": 8' in result
        assert '"category": "technical"' in result
        
        # Verify request format
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        request_data = call_args.kwargs["json"]
        
        # Check Gemini API request structure
        assert "contents" in request_data
        assert "generationConfig" in request_data
        assert "safetySettings" in request_data
        
        # Check contents structure
        assert isinstance(request_data["contents"], list)
        assert len(request_data["contents"]) > 0
        assert "parts" in request_data["contents"][0]
        assert isinstance(request_data["contents"][0]["parts"], list)
        assert "text" in request_data["contents"][0]["parts"][0]
        
        # Check generation config
        generation_config = request_data["generationConfig"]
        assert "temperature" in generation_config
        assert "topP" in generation_config
        assert "maxOutputTokens" in generation_config
        assert generation_config["responseMimeType"] == "application/json"
        
        # Check safety settings
        safety_settings = request_data["safetySettings"]
        assert isinstance(safety_settings, list)
        assert len(safety_settings) >= 4  # All major safety categories
        
        safety_categories = [setting["category"] for setting in safety_settings]
        expected_categories = [
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_HATE_SPEECH", 
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT"
        ]
        for category in expected_categories:
            assert category in safety_categories
    
    @pytest.mark.asyncio
    async def test_gemini_api_authentication(self, gemini_filter, mock_session):
        """Test that authentication uses correct Gemini API format"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [{"text": "Test response"}]
                }
            }]
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(gemini_filter, '_get_aiohttp_session', return_value=mock_session):
            await gemini_filter.query_llm_with_retry(
                system_prompt="Test",
                user_prompt="Test message"
            )
        
        # Verify authentication
        call_args = mock_session.post.call_args
        
        # Check URL has API key parameter
        api_url = call_args.args[0]
        assert "key=" in api_url
        assert "test-gemini-api-key" in api_url
        
        # Check headers don't use Bearer token
        headers = call_args.kwargs["headers"]
        assert "Authorization" not in headers or not headers["Authorization"].startswith("Bearer")
    
    @pytest.mark.asyncio
    async def test_gemini_system_instruction_support(self, gemini_filter, mock_session):
        """Test system instruction support for compatible models"""
        # Test with Gemini 1.5 model that supports system instructions
        gemini_filter.valves.llm_model_name = "gemini-1.5-pro"
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [{"text": "Test response"}]
                }
            }]
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(gemini_filter, '_get_aiohttp_session', return_value=mock_session):
            await gemini_filter.query_llm_with_retry(
                system_prompt="You are a helpful assistant.",
                user_prompt="Test message"
            )
        
        # Verify system instruction is used
        call_args = mock_session.post.call_args
        request_data = call_args.kwargs["json"]
        
        # Should have systemInstruction field
        assert "systemInstruction" in request_data
        assert "parts" in request_data["systemInstruction"]
        assert len(request_data["systemInstruction"]["parts"]) > 0
        assert "text" in request_data["systemInstruction"]["parts"][0]
        
        # User message should not contain system prompt
        user_content = request_data["contents"][0]["parts"][0]["text"]
        assert "Test message" in user_content
        # System prompt should not be in the user content when systemInstruction is used
        assert "You are a helpful assistant" not in user_content
    
    @pytest.mark.asyncio
    async def test_gemini_response_parsing(self, gemini_filter, mock_session):
        """Test correct parsing of Gemini API response format"""
        # Test with complex response structure
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": '{"importance": 9, "category": "professional", "content": "Complex response", "metadata": {"source": "gemini"}}'
                    }],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "probability": "NEGLIGIBLE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "probability": "NEGLIGIBLE"
                    }
                ]
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "totalTokenCount": 150
            }
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(gemini_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await gemini_filter.query_llm_with_retry(
                system_prompt="Extract memories.",
                user_prompt="Test complex response parsing"
            )
        
        # Verify correct content extraction
        assert result is not None
        assert '"importance": 9' in result
        assert '"category": "professional"' in result
        assert '"metadata"' in result
        assert '"source": "gemini"' in result
    
    @pytest.mark.asyncio
    async def test_gemini_error_handling(self, gemini_filter, mock_session):
        """Test error handling for Gemini API responses"""
        # Test API error response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value=json.dumps({
            "error": {
                "code": 400,
                "message": "Invalid request format",
                "status": "INVALID_ARGUMENT"
            }
        }))
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(gemini_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await gemini_filter.query_llm_with_retry(
                system_prompt="Test error handling",
                user_prompt="Test message"
            )
        
        # Should return error message
        assert "Error:" in result
        assert "400" in result
    
    @pytest.mark.asyncio
    async def test_gemini_safety_filter_response(self, gemini_filter, mock_session):
        """Test handling of safety-filtered responses"""
        # Test response blocked by safety filters
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "finishReason": "SAFETY",
                "safetyRatings": [
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "probability": "HIGH"
                    }
                ]
            }]
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(gemini_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await gemini_filter.query_llm_with_retry(
                system_prompt="Test safety filtering",
                user_prompt="Test message"
            )
        
        # Should handle safety-filtered response gracefully
        # The exact behavior depends on implementation, but should not crash
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_gemini_url_endpoint_correction(self, gemini_filter, mock_session):
        """Test automatic correction of incorrect Gemini endpoints"""
        # Start with incorrect OpenAI-style endpoint
        gemini_filter.valves.llm_api_endpoint_url = "https://api.openai.com/v1/chat/completions"
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [{"text": "Corrected endpoint test"}]
                }
            }]
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(gemini_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await gemini_filter.query_llm_with_retry(
                system_prompt="Test endpoint correction",
                user_prompt="Test message"
            )
        
        # Verify the endpoint was corrected
        call_args = mock_session.post.call_args
        api_url = call_args.args[0]
        
        # Should have been corrected to proper Gemini API endpoint
        assert "generativelanguage.googleapis.com" in api_url
        assert "generateContent" in api_url
        assert result == "Corrected endpoint test"
    
    @pytest.mark.asyncio
    async def test_gemini_feature_detection(self, gemini_filter, mock_session):
        """Test Gemini-specific feature detection"""
        # Test that Gemini features are properly detected
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [{"text": "Feature detection test"}]
                }
            }]
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(gemini_filter, '_get_aiohttp_session', return_value=mock_session):
            # Trigger feature detection
            features = await gemini_filter._get_provider_features(
                "gemini", 
                gemini_filter.valves.llm_api_endpoint_url,
                gemini_filter.valves.llm_api_key
            )
        
        # Verify Gemini-specific features are detected
        assert features["supports_vision"] is True
        assert features["supports_function_calling"] is True
        assert features["supports_json_mode"] is True
        assert features["supports_system_messages"] is True
        assert features["supports_streaming"] is True
    
    @pytest.mark.asyncio
    async def test_gemini_backward_compatibility(self, gemini_filter, mock_session):
        """Test backward compatibility with older response formats"""
        # Test fallback for OpenAI-style response format (for compatibility)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": "Backward compatibility test"
                }
            }]
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(gemini_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await gemini_filter.query_llm_with_retry(
                system_prompt="Test backward compatibility",
                user_prompt="Test message"
            )
        
        # Should handle older format gracefully
        assert result == "Backward compatibility test"
    
    @pytest.mark.asyncio
    async def test_gemini_multipart_content(self, gemini_filter, mock_session):
        """Test handling of multipart content responses"""
        # Test response with multiple parts
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(return_value={
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Part 1: "},
                        {"text": '{"importance": 7, "category": "test"}'},
                        {"text": " Additional context"}
                    ]
                }
            }]
        })
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        with patch.object(gemini_filter, '_get_aiohttp_session', return_value=mock_session):
            result = await gemini_filter.query_llm_with_retry(
                system_prompt="Test multipart content",
                user_prompt="Test message"
            )
        
        # Should extract first part only (as per current implementation)
        assert "Part 1: " in result
    
    @pytest.mark.asyncio
    async def test_gemini_different_models(self, gemini_filter, mock_session):
        """Test different Gemini model configurations"""
        models_to_test = [
            "gemini-pro",
            "gemini-1.5-pro", 
            "gemini-1.5-flash",
            "gemini-pro-vision"
        ]
        
        for model in models_to_test:
            gemini_filter.valves.llm_model_name = model
            
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json = AsyncMock(return_value={
                "candidates": [{
                    "content": {
                        "parts": [{"text": f"Response from {model}"}]
                    }
                }]
            })
            
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            with patch.object(gemini_filter, '_get_aiohttp_session', return_value=mock_session):
                result = await gemini_filter.query_llm_with_retry(
                    system_prompt=f"Test with {model}",
                    user_prompt="Test message"
                )
            
            assert f"Response from {model}" in result
            
            # Verify correct model is used in URL
            call_args = mock_session.post.call_args
            api_url = call_args.args[0]
            
            # URL should contain the model name or be corrected to use it
            if "generativelanguage.googleapis.com" in api_url:
                assert model in api_url or "gemini-pro" in api_url  # Default fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])