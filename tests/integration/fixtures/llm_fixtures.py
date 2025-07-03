"""
Fixtures for LLM integration testing.

This module provides shared fixtures for testing LLM provider integrations
and memory extraction workflows.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional
import json

from tests.integration.mocks.llm_api_mock import (
    LLMAPIMock,
    OpenAIAPIMock,
    OllamaAPIMock,
    AnthropicAPIMock,
    LLMProvider
)


@pytest.fixture
def sample_conversation_data():
    """Sample conversation data for testing memory extraction"""
    return {
        "messages": [
            {
                "role": "user",
                "content": "Hi there! I'm John and I work as a software engineer at TechCorp."
            },
            {
                "role": "assistant", 
                "content": "Hello John! Nice to meet you. Software engineering is a fascinating field. What kind of projects do you work on at TechCorp?"
            },
            {
                "role": "user",
                "content": "I mainly work on machine learning infrastructure. I'm particularly interested in deep learning frameworks and I prefer using Python and PyTorch."
            },
            {
                "role": "assistant",
                "content": "That sounds very exciting! PyTorch is a great choice for deep learning. Do you have experience with any other frameworks?"
            },
            {
                "role": "user", 
                "content": "Yes, I've also used TensorFlow, but I find PyTorch more intuitive. By the way, I live in San Francisco and love hiking in my free time."
            }
        ],
        "expected_memories": [
            {
                "importance": 9,
                "category": "professional",
                "content": "John works as a software engineer at TechCorp",
                "keywords": ["John", "software engineer", "TechCorp", "work"]
            },
            {
                "importance": 8,
                "category": "technical_preferences",
                "content": "John specializes in machine learning infrastructure and prefers PyTorch over TensorFlow",
                "keywords": ["machine learning", "PyTorch", "TensorFlow", "deep learning"]
            },
            {
                "importance": 7,
                "category": "location",
                "content": "John lives in San Francisco",
                "keywords": ["San Francisco", "location", "lives"]
            },
            {
                "importance": 6,
                "category": "hobbies",
                "content": "John enjoys hiking in his free time", 
                "keywords": ["hiking", "hobbies", "free time"]
            }
        ]
    }


@pytest.fixture
def memory_extraction_prompts():
    """Standard prompts used for memory extraction testing"""
    return {
        "system_prompt": """You are an AI assistant specialized in extracting important information from conversations to create user memories.

Your task is to analyze conversation content and identify information that should be remembered about the user for future interactions. Focus on:

1. Personal information (name, location, job, etc.)
2. Preferences and opinions
3. Technical skills and interests  
4. Goals and aspirations
5. Relationships and social context

For each memory, provide:
- importance: Score from 1-10 (10 = most important)
- category: Type of information (personal, professional, preferences, etc.)
- content: Clear, concise description of what to remember
- keywords: List of relevant search terms

Respond in valid JSON format with a "memories" array containing the extracted memories.""",
        
        "importance_scoring_prompt": """Rate the importance of storing this information as a user memory on a scale of 1-10.

Consider:
- 1-3: Trivial information (greetings, weather chat)
- 4-6: Moderately useful (preferences, casual interests)
- 7-8: Important personal/professional information
- 9-10: Critical identity or professional information

Respond in JSON format with: importance, category, reasoning.""",
        
        "category_classification_prompt": """Classify this memory into appropriate categories.

Available categories:
- personal: Basic personal information (name, age, family)
- professional: Work, career, job-related information
- technical_preferences: Technology choices, programming languages, tools
- food_preferences: Food likes/dislikes, dietary restrictions
- hobbies: Recreational activities and interests
- education: Learning, courses, academic background
- location: Geographic information, places lived/visited
- goals: Aspirations, future plans, objectives
- health: Medical information, fitness preferences
- relationships: Information about family, friends, colleagues

Respond in JSON format with: category, confidence, subcategories.""",
        
        "content_filtering_prompt": """Review this content for sensitive information that should not be stored.

Filter out:
- Social Security Numbers, credit card numbers, passwords
- Private addresses and phone numbers
- Medical record numbers or health IDs  
- Financial account information
- Any other personally identifiable information that could be misused

Respond in JSON format with:
- should_store: boolean indicating if content is safe to store
- reason: explanation of decision
- filtered_content: safe version with sensitive data removed or marked
- content_type: "safe", "sensitive", or "mixed"
"""
    }


@pytest.fixture
def llm_provider_configs():
    """Configuration templates for different LLM providers"""
    return {
        "openai": {
            "provider_type": "openai_compatible",
            "endpoint_url": "https://api.openai.com/v1/chat/completions",
            "model_name": "gpt-4",
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_function_calling": True
        },
        "ollama": {
            "provider_type": "ollama", 
            "endpoint_url": "http://localhost:11434/api/chat",
            "model_name": "llama2",
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_function_calling": False
        },
        "anthropic": {
            "provider_type": "openai_compatible",  # Using OpenAI-compatible endpoint
            "endpoint_url": "https://api.anthropic.com/v1/chat/completions",
            "model_name": "claude-3-sonnet",
            "supports_json_mode": False,  # Anthropic doesn't support JSON mode yet
            "supports_streaming": True,
            "supports_function_calling": False
        },
        "gemini": {
            "provider_type": "gemini",
            "endpoint_url": "https://api.gemini.com/v1/chat/completions", 
            "model_name": "gemini-pro",
            "supports_json_mode": True,
            "supports_streaming": True,
            "supports_function_calling": True
        }
    }


@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing"""
    return {
        "api_key_invalid": {
            "status": 401,
            "response": {"error": "Invalid API key provided"},
            "should_retry": False
        },
        "rate_limited": {
            "status": 429,
            "response": {"error": "Rate limit exceeded", "retry_after": 60},
            "should_retry": True
        },
        "model_not_found": {
            "status": 404,
            "response": {"error": "Model not found"},
            "should_retry": False
        },
        "server_error": {
            "status": 500,
            "response": {"error": "Internal server error"},
            "should_retry": True
        },
        "bad_gateway": {
            "status": 502,
            "response": {"error": "Bad gateway"},
            "should_retry": True
        },
        "service_unavailable": {
            "status": 503,
            "response": {"error": "Service unavailable"},
            "should_retry": True
        },
        "timeout": {
            "status": 504,
            "response": {"error": "Gateway timeout"},
            "should_retry": True
        },
        "json_mode_not_supported": {
            "status": 400,
            "response": {"error": "json_object response format is not supported"},
            "should_retry": True,  # Should retry without JSON mode
            "special_handling": "disable_json_mode"
        },
        "invalid_request_format": {
            "status": 400,
            "response": {"error": "Invalid request format"},
            "should_retry": False
        }
    }


@pytest.fixture
def memory_importance_test_cases():
    """Test cases for memory importance scoring"""
    return [
        {
            "input": "User said hello",
            "expected_importance": 1,
            "category": "greeting",
            "reasoning": "Simple greeting with no lasting value"
        },
        {
            "input": "User mentioned they like coffee",
            "expected_importance": 4,
            "category": "preferences",
            "reasoning": "Minor preference information"
        },
        {
            "input": "User prefers dark mode in all applications",
            "expected_importance": 6,
            "category": "technical_preferences", 
            "reasoning": "Specific technical preference useful for recommendations"
        },
        {
            "input": "User works as a senior data scientist at Google",
            "expected_importance": 9,
            "category": "professional",
            "reasoning": "Critical professional identity information"
        },
        {
            "input": "User is allergic to peanuts",
            "expected_importance": 10,
            "category": "health",
            "reasoning": "Critical health information that could impact safety"
        },
        {
            "input": "User lives in New York City",
            "expected_importance": 8,
            "category": "location",
            "reasoning": "Important location information for context and recommendations"
        },
        {
            "input": "User is learning Spanish and plans to travel to Spain next year",
            "expected_importance": 7,
            "category": "goals",
            "reasoning": "Important learning goal and travel plan"
        }
    ]


@pytest.fixture
def sensitive_content_test_cases():
    """Test cases for sensitive content filtering"""
    return {
        "sensitive": [
            {
                "input": "My social security number is 123-45-6789",
                "expected_filtered": "[SSN REDACTED]",
                "should_store": False,
                "reason": "Contains social security number"
            },
            {
                "input": "My password is mySecretPass123",
                "expected_filtered": "[PASSWORD REDACTED]",
                "should_store": False,
                "reason": "Contains password information"
            },
            {
                "input": "My credit card number is 4111-1111-1111-1111",
                "expected_filtered": "[CREDIT CARD REDACTED]",
                "should_store": False,
                "reason": "Contains credit card information"
            },
            {
                "input": "My phone number is (555) 123-4567 and my address is 123 Main St",
                "expected_filtered": "My phone number is [PHONE REDACTED] and my address is [ADDRESS REDACTED]",
                "should_store": False,
                "reason": "Contains private contact information"
            }
        ],
        "safe": [
            {
                "input": "I enjoy reading science fiction books",
                "expected_filtered": "I enjoy reading science fiction books",
                "should_store": True,
                "reason": "Safe hobby information"
            },
            {
                "input": "My favorite programming language is Python",
                "expected_filtered": "My favorite programming language is Python",
                "should_store": True,
                "reason": "Safe technical preference"
            },
            {
                "input": "I work in software development",
                "expected_filtered": "I work in software development",
                "should_store": True,
                "reason": "Safe professional information"
            }
        ],
        "mixed": [
            {
                "input": "I work at Google (my employee ID is EMP12345) in the AI division",
                "expected_filtered": "I work at Google ([EMPLOYEE ID REDACTED]) in the AI division",
                "should_store": True,
                "reason": "Contains both safe and sensitive information"
            }
        ]
    }


@pytest.fixture
def streaming_response_chunks():
    """Sample streaming response chunks for testing"""
    return [
        {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        },
        {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk", 
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": '{"importance":'},
                "finish_reason": None
            }]
        },
        {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": ' 8, "category":'},
                "finish_reason": None
            }]
        },
        {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": ' "technical"}'},
                "finish_reason": None
            }]
        },
        {
            "id": "chatcmpl-test123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
    ]


@pytest.fixture 
def function_calling_examples():
    """Examples of function calling requests and responses"""
    return {
        "extract_memory_function": {
            "name": "extract_memory",
            "description": "Extract important information to remember about the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "importance": {
                        "type": "integer",
                        "description": "Importance score from 1-10",
                        "minimum": 1,
                        "maximum": 10
                    },
                    "category": {
                        "type": "string",
                        "description": "Category of the memory",
                        "enum": ["personal", "professional", "preferences", "technical", "hobbies"]
                    },
                    "content": {
                        "type": "string",
                        "description": "Content of the memory to store"
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keywords for searching this memory"
                    }
                },
                "required": ["importance", "category", "content", "keywords"]
            }
        },
        "sample_function_call": {
            "name": "extract_memory",
            "arguments": json.dumps({
                "importance": 8,
                "category": "professional",
                "content": "User works as a software engineer at TechCorp",
                "keywords": ["software engineer", "TechCorp", "work", "job"]
            })
        }
    }


class MockAsyncContextManager:
    """Helper class for mocking async context managers"""
    
    def __init__(self, mock_response):
        self.mock_response = mock_response
    
    async def __aenter__(self):
        return self.mock_response
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def create_mock_response(status: int, data: Dict[str, Any], 
                        content_type: str = "application/json") -> AsyncMock:
    """Create a mock HTTP response"""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.headers = {"content-type": content_type}
    
    if status == 200:
        mock_response.json = AsyncMock(return_value=data)
        mock_response.text = AsyncMock(return_value=json.dumps(data))
    else:
        mock_response.text = AsyncMock(return_value=json.dumps(data))
        mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))
    
    return mock_response


def create_ndjson_response(chunks: List[Dict[str, Any]]) -> AsyncMock:
    """Create a mock NDJSON streaming response"""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {"content-type": "application/x-ndjson"}
    
    ndjson_content = "\n".join(json.dumps(chunk) for chunk in chunks)
    mock_response.text = AsyncMock(return_value=ndjson_content)
    
    return mock_response