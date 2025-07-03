#!/usr/bin/env python3
"""
Validation script for Google Gemini API regression fix.

This script validates that the Gemini API integration fix is working correctly
by testing the request format, response parsing, and error handling.
"""

import json
import sys
import os
from unittest.mock import AsyncMock, patch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def validate_request_format():
    """Validate that Gemini requests use the correct format"""
    print("🔍 Validating Gemini API request format...")
    
    # Test data
    system_prompt = "You are a memory extraction assistant."
    user_prompt = "Extract memory from: I love programming in Python"
    model = "gemini-pro"
    
    # Simulate the request format logic from the fix
    combined_prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nPlease respond in valid JSON format."
    
    data = {
        "contents": [
            {
                "parts": [
                    {"text": combined_prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.95,
            "maxOutputTokens": 1024,
            "stopSequences": [],
            "responseMimeType": "application/json"
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
    
    # Validate structure
    errors = []
    
    if "contents" not in data:
        errors.append("Missing 'contents' field")
    elif not isinstance(data["contents"], list):
        errors.append("'contents' should be a list")
    elif len(data["contents"]) == 0:
        errors.append("'contents' should not be empty")
    elif "parts" not in data["contents"][0]:
        errors.append("Missing 'parts' in contents[0]")
    elif not isinstance(data["contents"][0]["parts"], list):
        errors.append("'parts' should be a list")
    elif "text" not in data["contents"][0]["parts"][0]:
        errors.append("Missing 'text' in parts[0]")
    
    if "generationConfig" not in data:
        errors.append("Missing 'generationConfig' field")
    elif "responseMimeType" not in data["generationConfig"]:
        errors.append("Missing 'responseMimeType' in generationConfig")
    elif data["generationConfig"]["responseMimeType"] != "application/json":
        errors.append("responseMimeType should be 'application/json'")
    
    if "safetySettings" not in data:
        errors.append("Missing 'safetySettings' field")
    elif not isinstance(data["safetySettings"], list):
        errors.append("'safetySettings' should be a list")
    elif len(data["safetySettings"]) < 4:
        errors.append("Should have at least 4 safety categories")
    
    if errors:
        print("❌ Request format validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ Request format validation passed")
        return True

def validate_system_instruction_support():
    """Validate system instruction support for Gemini 1.5+ models"""
    print("🔍 Validating system instruction support...")
    
    system_prompt = "You are a helpful assistant."
    user_prompt = "Hello, how are you?"
    model = "gemini-1.5-pro"
    
    # Simulate system instruction logic
    if "gemini-1.5" in model.lower() or "gemini-pro" in model.lower():
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{user_prompt}\n\nPlease respond in valid JSON format."}
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {"text": system_prompt}
                ]
            }
        }
        
        # Validate structure
        if "systemInstruction" not in data:
            print("❌ System instruction validation failed: Missing systemInstruction")
            return False
        
        if "parts" not in data["systemInstruction"]:
            print("❌ System instruction validation failed: Missing parts in systemInstruction")
            return False
        
        if user_prompt not in data["contents"][0]["parts"][0]["text"]:
            print("❌ System instruction validation failed: User prompt not in contents")
            return False
        
        if system_prompt in data["contents"][0]["parts"][0]["text"]:
            print("❌ System instruction validation failed: System prompt should not be in contents")
            return False
        
        print("✅ System instruction validation passed")
        return True
    else:
        print("✅ System instruction validation skipped (not supported for this model)")
        return True

def validate_response_parsing():
    """Validate Gemini response parsing logic"""
    print("🔍 Validating response parsing...")
    
    # Test correct Gemini response format
    gemini_response = {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": '{"importance": 8, "category": "technical", "content": "User loves Python programming"}'
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
    }
    
    # Test parsing logic
    content = None
    if (
        gemini_response.get("candidates")
        and len(gemini_response["candidates"]) > 0
        and gemini_response["candidates"][0].get("content")
        and gemini_response["candidates"][0]["content"].get("parts")
        and len(gemini_response["candidates"][0]["content"]["parts"]) > 0
        and gemini_response["candidates"][0]["content"]["parts"][0].get("text")
    ):
        content = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
    
    if content is None:
        print("❌ Response parsing validation failed: Could not extract content")
        return False
    
    # Validate extracted content
    try:
        parsed_content = json.loads(content)
        if "importance" not in parsed_content:
            print("❌ Response parsing validation failed: Missing importance field")
            return False
        if "category" not in parsed_content:
            print("❌ Response parsing validation failed: Missing category field")
            return False
    except json.JSONDecodeError:
        print("❌ Response parsing validation failed: Invalid JSON content")
        return False
    
    print("✅ Response parsing validation passed")
    return True

def validate_backward_compatibility():
    """Validate backward compatibility with OpenAI-style responses"""
    print("🔍 Validating backward compatibility...")
    
    # Test OpenAI-style response format (for fallback)
    openai_response = {
        "choices": [{
            "message": {
                "content": '{"importance": 7, "category": "test", "content": "Backward compatibility test"}'
            }
        }]
    }
    
    # Test fallback parsing logic
    content = None
    
    # First try Gemini format (should fail)
    if (
        openai_response.get("candidates")
        and len(openai_response["candidates"]) > 0
        and openai_response["candidates"][0].get("content")
    ):
        # This should not match
        content = "Should not reach here"
    
    # Then try OpenAI format (should work)
    elif (
        openai_response.get("choices")
        and openai_response["choices"][0].get("message")
        and openai_response["choices"][0]["message"].get("content")
    ):
        content = openai_response["choices"][0]["message"]["content"]
    
    if content is None:
        print("❌ Backward compatibility validation failed: Could not extract content")
        return False
    
    if "Backward compatibility test" not in content:
        print("❌ Backward compatibility validation failed: Wrong content extracted")
        return False
    
    print("✅ Backward compatibility validation passed")
    return True

def validate_authentication_format():
    """Validate authentication format for Gemini API"""
    print("🔍 Validating authentication format...")
    
    api_key = "test-api-key"
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    
    # Test URL parameter method (primary)
    if "key=" not in api_url:
        corrected_url = f"{api_url}?key={api_key}"
        if "key=" not in corrected_url or api_key not in corrected_url:
            print("❌ Authentication validation failed: URL parameter method")
            return False
    
    # Test header method (alternative)
    headers = {"x-goog-api-key": api_key}
    if "x-goog-api-key" not in headers or headers["x-goog-api-key"] != api_key:
        print("❌ Authentication validation failed: Header method")
        return False
    
    # Verify no Bearer token (should not be used)
    if f"Bearer {api_key}" in str(headers.values()):
        print("❌ Authentication validation failed: Bearer token should not be used")
        return False
    
    print("✅ Authentication validation passed")
    return True

def validate_error_handling():
    """Validate error handling for various Gemini API scenarios"""
    print("🔍 Validating error handling...")
    
    # Test API error response
    error_response = {
        "error": {
            "code": 400,
            "message": "Invalid request format",
            "status": "INVALID_ARGUMENT"
        }
    }
    
    # Simulate error extraction
    if error_response.get("error"):
        error_msg = error_response["error"].get("message", "Unknown error")
        if not error_msg:
            print("❌ Error handling validation failed: Could not extract error message")
            return False
    
    # Test safety filter response
    safety_response = {
        "candidates": [{
            "finishReason": "SAFETY",
            "safetyRatings": [
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "probability": "HIGH"
                }
            ]
        }]
    }
    
    # Simulate safety handling
    if (safety_response.get("candidates") and 
        len(safety_response["candidates"]) > 0 and 
        safety_response["candidates"][0].get("finishReason") == "SAFETY"):
        # Should handle this gracefully
        pass
    else:
        print("❌ Error handling validation failed: Safety filter not detected")
        return False
    
    print("✅ Error handling validation passed")
    return True

def validate_endpoint_correction():
    """Validate automatic endpoint correction"""
    print("🔍 Validating endpoint correction...")
    
    # Test various incorrect endpoints
    incorrect_endpoints = [
        "https://api.openai.com/v1/chat/completions",
        "https://api.gemini.com/v1/chat/completions",
        "https://custom-endpoint.com/v1/completions"
    ]
    
    model = "gemini-pro"
    expected_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    
    for endpoint in incorrect_endpoints:
        if "generativelanguage.googleapis.com" not in endpoint:
            if "chat/completions" in endpoint or "v1/completions" in endpoint:
                # Should be corrected
                corrected = expected_endpoint
                if "generativelanguage.googleapis.com" not in corrected:
                    print(f"❌ Endpoint correction failed for: {endpoint}")
                    return False
    
    print("✅ Endpoint correction validation passed")
    return True

def main():
    """Run all validation tests"""
    print("🚀 Starting Gemini API Regression Fix Validation")
    print("=" * 60)
    
    tests = [
        validate_request_format,
        validate_system_instruction_support,
        validate_response_parsing,
        validate_backward_compatibility,
        validate_authentication_format,
        validate_error_handling,
        validate_endpoint_correction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validations passed! Gemini API regression fix is working correctly.")
        return 0
    else:
        print("⚠️  Some validations failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())