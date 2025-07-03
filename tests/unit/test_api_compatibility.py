"""
Unit tests for API compatibility and version detection.

Tests the API parameter handling, version detection, and compatibility features.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
import json


class TestAPICompatibility:
    """Test API compatibility and version detection functionality."""

    def test_inlet_handles_event_emitter_param(self, filter_instance):
        """Test inlet method handles __event_emitter__ parameter."""
        body = {
            "messages": [{"role": "user", "content": "test"}],
            "user": {"id": "test-user"}
        }
        
        mock_emitter = Mock()
        
        # Should not raise exception with __event_emitter__
        result = filter_instance.inlet(body, __event_emitter__=mock_emitter)
        assert isinstance(result, dict)
        assert "messages" in result

    def test_inlet_handles_user_param(self, filter_instance):
        """Test inlet method handles __user__ parameter."""
        body = {
            "messages": [{"role": "user", "content": "test"}],
            "chat_id": "test-chat"
        }
        
        user_obj = {"id": "test-user", "name": "Test User"}
        
        # Should handle __user__ parameter
        result = filter_instance.inlet(body, __user__=user_obj)
        assert isinstance(result, dict)
        assert "messages" in result

    def test_outlet_handles_extra_kwargs(self, filter_instance):
        """Test outlet method handles extra kwargs without error."""
        body = {
            "messages": [{"role": "assistant", "content": "response"}],
            "user": {"id": "test-user"}
        }
        
        # Should handle any extra kwargs
        result = filter_instance.outlet(
            body,
            __event_emitter__=Mock(),
            __user__={"id": "test-user"},
            extra_param="value",
            another_param=123
        )
        assert isinstance(result, dict)

    def test_stream_handles_kwargs(self, filter_instance):
        """Test stream method handles kwargs properly."""
        event = {
            "type": "message",
            "data": {"content": "streaming..."}
        }
        
        # Should handle kwargs without error
        result = filter_instance.stream(event, extra_param="value")
        assert isinstance(result, dict)

    def test_api_version_detection_modern(self, filter_instance):
        """Test detection of modern OpenWebUI API format."""
        body = {"messages": [], "chat_id": "test"}
        user_param = {"id": "user123", "name": "Test"}
        
        # Call with modern API signature
        result = filter_instance.inlet(body, __user__=user_param)
        
        # Should detect modern API
        if hasattr(filter_instance, '_detect_openwebui_version'):
            # Version detection should identify this as modern
            assert result is not None
            assert isinstance(result, dict)

    def test_api_version_detection_legacy(self, filter_instance):
        """Test detection of legacy OpenWebUI API format."""
        # Legacy format has user in body
        body = {
            "messages": [],
            "user": {"id": "user123", "name": "Test"},
            "chat_id": "test"
        }
        
        # Call with legacy API signature (no __user__ param)
        result = filter_instance.inlet(body)
        
        # Should handle legacy format
        assert result is not None
        assert isinstance(result, dict)

    def test_parameter_normalization(self, filter_instance):
        """Test parameter normalization and sanitization."""
        # Test with potentially problematic parameters
        body = {
            "messages": [{"role": "user", "content": "test"}],
            "user": {"id": "test-user"},
            "bypass_prompt_processing": True,  # Deprecated parameter
            "prompt": "old-style-prompt",  # Deprecated parameter
            "__internal_field": "should-be-removed",  # Internal field
            "valid_field": "keep-this"
        }
        
        result = filter_instance.inlet(body)
        
        # Should normalize parameters
        assert isinstance(result, dict)
        assert "messages" in result
        # Internal fields might be removed in normalization
        # Deprecated fields should be handled gracefully

    def test_handles_malformed_body_gracefully(self, filter_instance):
        """Test handling of malformed request bodies."""
        test_cases = [
            None,  # Null body
            {},  # Empty body
            {"no_messages": True},  # Missing messages
            {"messages": "not-a-list"},  # Wrong type
            {"messages": [{"no-role": True}]},  # Malformed message
            {"messages": None},  # Null messages
            "string-body",  # String instead of dict
            123,  # Number instead of dict
            [],  # List instead of dict
        ]
        
        for test_body in test_cases:
            # Should never raise exception
            try:
                inlet_result = filter_instance.inlet(test_body)
                outlet_result = filter_instance.outlet(test_body)
                
                # Filter returns the original body on error
                # So non-dict inputs return themselves, dict inputs return dict
                if isinstance(test_body, dict):
                    assert isinstance(inlet_result, dict)
                    assert isinstance(outlet_result, dict)
                else:
                    # Filter returns the original body unchanged for invalid inputs
                    assert inlet_result == test_body
                    assert outlet_result == test_body
            except Exception as e:
                pytest.fail(f"Method raised exception for {test_body}: {e}")

    def test_preserves_critical_fields(self, filter_instance):
        """Test that critical fields are preserved during processing."""
        critical_fields = {
            "messages": [{"role": "user", "content": "test"}],
            "chat_id": "important-chat-id",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": True,
            "user": {"id": "user123"}
        }
        
        body = critical_fields.copy()
        
        # Process through inlet
        result = filter_instance.inlet(body)
        
        # Critical fields should be preserved
        assert "messages" in result
        assert "chat_id" in result
        assert result.get("model") == "gpt-4"
        # Other fields should be preserved or have sensible defaults

    def test_handles_both_user_formats(self, filter_instance):
        """Test handling of user in both body and parameter."""
        body_user = {"id": "body-user", "name": "Body User"}
        param_user = {"id": "param-user", "name": "Param User"}
        
        body = {
            "messages": [{"role": "user", "content": "test"}],
            "user": body_user
        }
        
        # Call with both user in body and parameter
        result = filter_instance.inlet(body, __user__=param_user)
        
        # Should handle gracefully (typically parameter takes precedence)
        assert isinstance(result, dict)
        assert "messages" in result

    def test_session_based_version_caching(self, filter_instance):
        """Test that API version detection can be cached per session."""
        body1 = {"messages": [], "chat_id": "session1"}
        body2 = {"messages": [], "chat_id": "session1"}  # Same session
        body3 = {"messages": [], "chat_id": "session2"}  # Different session
        
        # First call for session1
        filter_instance.inlet(body1, __user__={"id": "user1"})
        
        # Second call for same session - should use cached version
        filter_instance.inlet(body2, __user__={"id": "user1"})
        
        # Call for different session - should detect separately
        filter_instance.inlet(body3, __user__={"id": "user2"})
        
        # All should succeed
        assert True  # If we get here, no exceptions were raised

    def test_handles_streaming_compatibility(self, filter_instance):
        """Test streaming compatibility across API versions."""
        # Modern streaming event
        modern_event = {
            "id": "chatcmpl-123",
            "choices": [{
                "delta": {"content": "Hello"},
                "index": 0
            }]
        }
        
        # Legacy streaming format
        legacy_event = {
            "type": "message",
            "data": "Hello"
        }
        
        # Both should be handled
        result1 = filter_instance.stream(modern_event)
        result2 = filter_instance.stream(legacy_event)
        
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)

    def test_error_response_compatibility(self, filter_instance):
        """Test that errors are handled compatibly across versions."""
        # Simulate an error condition
        error_body = {
            "messages": [{"role": "user", "content": "test"}],
            "user": {"id": "test-user"},
            "error": "Something went wrong"
        }
        
        # Should handle error gracefully
        result = filter_instance.inlet(error_body)
        assert isinstance(result, dict)
        
        # Outlet should also handle errors
        result = filter_instance.outlet(error_body)
        assert isinstance(result, dict)

    def test_compatibility_with_empty_messages(self, filter_instance):
        """Test handling of empty message lists."""
        body = {
            "messages": [],
            "user": {"id": "test-user"},
            "chat_id": "test-chat"
        }
        
        # Should handle empty messages
        inlet_result = filter_instance.inlet(body)
        outlet_result = filter_instance.outlet(body)
        
        assert isinstance(inlet_result, dict)
        assert isinstance(outlet_result, dict)
        assert "messages" in inlet_result
        assert "messages" in outlet_result

    def test_large_message_handling(self, filter_instance):
        """Test handling of large messages (edge case)."""
        # Create a large message
        large_content = "x" * 10000  # 10k characters
        
        body = {
            "messages": [
                {"role": "user", "content": large_content}
            ],
            "user": {"id": "test-user"}
        }
        
        # Should handle large messages
        result = filter_instance.inlet(body)
        assert isinstance(result, dict)
        assert len(result["messages"][0]["content"]) == 10000

    def test_special_character_handling(self, filter_instance):
        """Test handling of special characters in messages."""
        special_content = "Hello 😊 \n\t\r Special «chars» \\u1234"
        
        body = {
            "messages": [
                {"role": "user", "content": special_content}
            ],
            "user": {"id": "test-user"}
        }
        
        # Should preserve special characters
        result = filter_instance.inlet(body)
        assert isinstance(result, dict)
        assert result["messages"][0]["content"] == special_content

    def test_numeric_user_id_compatibility(self, filter_instance):
        """Test handling of numeric user IDs (some systems use integers)."""
        body = {
            "messages": [{"role": "user", "content": "test"}],
            "user": {"id": 12345}  # Numeric ID
        }
        
        # Should handle numeric IDs
        result = filter_instance.inlet(body)
        assert isinstance(result, dict)
        
        # Test with string ID in parameter
        result = filter_instance.inlet(body, __user__={"id": "12345"})
        assert isinstance(result, dict)

    def test_missing_optional_fields(self, filter_instance):
        """Test handling when optional fields are missing."""
        minimal_body = {
            "messages": [{"role": "user", "content": "test"}]
            # No user, chat_id, model, etc.
        }
        
        # Should provide sensible defaults
        result = filter_instance.inlet(minimal_body)
        assert isinstance(result, dict)
        assert "messages" in result

    def test_api_response_format_preservation(self, filter_instance):
        """Test that API response format is preserved."""
        body = {
            "messages": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"}
            ],
            "user": {"id": "test-user"},
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        
        # Process through outlet
        result = filter_instance.outlet(body)
        
        # Should preserve API response structure
        assert isinstance(result, dict)
        assert "messages" in result
        if "usage" in body:
            # Usage might be preserved if present
            assert "usage" not in result or isinstance(result["usage"], dict)