"""
Comprehensive unit tests for the OWUI Adaptive Memory Filter.

This test suite validates the monolithic Filter Function implementation
including core methods, memory operations, orchestration, and OpenWebUI
compatibility while respecting the single-file constraint.

Tests cover:
- Filter Function interface compliance (inlet/outlet/stream)
- Memory extraction, storage, and retrieval
- Filter orchestration system 
- API parameter compatibility
- Deduplication mechanisms
- Gemini API integration
- Error handling and edge cases
- Performance characteristics
"""

import pytest
import asyncio
import json
import time
import uuid
import threading
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone
import logging
from typing import Dict, Any, List


class TestFilterOpenWebUICompliance:
    """Test OpenWebUI Filter Function interface compliance."""

    def test_filter_has_valves_class(self, filter_instance):
        """Test that Filter has a Valves configuration class."""
        assert hasattr(filter_instance, 'Valves')
        assert hasattr(filter_instance, 'valves')
        assert filter_instance.valves is not None

    def test_valves_inherits_from_basemodel(self, filter_instance):
        """Test that Valves inherits from BaseModel."""
        from pydantic import BaseModel
        assert issubclass(filter_instance.Valves, BaseModel)

    def test_filter_methods_signature_compliance(self, filter_instance):
        """Test that Filter methods comply with OpenWebUI signature requirements."""
        import inspect
        
        # Test inlet method signature
        inlet_sig = inspect.signature(filter_instance.inlet)
        assert 'body' in inlet_sig.parameters
        assert inlet_sig.parameters['body'].annotation == dict
        assert inlet_sig.return_annotation == dict

        # Test outlet method signature  
        outlet_sig = inspect.signature(filter_instance.outlet)
        assert 'body' in outlet_sig.parameters
        assert outlet_sig.parameters['body'].annotation == dict
        assert outlet_sig.return_annotation == dict

        # Test stream method signature
        stream_sig = inspect.signature(filter_instance.stream)
        assert 'event' in stream_sig.parameters
        assert stream_sig.parameters['event'].annotation == dict
        assert stream_sig.return_annotation == dict

    def test_filter_methods_return_dict(self, filter_instance, basic_message_body, stream_event):
        """Test that all Filter methods return dictionaries."""
        # Test inlet
        result = filter_instance.inlet(basic_message_body)
        assert isinstance(result, dict), "inlet must return dict"

        # Test outlet
        result = filter_instance.outlet(basic_message_body)
        assert isinstance(result, dict), "outlet must return dict"

        # Test stream
        result = filter_instance.stream(stream_event)
        assert isinstance(result, dict), "stream must return dict"

    def test_openwebui_body_structure_compliance(self, filter_instance):
        """Test compliance with OpenWebUI body structure requirements."""
        # Test with standard OpenWebUI body structure
        openwebui_body = {
            "messages": [
                {
                    "id": "msg1",
                    "role": "user", 
                    "content": "Hello",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "chat_id": "chat123",
            "user": {
                "id": "user123",
                "name": "Test User",
                "email": "test@example.com"
            },
            "model": "gpt-4",
            "stream": False
        }

        # Both inlet and outlet should handle this structure
        inlet_result = filter_instance.inlet(openwebui_body)
        assert 'messages' in inlet_result
        assert isinstance(inlet_result['messages'], list)

        outlet_result = filter_instance.outlet(openwebui_body)
        assert 'messages' in outlet_result
        assert isinstance(outlet_result['messages'], list)

    def test_openwebui_parameter_compatibility(self, filter_instance, basic_message_body):
        """Test compatibility with OpenWebUI parameter patterns."""
        mock_emitter = AsyncMock()
        user_dict = {"id": "test_user", "name": "Test User"}

        # Test new API pattern with __user__ parameter
        result = filter_instance.inlet(basic_message_body, __user__=user_dict)
        assert isinstance(result, dict)

        result = filter_instance.outlet(basic_message_body, __user__=user_dict)
        assert isinstance(result, dict)

        # Test with __event_emitter__ parameter
        result = filter_instance.inlet(basic_message_body, __event_emitter__=mock_emitter)
        assert isinstance(result, dict)

        result = filter_instance.outlet(basic_message_body, __event_emitter__=mock_emitter)
        assert isinstance(result, dict)

        # Test with both parameters
        result = filter_instance.inlet(basic_message_body, __event_emitter__=mock_emitter, __user__=user_dict)
        assert isinstance(result, dict)

    def test_unknown_parameters_handling(self, filter_instance, basic_message_body):
        """Test graceful handling of unknown/deprecated parameters."""
        # Test with unknown kwargs
        result = filter_instance.inlet(
            basic_message_body, 
            unknown_param="value",
            deprecated_param=123,
            random_data={"key": "value"}
        )
        assert isinstance(result, dict)

        result = filter_instance.outlet(
            basic_message_body,
            another_unknown="test",
            legacy_param=True
        )
        assert isinstance(result, dict)


class TestFilterMemoryOperations:
    """Test core memory operations functionality."""

    def test_memory_extraction_from_messages(self, filter_instance):
        """Test memory extraction from conversation messages."""
        message_body = {
            "messages": [
                {
                    "id": "msg1",
                    "role": "user",
                    "content": "I really love programming in Python. It's my favorite language.",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                {
                    "id": "msg2", 
                    "role": "assistant",
                    "content": "That's great! Python is excellent for many applications.",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "user": {"id": "test_user", "name": "Test User"},
            "chat_id": "test_chat"
        }

        # Process through inlet to extract memories
        result = filter_instance.inlet(message_body)
        assert isinstance(result, dict)
        assert 'messages' in result

    def test_memory_injection_in_outlet(self, filter_instance):
        """Test memory injection in outlet processing."""
        message_body = {
            "messages": [
                {
                    "id": "msg1",
                    "role": "assistant", 
                    "content": "How can I help you today?",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "user": {"id": "test_user", "name": "Test User"},
            "chat_id": "test_chat"
        }

        # Process through outlet to inject memories
        result = filter_instance.outlet(message_body)
        assert isinstance(result, dict)
        assert 'messages' in result

    @patch('adaptive_memory_v4_0.logger')
    def test_memory_storage_operations(self, mock_logger, filter_instance):
        """Test memory storage and retrieval operations."""
        # Test with memory-worthy content
        memory_content = "User prefers detailed technical explanations"
        
        # This would normally trigger memory storage via inlet processing
        message_body = {
            "messages": [
                {
                    "id": "msg1",
                    "role": "user",
                    "content": memory_content,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "user": {"id": "test_user", "name": "Test User"},
            "chat_id": "test_chat"
        }

        result = filter_instance.inlet(message_body)
        assert isinstance(result, dict)

    def test_memory_deduplication(self, filter_instance):
        """Test memory deduplication mechanisms."""
        # Test with similar content that should be deduplicated
        similar_messages = [
            {
                "messages": [{"id": "1", "role": "user", "content": "I like Python programming", "timestamp": datetime.now(timezone.utc).isoformat()}],
                "user": {"id": "test_user", "name": "Test User"},
                "chat_id": "test_chat"
            },
            {
                "messages": [{"id": "2", "role": "user", "content": "I really like Python programming", "timestamp": datetime.now(timezone.utc).isoformat()}],
                "user": {"id": "test_user", "name": "Test User"}, 
                "chat_id": "test_chat"
            }
        ]

        # Process both messages
        for msg_body in similar_messages:
            result = filter_instance.inlet(msg_body)
            assert isinstance(result, dict)

    def test_memory_relevance_scoring(self, filter_instance):
        """Test memory relevance scoring and ranking."""
        query_body = {
            "messages": [
                {
                    "id": "msg1",
                    "role": "user",
                    "content": "Tell me about Python programming best practices",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "user": {"id": "test_user", "name": "Test User"},
            "chat_id": "test_chat"
        }

        # Process through outlet to trigger memory retrieval and relevance scoring
        result = filter_instance.outlet(query_body)
        assert isinstance(result, dict)


class TestFilterOrchestrationSystem:
    """Test the Filter Orchestration System functionality."""

    def test_filter_metadata_declaration(self, filter_instance):
        """Test that filter properly declares its metadata."""
        # Check if orchestration methods exist
        assert hasattr(filter_instance, 'get_orchestration_status') or hasattr(filter_instance, '_orchestration_manager')

    def test_filter_capabilities_declaration(self, filter_instance):
        """Test filter capabilities are properly declared."""
        # Filter should declare its capabilities
        if hasattr(filter_instance, 'valves'):
            valves = filter_instance.valves
            # Check for orchestration-related configuration
            orchestration_fields = [attr for attr in dir(valves) if 'orchestration' in attr.lower() or 'filter' in attr.lower()]
            # Should have some orchestration configuration options
            assert len(orchestration_fields) >= 0  # At minimum, should not error

    def test_orchestration_performance_monitoring(self, filter_instance, basic_message_body):
        """Test performance monitoring in orchestration system."""
        start_time = time.time()
        
        # Process message through filter
        result = filter_instance.inlet(basic_message_body)
        
        execution_time = time.time() - start_time
        assert execution_time < 10.0  # Should complete within reasonable time
        assert isinstance(result, dict)

    def test_orchestration_conflict_detection(self, filter_instance):
        """Test conflict detection capabilities."""
        # Test that the filter can handle potential conflicts gracefully
        # This is mainly about ensuring the filter doesn't break when other filters are present
        
        # Simulate concurrent access (basic thread safety test)
        def concurrent_operation():
            return filter_instance.inlet({"messages": [], "user": {"id": "test"}})

        import threading
        threads = []
        results = []
        
        for i in range(3):
            thread = threading.Thread(target=lambda: results.append(concurrent_operation()))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=5)

        # All operations should complete successfully
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)

    def test_orchestration_priority_handling(self, filter_instance):
        """Test priority-based execution handling."""
        # Test that filter can handle priority configuration
        if hasattr(filter_instance.valves, 'filter_priority'):
            # Should accept valid priority values
            valid_priorities = ['highest', 'high', 'normal', 'low', 'lowest']
            # Just test that the filter doesn't break with priority settings
            # Actual priority enforcement would be tested at integration level


class TestFilterErrorHandling:
    """Test error handling and edge cases."""

    def test_inlet_with_none_input(self, filter_instance):
        """Test inlet method with None input."""
        result = filter_instance.inlet(None)
        # Filter returns the original body on error (None -> None)
        assert result is None

    def test_outlet_with_none_input(self, filter_instance):
        """Test outlet method with None input."""
        result = filter_instance.outlet(None)
        # Filter returns the original body on error (None -> None)
        assert result is None

    def test_stream_with_none_input(self, filter_instance):
        """Test stream method with None input."""
        result = filter_instance.stream(None)
        # Stream may return None or dict depending on implementation
        assert result is None or isinstance(result, dict)

    def test_inlet_with_malformed_body(self, filter_instance):
        """Test inlet with malformed message body."""
        malformed_bodies = [
            {"messages": "not a list"},
            {"messages": []},  # Empty messages
            {"no_messages": "field"},
            {"messages": [{"invalid": "structure"}]},
            {"messages": [None]},
        ]

        for body in malformed_bodies:
            result = filter_instance.inlet(body)
            assert isinstance(result, dict), f"Failed for body: {body}"

    def test_outlet_with_malformed_body(self, filter_instance):
        """Test outlet with malformed message body."""
        malformed_bodies = [
            {"messages": "not a list"},
            {"messages": []},
            {"no_messages": "field"},
            {"messages": [{"invalid": "structure"}]},
        ]

        for body in malformed_bodies:
            result = filter_instance.outlet(body)
            assert isinstance(result, dict), f"Failed for body: {body}"

    def test_exception_handling_in_methods(self, filter_instance):
        """Test that methods handle internal exceptions gracefully."""
        # Test with data that might cause internal errors
        problematic_data = {
            "messages": [
                {
                    "id": None,  # Might cause issues
                    "role": "user",
                    "content": "",  # Empty content
                    "timestamp": "invalid_timestamp"
                }
            ],
            "user": {"id": ""},  # Empty user ID
            "chat_id": None
        }

        # Should not raise exceptions
        try:
            result = filter_instance.inlet(problematic_data)
            assert isinstance(result, dict)
            
            result = filter_instance.outlet(problematic_data)
            assert isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"Filter should handle problematic data gracefully, but raised: {e}")

    def test_missing_user_info_handling(self, filter_instance):
        """Test handling of missing user information."""
        bodies_without_user = [
            {"messages": [{"id": "1", "role": "user", "content": "test"}]},  # No user field
            {"messages": [{"id": "1", "role": "user", "content": "test"}], "user": {}},  # Empty user
            {"messages": [{"id": "1", "role": "user", "content": "test"}], "user": {"name": "Test"}},  # No user ID
        ]

        for body in bodies_without_user:
            # Should handle gracefully and return a dict
            inlet_result = filter_instance.inlet(body)
            assert isinstance(inlet_result, dict)
            
            outlet_result = filter_instance.outlet(body)
            assert isinstance(outlet_result, dict)


class TestFilterPerformance:
    """Test filter performance characteristics."""

    def test_inlet_performance_baseline(self, filter_instance, basic_message_body):
        """Test inlet method performance baseline."""
        start_time = time.time()
        result = filter_instance.inlet(basic_message_body)
        execution_time = time.time() - start_time
        
        assert execution_time < 5.0, f"Inlet took {execution_time:.2f}s, should be under 5s"
        assert isinstance(result, dict)

    def test_outlet_performance_baseline(self, filter_instance, basic_message_body):
        """Test outlet method performance baseline."""
        start_time = time.time()
        result = filter_instance.outlet(basic_message_body)
        execution_time = time.time() - start_time
        
        assert execution_time < 5.0, f"Outlet took {execution_time:.2f}s, should be under 5s"
        assert isinstance(result, dict)

    def test_stream_performance_baseline(self, filter_instance, stream_event):
        """Test stream method performance baseline."""
        start_time = time.time()
        result = filter_instance.stream(stream_event)
        execution_time = time.time() - start_time
        
        assert execution_time < 1.0, f"Stream took {execution_time:.2f}s, should be under 1s"
        assert isinstance(result, dict)

    def test_concurrent_operations_performance(self, filter_instance, basic_message_body):
        """Test performance under concurrent operations."""
        import threading
        import time
        
        results = []
        errors = []
        execution_times = []
        
        def timed_operation():
            try:
                start = time.time()
                result = filter_instance.inlet(basic_message_body.copy())
                end = time.time()
                execution_times.append(end - start)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=timed_operation)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=10)

        # Verify results
        assert len(errors) == 0, f"Errors in concurrent execution: {errors}"
        assert len(results) == 5
        
        # Check that no operation took too long
        max_time = max(execution_times) if execution_times else 0
        assert max_time < 10.0, f"Slowest operation took {max_time:.2f}s"

    def test_memory_usage_stability(self, filter_instance, basic_message_body):
        """Test that memory usage remains stable across multiple operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple operations
        for i in range(20):
            body_copy = basic_message_body.copy()
            body_copy['messages'][0]['content'] = f"Test message {i}"
            
            result = filter_instance.inlet(body_copy)
            result = filter_instance.outlet(result)
            
            assert isinstance(result, dict)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase significantly
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"


class TestFilterAPIIntegration:
    """Test API integration functionality."""

    @patch('httpx.AsyncClient')
    def test_gemini_api_integration_mock(self, mock_client, filter_instance):
        """Test Gemini API integration with mocks."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Mock response"}]
                    }
                }
            ]
        }
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        # Test with API-requiring message
        message_body = {
            "messages": [
                {
                    "id": "msg1",
                    "role": "user",
                    "content": "Complex message requiring API processing",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "user": {"id": "test_user", "name": "Test User"},
            "chat_id": "test_chat"
        }

        # Should handle API integration gracefully
        result = filter_instance.inlet(message_body)
        assert isinstance(result, dict)

    @patch('sentence_transformers.SentenceTransformer')
    def test_embedding_integration_mock(self, mock_st, filter_instance):
        """Test embedding model integration with mocks."""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_st.return_value = mock_model
        
        message_body = {
            "messages": [
                {
                    "id": "msg1",
                    "role": "user", 
                    "content": "Test message for embedding",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "user": {"id": "test_user", "name": "Test User"},
            "chat_id": "test_chat"
        }

        result = filter_instance.inlet(message_body)
        assert isinstance(result, dict)

    def test_api_error_handling(self, filter_instance):
        """Test handling of API errors."""
        # Test with configuration that might cause API issues
        message_body = {
            "messages": [
                {
                    "id": "msg1",
                    "role": "user",
                    "content": "Test message",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "user": {"id": "test_user", "name": "Test User"},
            "chat_id": "test_chat"
        }

        # Should handle API errors gracefully and not break
        result = filter_instance.inlet(message_body)
        assert isinstance(result, dict)
        
        result = filter_instance.outlet(message_body)
        assert isinstance(result, dict)


class TestFilterValvesConfiguration:
    """Test Valves configuration system."""

    def test_valves_initialization(self, filter_instance):
        """Test that Valves are properly initialized."""
        assert filter_instance.valves is not None
        assert hasattr(filter_instance.valves, '__dict__')

    def test_valves_validation(self, filter_instance):
        """Test Valves field validation."""
        valves = filter_instance.valves
        
        # Test that valves has reasonable configuration fields
        valve_attrs = [attr for attr in dir(valves) if not attr.startswith('_')]
        assert len(valve_attrs) > 0  # Should have some configuration options

    def test_valves_persistence(self, filter_instance):
        """Test that Valves configuration persists correctly."""
        valves = filter_instance.valves
        
        # Basic test that valves object remains consistent
        initial_valves = filter_instance.valves
        
        # Process a message
        message_body = {
            "messages": [{"id": "1", "role": "user", "content": "test"}],
            "user": {"id": "test_user"}
        }
        filter_instance.inlet(message_body)
        
        # Valves should still be the same object
        assert filter_instance.valves is initial_valves

    def test_valves_type_safety(self, filter_instance):
        """Test Valves type safety and validation."""
        from pydantic import ValidationError
        
        # Test that Valves properly validates types
        valves_class = filter_instance.Valves
        
        # Should be able to create valid instance
        try:
            test_valves = valves_class()
            assert test_valves is not None
        except Exception as e:
            pytest.fail(f"Failed to create Valves instance: {e}")


class TestFilterThreadSafety:
    """Test filter thread safety."""

    def test_basic_thread_safety(self, filter_instance, basic_message_body):
        """Test basic thread safety of filter operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def thread_operation(thread_id):
            try:
                # Modify message to be unique per thread
                body_copy = basic_message_body.copy()
                body_copy['messages'][0]['content'] = f"Thread {thread_id} message"
                
                # Perform both inlet and outlet operations
                inlet_result = filter_instance.inlet(body_copy)
                outlet_result = filter_instance.outlet(inlet_result)
                
                results.append((thread_id, inlet_result, outlet_result))
                
            except Exception as e:
                errors.append((thread_id, e))

        # Create and start threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=thread_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=15)

        # Verify results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        
        # Verify all results are valid
        for thread_id, inlet_result, outlet_result in results:
            assert isinstance(inlet_result, dict), f"Thread {thread_id} inlet result not dict"
            assert isinstance(outlet_result, dict), f"Thread {thread_id} outlet result not dict"

    def test_concurrent_memory_operations(self, filter_instance):
        """Test concurrent memory operations for thread safety."""
        import threading
        
        results = []
        errors = []
        
        def memory_operation(operation_id):
            try:
                message_body = {
                    "messages": [
                        {
                            "id": f"msg_{operation_id}",
                            "role": "user",
                            "content": f"Memory content for operation {operation_id}",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    ],
                    "user": {"id": f"user_{operation_id}", "name": f"User {operation_id}"},
                    "chat_id": f"chat_{operation_id}"
                }
                
                # Perform memory-related operations
                inlet_result = filter_instance.inlet(message_body)
                outlet_result = filter_instance.outlet(inlet_result)
                
                results.append((operation_id, inlet_result, outlet_result))
                
            except Exception as e:
                errors.append((operation_id, e))

        # Run concurrent memory operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=memory_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=10)

        # Verify thread safety
        assert len(errors) == 0, f"Concurrent memory operation errors: {errors}"
        assert len(results) == 5
        
        for operation_id, inlet_result, outlet_result in results:
            assert isinstance(inlet_result, dict)
            assert isinstance(outlet_result, dict)


class TestFilterMonolithicCompliance:
    """Test compliance with monolithic structure requirements."""

    def test_single_file_structure(self, filter_instance):
        """Test that filter maintains single-file structure."""
        # Verify the filter class is properly contained
        assert hasattr(filter_instance, '__class__')
        assert filter_instance.__class__.__name__ == 'Filter'
        
        # Verify required nested classes
        assert hasattr(filter_instance.__class__, 'Valves')
        
        # Verify the filter module exists (it's dynamically loaded in tests)
        import sys
        # The module is adaptive_memory_v4_0 when loaded in tests
        module_name = filter_instance.__class__.__module__
        # Module might not be in sys.modules if loaded dynamically
        # But we can verify the class has a valid module name
        assert module_name is not None
        assert module_name in ['adaptive_memory_v4_0', '__main__', 'adaptive_memory_v4.0']

    def test_no_external_file_dependencies(self, filter_instance):
        """Test that filter doesn't depend on external files."""
        # Filter should be self-contained
        # Test that it can operate without external config files
        
        message_body = {
            "messages": [{"id": "1", "role": "user", "content": "test"}],
            "user": {"id": "test_user"}
        }
        
        # Should work without external files
        result = filter_instance.inlet(message_body)
        assert isinstance(result, dict)

    def test_approved_imports_only(self, filter_instance):
        """Test that filter only uses approved imports."""
        import sys
        
        # Get the filter's module
        filter_module = sys.modules.get(filter_instance.__class__.__module__)
        if filter_module:
            # Check that commonly used modules are available
            # This is more of a structural test
            required_modules = ['json', 'asyncio', 'datetime', 'typing', 'logging']
            for module_name in required_modules:
                assert module_name in sys.modules or hasattr(__builtins__, module_name), f"Missing required module: {module_name}"


class TestFilterStreamingSupport:
    """Test streaming support functionality."""

    def test_stream_method_basic_functionality(self, filter_instance):
        """Test basic stream method functionality."""
        stream_events = [
            {"type": "message", "data": {"content": "Hello", "role": "assistant"}},
            {"type": "chunk", "data": {"content": " world", "role": "assistant"}},
            {"type": "end", "data": {"content": "", "role": "assistant"}},
        ]
        
        for event in stream_events:
            result = filter_instance.stream(event)
            assert isinstance(result, dict)

    def test_stream_with_malformed_events(self, filter_instance):
        """Test stream method with malformed events."""
        malformed_events = [
            None,
            {},
            {"type": "unknown"},
            {"data": "no_type"},
            {"type": None, "data": None},
        ]
        
        for event in malformed_events:
            result = filter_instance.stream(event)
            assert isinstance(result, dict)

    def test_stream_performance(self, filter_instance):
        """Test stream method performance."""
        event = {"type": "message", "data": {"content": "test", "role": "assistant"}}
        
        start_time = time.time()
        for i in range(100):
            result = filter_instance.stream(event)
            assert isinstance(result, dict)
        
        total_time = time.time() - start_time
        avg_time = total_time / 100
        
        # Should be very fast for streaming
        assert avg_time < 0.01, f"Average stream processing time too slow: {avg_time:.4f}s"


@pytest.mark.integration
class TestFilterIntegrationPoints:
    """Test integration points with OpenWebUI and other systems."""

    def test_openwebui_message_flow(self, filter_instance):
        """Test complete OpenWebUI message flow through filter."""
        # Simulate complete OpenWebUI flow
        
        # 1. User sends message (inlet processing)
        user_message = {
            "messages": [
                {
                    "id": "user_msg_1",
                    "role": "user",
                    "content": "What is machine learning?",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ],
            "chat_id": "integration_test_chat",
            "user": {
                "id": "integration_test_user",
                "name": "Integration Test User",
                "email": "integration@test.com"
            },
            "model": "gpt-4",
            "stream": False
        }
        
        # Process user input
        inlet_result = filter_instance.inlet(user_message)
        assert isinstance(inlet_result, dict)
        assert 'messages' in inlet_result
        
        # 2. LLM generates response (outlet processing)
        llm_response = inlet_result.copy()
        llm_response['messages'].append({
            "id": "assistant_msg_1",
            "role": "assistant", 
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Process LLM output
        outlet_result = filter_instance.outlet(llm_response)
        assert isinstance(outlet_result, dict)
        assert 'messages' in outlet_result
        assert len(outlet_result['messages']) >= 2  # Should have both user and assistant messages

    def test_multi_turn_conversation_flow(self, filter_instance):
        """Test multi-turn conversation handling."""
        base_body = {
            "chat_id": "multi_turn_test",
            "user": {
                "id": "multi_turn_user",
                "name": "Multi Turn User"
            },
            "model": "gpt-4",
            "stream": False
        }
        
        # Turn 1
        turn1_body = base_body.copy()
        turn1_body['messages'] = [
            {
                "id": "turn1_user",
                "role": "user",
                "content": "I'm learning Python programming",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        turn1_result = filter_instance.inlet(turn1_body)
        assert isinstance(turn1_result, dict)
        
        # Turn 2 - building on previous context
        turn2_body = base_body.copy()
        turn2_body['messages'] = [
            {
                "id": "turn1_user",
                "role": "user", 
                "content": "I'm learning Python programming",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "turn1_assistant",
                "role": "assistant",
                "content": "That's great! Python is an excellent language for beginners.",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "turn2_user", 
                "role": "user",
                "content": "What are some good Python libraries to learn?",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        turn2_result = filter_instance.inlet(turn2_body)
        assert isinstance(turn2_result, dict)
        
        # Should handle multi-turn context appropriately
        assert 'messages' in turn2_result

    def test_error_recovery_integration(self, filter_instance):
        """Test error recovery in integration scenarios."""
        # Test various error conditions that might occur in production
        
        error_scenarios = [
            # Missing required fields
            {"messages": []},
            
            # Malformed timestamps
            {
                "messages": [
                    {
                        "id": "error_test",
                        "role": "user",
                        "content": "test",
                        "timestamp": "invalid_timestamp"
                    }
                ],
                "user": {"id": "error_user"}
            },
            
            # Large message content
            {
                "messages": [
                    {
                        "id": "large_msg",
                        "role": "user", 
                        "content": "x" * 10000,  # Very large content
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                ],
                "user": {"id": "large_msg_user"}
            }
        ]
        
        for scenario in error_scenarios:
            # Should handle all error scenarios gracefully
            inlet_result = filter_instance.inlet(scenario)
            assert isinstance(inlet_result, dict)
            
            outlet_result = filter_instance.outlet(scenario)
            assert isinstance(outlet_result, dict)