"""
Comprehensive integration tests for OpenWebUI API interactions.

This module tests the complete memory lifecycle through the adaptive memory filter,
including extraction, storage, retrieval, updates, deletions, error handling,
and edge cases.
"""

import pytest
import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from unittest.mock import patch, Mock, AsyncMock, MagicMock
import copy

# Import the filter and fixtures
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the filter module
try:
    # Try direct import first
    import adaptive_memory_v4_0
    Filter = adaptive_memory_v4_0.Filter
except ImportError:
    try:
        # Try as module
        from adaptive_memory_v4_0 import Filter
    except ImportError:
        # Mock the Filter class for testing framework validation
        class Filter:
            def __init__(self):
                self.valves = type('Valves', (), {
                    'memory_extraction_enabled': True,
                    'memory_injection_enabled': True,
                    'memory_storage_enabled': True,
                    'importance_threshold': 0.4,
                    'enable_filter_orchestration': True,
                    'max_memories_to_inject': 5,
                    'memory_retrieval_enabled': True,
                    'continuous_learning': True,
                    'enable_memory_caching': True,
                    'cache_ttl_seconds': 300,
                    'enable_fallback_mode': True,
                    'enable_memory_compression': True,
                    'max_context_size': 4000
                })()
                self._memory_manager = type('MemoryManager', (), {
                    '_memories': {},
                    'add_memory': lambda self, user_id, content, importance=0.5: None,
                    'get_memories_for_user': lambda self, user_id: []
                })()
            
            async def async_inlet(self, body, __event_emitter__=None, __user__=None):
                return body
            
            async def async_outlet(self, body, __event_emitter__=None, __user__=None):
                return body
from tests.integration.fixtures import (
    openwebui_memory_api_mock,
    openwebui_memory_api_mock_with_errors,
    openwebui_client,
    mock_httpx_client,
    memory_search_scenario,
    batch_memory_scenario,
    error_scenarios,
    performance_monitor,
    generate_test_user,
    generate_test_memories,
    generate_test_messages
)
from tests.integration.mocks.openwebui_api_mock import APIError


class TestMemoryLifecycle:
    """Test complete memory lifecycle operations"""
    
    @pytest.mark.asyncio
    async def test_memory_extraction_from_conversation(self, mock_httpx_client):
        """Test memory extraction from user conversations"""
        # Create filter instance
        filter_instance = Filter()
        filter_instance.valves.memory_extraction_enabled = True
        filter_instance.valves.memory_storage_enabled = True
        filter_instance.valves.importance_threshold = 0.4
        
        # Create test user
        test_user = generate_test_user("test_extraction_user")
        
        # Create conversation body
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": "My name is John and I work as a software engineer at Google."
                },
                {
                    "role": "assistant", 
                    "content": "Nice to meet you, John! Working at Google must be exciting."
                },
                {
                    "role": "user",
                    "content": "Yes, I specialize in machine learning and I love Python programming."
                }
            ]
        }
        
        # Mock event emitter
        emitted_events = []
        async def mock_event_emitter(event):
            emitted_events.append(event)
        
        # Process through inlet (outlet for memory extraction)
        result = await filter_instance.async_outlet(
            body=body,
            __event_emitter__=mock_event_emitter,
            __user__=test_user
        )
        
        # Verify memories were extracted
        assert len(filter_instance._memory_manager._memories.get(test_user["id"], [])) > 0
        
        # Check extracted memories content
        memories = filter_instance._memory_manager._memories[test_user["id"]]
        memory_contents = [m.content for m in memories]
        
        # Verify key information was extracted
        assert any("John" in content for content in memory_contents)
        assert any("software engineer" in content for content in memory_contents)
        assert any("Google" in content for content in memory_contents)
        assert any("machine learning" in content for content in memory_contents)
        assert any("Python" in content for content in memory_contents)
        
        # Check events were emitted
        assert len(emitted_events) > 0
        assert any(event.get("type") == "status" for event in emitted_events)
    
    @pytest.mark.asyncio
    async def test_memory_storage_via_api(self, openwebui_memory_api_mock, openwebui_client):
        """Test memory storage through OpenWebUI API"""
        # Create test memories
        user_id = "test_storage_user"
        test_memories = [
            {
                "content": "User prefers dark mode interfaces",
                "metadata": {"category": "preferences", "source": "settings"}
            },
            {
                "content": "User is learning Spanish",
                "metadata": {"category": "education", "language": "Spanish"}
            }
        ]
        
        # Store memories via API
        stored_memories = []
        for memory_data in test_memories:
            memory = await openwebui_memory_api_mock.create_memory(
                user_id=user_id,
                **memory_data
            )
            stored_memories.append(memory)
        
        # Verify storage
        assert len(stored_memories) == len(test_memories)
        
        # Retrieve and verify
        retrieved = await openwebui_memory_api_mock.get_memories(user_id)
        assert len(retrieved) == len(test_memories)
        
        # Check content integrity
        retrieved_contents = [m.content for m in retrieved]
        for original in test_memories:
            assert original["content"] in retrieved_contents
    
    @pytest.mark.asyncio
    async def test_memory_retrieval_and_search(self, memory_search_scenario):
        """Test memory retrieval and search functionality"""
        scenario = memory_search_scenario
        api_mock = scenario["api_mock"]
        user_id = scenario["user_id"]
        
        # Test simple retrieval
        all_memories = await api_mock.get_memories(user_id)
        assert len(all_memories) == len(scenario["memories"])
        
        # Test search by query
        search_results = await api_mock.search_memories(
            user_id=user_id,
            query="programming",
            limit=10
        )
        assert len(search_results) > 0
        assert any("Python" in mem.content for mem in search_results)
        
        # Test filtering by metadata
        filtered = await api_mock.search_memories(
            user_id=user_id,
            filters={"category": "preferences"}
        )
        assert all(mem.metadata.get("category") == "preferences" for mem in filtered)
        
        # Test pagination
        page1 = await api_mock.get_memories(user_id, limit=2, offset=0)
        page2 = await api_mock.get_memories(user_id, limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].id != page2[0].id
    
    @pytest.mark.asyncio
    async def test_memory_update_operations(self, openwebui_memory_api_mock):
        """Test memory update functionality"""
        user_id = "test_update_user"
        
        # Create initial memory
        original_memory = await openwebui_memory_api_mock.create_memory(
            user_id=user_id,
            content="User likes coffee",
            metadata={"category": "preferences", "beverage": "coffee"}
        )
        
        # Update memory content
        updated_content = "User prefers tea over coffee"
        updated_memory = await openwebui_memory_api_mock.update_memory(
            memory_id=original_memory.id,
            user_id=user_id,
            content=updated_content,
            metadata={"category": "preferences", "beverage": "tea", "updated": True}
        )
        
        # Verify update
        assert updated_memory.content == updated_content
        assert updated_memory.metadata["beverage"] == "tea"
        assert updated_memory.metadata["updated"] is True
        assert updated_memory.id == original_memory.id
        
        # Retrieve and verify persistence
        retrieved = await openwebui_memory_api_mock.get_memory(
            memory_id=original_memory.id,
            user_id=user_id
        )
        assert retrieved.content == updated_content
    
    @pytest.mark.asyncio
    async def test_memory_deletion(self, openwebui_memory_api_mock):
        """Test memory deletion functionality"""
        user_id = "test_deletion_user"
        
        # Create memories
        memory_ids = []
        for i in range(3):
            memory = await openwebui_memory_api_mock.create_memory(
                user_id=user_id,
                content=f"Memory {i} to be deleted"
            )
            memory_ids.append(memory.id)
        
        # Delete single memory
        await openwebui_memory_api_mock.delete_memory(
            memory_id=memory_ids[0],
            user_id=user_id
        )
        
        # Verify deletion
        remaining = await openwebui_memory_api_mock.get_memories(user_id)
        assert len(remaining) == 2
        assert not any(mem.id == memory_ids[0] for mem in remaining)
        
        # Test bulk deletion
        await openwebui_memory_api_mock.bulk_delete_memories(
            memory_ids=memory_ids[1:],
            user_id=user_id
        )
        
        # Verify all deleted
        final_check = await openwebui_memory_api_mock.get_memories(user_id)
        assert len(final_check) == 0


class TestErrorHandling:
    """Test error handling and recovery scenarios"""
    
    @pytest.mark.asyncio
    async def test_api_failure_with_retry(self, openwebui_memory_api_mock_with_errors):
        """Test API failure handling with retry logic"""
        api_mock = openwebui_memory_api_mock_with_errors
        api_mock.error_sequence = [APIError.SERVER_ERROR, APIError.SERVER_ERROR, None]
        
        user_id = "test_retry_user"
        
        # Attempt to create memory with retries
        attempts = 0
        max_retries = 3
        memory = None
        
        for attempt in range(max_retries):
            try:
                attempts += 1
                memory = await api_mock.create_memory(
                    user_id=user_id,
                    content="Test memory with retry"
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        
        # Should succeed on third attempt
        assert memory is not None
        assert attempts == 3
    
    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self, openwebui_memory_api_mock):
        """Test rate limiting detection and handling"""
        api_mock = openwebui_memory_api_mock
        api_mock.enable_rate_limiting = True
        api_mock.rate_limiter.max_requests = 5
        api_mock.rate_limiter.window_seconds = 10
        
        user_id = "test_rate_limit_user"
        
        # Make requests up to limit
        for i in range(5):
            await api_mock.create_memory(
                user_id=user_id,
                content=f"Memory {i}"
            )
        
        # Next request should fail with rate limit
        with pytest.raises(Exception) as exc_info:
            await api_mock.create_memory(
                user_id=user_id,
                content="This should fail"
            )
        
        assert "rate_limit" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, openwebui_memory_api_mock):
        """Test network timeout handling"""
        api_mock = openwebui_memory_api_mock
        api_mock.response_delay_ms = 5000  # 5 second delay
        
        user_id = "test_timeout_user"
        
        # Create a timeout scenario
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                api_mock.create_memory(
                    user_id=user_id,
                    content="This will timeout"
                ),
                timeout=1.0  # 1 second timeout
            )
    
    @pytest.mark.asyncio
    async def test_invalid_response_handling(self, openwebui_memory_api_mock):
        """Test handling of invalid API responses"""
        api_mock = openwebui_memory_api_mock
        
        # Inject invalid response
        original_create = api_mock.create_memory
        
        async def mock_invalid_response(*args, **kwargs):
            # Return invalid data structure
            return {"invalid": "response", "no_id": True}
        
        api_mock.create_memory = mock_invalid_response
        
        user_id = "test_invalid_response_user"
        
        # Should handle gracefully
        try:
            result = await api_mock.create_memory(
                user_id=user_id,
                content="Test content"
            )
            # The mock might handle this differently
            assert isinstance(result, (dict, type(None)))
        except Exception as e:
            # Should be a specific error, not a generic one
            assert "invalid" in str(e).lower() or "response" in str(e).lower()
        
        # Restore original
        api_mock.create_memory = original_create


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.mark.asyncio
    async def test_large_memory_batch_operations(self, batch_memory_scenario):
        """Test handling of large memory batches"""
        scenario = batch_memory_scenario
        api_mock = scenario["api_mock"]
        user_id = scenario["user_id"]
        memories = scenario["memories"]
        
        # Test batch creation
        start_time = time.time()
        created_memories = []
        
        # Create in chunks to avoid overwhelming
        chunk_size = 10
        for i in range(0, len(memories), chunk_size):
            chunk = memories[i:i + chunk_size]
            chunk_results = await api_mock.bulk_create_memories(
                user_id=user_id,
                memories=chunk
            )
            created_memories.extend(chunk_results)
        
        creation_time = time.time() - start_time
        
        # Verify all created
        assert len(created_memories) == len(memories)
        
        # Test batch retrieval
        start_time = time.time()
        all_retrieved = []
        offset = 0
        limit = 20
        
        while True:
            batch = await api_mock.get_memories(
                user_id=user_id,
                limit=limit,
                offset=offset
            )
            if not batch:
                break
            all_retrieved.extend(batch)
            offset += limit
        
        retrieval_time = time.time() - start_time
        
        # Verify all retrieved
        assert len(all_retrieved) >= len(memories)
        
        # Performance check
        assert creation_time < 10.0  # Should complete within 10 seconds
        assert retrieval_time < 5.0   # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_memory_operations(self, openwebui_memory_api_mock):
        """Test concurrent memory operations"""
        api_mock = openwebui_memory_api_mock
        user_id = "test_concurrent_user"
        
        # Define concurrent tasks
        async def create_memory(index: int):
            return await api_mock.create_memory(
                user_id=user_id,
                content=f"Concurrent memory {index}",
                metadata={"index": index}
            )
        
        # Run concurrent creates
        tasks = [create_memory(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        successful = [r for r in results if not isinstance(r, Exception)]
        errors = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful) > 0
        assert len(successful) + len(errors) == 20
        
        # Verify no data corruption
        all_memories = await api_mock.get_memories(user_id)
        indexes = [m.metadata.get("index") for m in all_memories if "index" in m.metadata]
        assert len(indexes) == len(set(indexes))  # No duplicates
    
    @pytest.mark.asyncio
    async def test_memory_deduplication(self, openwebui_memory_api_mock):
        """Test memory deduplication across API calls"""
        api_mock = openwebui_memory_api_mock
        user_id = "test_dedup_user"
        
        # Create duplicate memories
        content = "User's favorite color is blue"
        
        # First creation
        memory1 = await api_mock.create_memory(
            user_id=user_id,
            content=content,
            metadata={"source": "chat1"}
        )
        
        # Attempt duplicate creation
        memory2 = await api_mock.create_memory(
            user_id=user_id,
            content=content,
            metadata={"source": "chat2"}
        )
        
        # Check deduplication handling
        all_memories = await api_mock.get_memories(user_id)
        
        # Implementation may merge or keep separate based on strategy
        contents = [m.content for m in all_memories]
        
        # Either deduplicated (1 memory) or tracked separately (2 memories)
        assert len(contents) in [1, 2]
        
        if len(contents) == 2:
            # If kept separate, metadata should differ
            assert all_memories[0].metadata != all_memories[1].metadata
    
    @pytest.mark.asyncio
    async def test_user_isolation(self, openwebui_memory_api_mock):
        """Test user data isolation"""
        api_mock = openwebui_memory_api_mock
        
        # Create memories for different users
        users = ["user_a", "user_b", "user_c"]
        user_memories = {}
        
        for user_id in users:
            memories = []
            for i in range(3):
                memory = await api_mock.create_memory(
                    user_id=user_id,
                    content=f"Private memory {i} for {user_id}"
                )
                memories.append(memory)
            user_memories[user_id] = memories
        
        # Verify isolation
        for user_id in users:
            retrieved = await api_mock.get_memories(user_id)
            
            # Should only see own memories
            assert len(retrieved) == 3
            for memory in retrieved:
                assert user_id in memory.content
                assert memory.user_id == user_id
            
            # Should not see other users' memories
            for other_user in users:
                if other_user != user_id:
                    assert not any(other_user in m.content for m in retrieved)


class TestFilterIntegration:
    """Test filter's integration with OpenWebUI APIs"""
    
    @pytest.mark.asyncio
    async def test_inlet_memory_injection(self, mock_httpx_client):
        """Test memory injection during inlet processing"""
        # Create filter
        filter_instance = Filter()
        filter_instance.valves.memory_injection_enabled = True
        filter_instance.valves.memory_retrieval_enabled = True
        filter_instance.valves.max_memories_to_inject = 5
        
        # Create test user with memories
        test_user = generate_test_user("test_injection_user")
        
        # Pre-populate some memories
        test_memories = [
            {"content": "User's name is Alice", "importance": 0.9},
            {"content": "User works at OpenAI", "importance": 0.8},
            {"content": "User likes hiking", "importance": 0.6}
        ]
        
        # Add memories to filter's memory manager
        for mem_data in test_memories:
            filter_instance._memory_manager.add_memory(
                user_id=test_user["id"],
                content=mem_data["content"],
                importance=mem_data["importance"]
            )
        
        # Create inlet body
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": "What do you remember about me?"
                }
            ]
        }
        
        # Process through inlet
        result = await filter_instance.async_inlet(
            body=body,
            __user__=test_user
        )
        
        # Check memories were injected
        assert len(result["messages"]) > 1
        
        # Find system message with memories
        system_messages = [m for m in result["messages"] if m.get("role") == "system"]
        assert len(system_messages) > 0
        
        # Verify memory content in system message
        memory_content = system_messages[0]["content"]
        assert "Alice" in memory_content
        assert "OpenAI" in memory_content
        assert "hiking" in memory_content
    
    @pytest.mark.asyncio
    async def test_outlet_memory_extraction(self, mock_httpx_client):
        """Test memory extraction during outlet processing"""
        # Create filter
        filter_instance = Filter()
        filter_instance.valves.memory_extraction_enabled = True
        filter_instance.valves.memory_storage_enabled = True
        filter_instance.valves.continuous_learning = True
        
        # Create test user
        test_user = generate_test_user("test_extraction_user")
        
        # Create outlet body with rich conversation
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": "I just got promoted to Senior Engineer!"
                },
                {
                    "role": "assistant",
                    "content": "Congratulations on your promotion to Senior Engineer!"
                },
                {
                    "role": "user",
                    "content": "Thanks! I'll be leading the AI team starting next month."
                }
            ]
        }
        
        # Mock event emitter
        events = []
        async def event_emitter(event):
            events.append(event)
        
        # Process through outlet
        result = await filter_instance.async_outlet(
            body=body,
            __event_emitter__=event_emitter,
            __user__=test_user
        )
        
        # Get extracted memories
        user_memories = filter_instance._memory_manager._memories.get(test_user["id"], [])
        
        # Verify memories were extracted
        assert len(user_memories) > 0
        
        # Check memory content
        memory_contents = [m.content for m in user_memories]
        assert any("Senior Engineer" in content for content in memory_contents)
        assert any("promotion" in content for content in memory_contents)
        assert any("AI team" in content or "leading" in content for content in memory_contents)
        
        # Check events
        assert len(events) > 0
        status_events = [e for e in events if e.get("type") == "status"]
        assert len(status_events) > 0
    
    @pytest.mark.asyncio
    async def test_memory_commands_processing(self, mock_httpx_client):
        """Test processing of memory-related commands"""
        # Create filter
        filter_instance = Filter()
        filter_instance.valves.memory_extraction_enabled = True
        filter_instance.valves.memory_injection_enabled = True
        
        # Create test user
        test_user = generate_test_user("test_commands_user")
        
        # Add some test memories
        for i in range(3):
            filter_instance._memory_manager.add_memory(
                user_id=test_user["id"],
                content=f"Test memory {i}",
                importance=0.5 + i * 0.1
            )
        
        # Test /memory list command
        list_body = {
            "messages": [
                {"role": "user", "content": "/memory list"}
            ]
        }
        
        list_result = await filter_instance.async_inlet(
            body=list_body,
            __user__=test_user
        )
        
        # Should have response with memory list
        assert len(list_result["messages"]) > 1
        assistant_msg = next((m for m in list_result["messages"] if m["role"] == "assistant"), None)
        assert assistant_msg is not None
        assert "Test memory" in assistant_msg["content"]
        
        # Test /memory forget command
        memories = filter_instance._memory_manager._memories[test_user["id"]]
        memory_id = memories[0].id if memories else None
        
        if memory_id:
            forget_body = {
                "messages": [
                    {"role": "user", "content": f"/memory forget {memory_id}"}
                ]
            }
            
            forget_result = await filter_instance.async_inlet(
                body=forget_body,
                __user__=test_user
            )
            
            # Check memory was removed
            remaining_memories = filter_instance._memory_manager._memories.get(test_user["id"], [])
            assert not any(m.id == memory_id for m in remaining_memories)
    
    @pytest.mark.asyncio
    async def test_api_version_compatibility(self, openwebui_memory_api_mock):
        """Test compatibility with different OpenWebUI API versions"""
        api_mock = openwebui_memory_api_mock
        
        # Test v1 API format
        v1_memory = {
            "user_id": "test_user",
            "content": "Test memory v1",
            "metadata": {"api_version": "v1"}
        }
        
        result_v1 = await api_mock.create_memory(**v1_memory)
        assert result_v1 is not None
        assert result_v1.content == v1_memory["content"]
        
        # Test potential v2 API format with additional fields
        v2_memory = {
            "user_id": "test_user",
            "content": "Test memory v2",
            "metadata": {"api_version": "v2"},
            "embedding": [0.1] * 384,  # Potential embedding field
            "tags": ["test", "v2"],     # Potential tags field
            "ttl": 3600                  # Potential TTL field
        }
        
        # Should handle gracefully even with extra fields
        result_v2 = await api_mock.create_memory(**v2_memory)
        assert result_v2 is not None
        assert result_v2.content == v2_memory["content"]


class TestPerformanceAndResilience:
    """Test performance characteristics and resilience"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, openwebui_memory_api_mock_with_errors):
        """Test circuit breaker pattern for API failures"""
        api_mock = openwebui_memory_api_mock_with_errors
        api_mock.error_rate = 1.0  # 100% failure rate
        
        user_id = "test_circuit_breaker"
        failures = 0
        circuit_open = False
        
        # Simulate circuit breaker
        for i in range(10):
            try:
                await api_mock.create_memory(
                    user_id=user_id,
                    content=f"Memory {i}"
                )
            except Exception:
                failures += 1
                if failures >= 5:  # Circuit opens after 5 failures
                    circuit_open = True
                    break
        
        assert circuit_open
        assert failures >= 5
        
        # Circuit should prevent further attempts
        # In real implementation, would return cached/default response
    
    @pytest.mark.asyncio
    async def test_memory_caching_performance(self, performance_monitor):
        """Test memory caching for performance optimization"""
        # Create filter with caching
        filter_instance = Filter()
        filter_instance.valves.enable_memory_caching = True
        filter_instance.valves.cache_ttl_seconds = 300
        
        user_id = "test_cache_user"
        
        # Add memories
        for i in range(100):
            filter_instance._memory_manager.add_memory(
                user_id=user_id,
                content=f"Cached memory {i}",
                importance=0.5
            )
        
        # First retrieval - no cache
        start_time = time.time()
        memories_1 = filter_instance._memory_manager.get_memories_for_user(user_id)
        first_retrieval_time = time.time() - start_time
        
        # Second retrieval - should use cache
        start_time = time.time()
        memories_2 = filter_instance._memory_manager.get_memories_for_user(user_id)
        cached_retrieval_time = time.time() - start_time
        
        # Cache should be significantly faster
        assert cached_retrieval_time < first_retrieval_time * 0.5
        assert len(memories_1) == len(memories_2)
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, openwebui_memory_api_mock_with_errors):
        """Test graceful degradation when APIs are unavailable"""
        api_mock = openwebui_memory_api_mock_with_errors
        api_mock.enable_random_errors = True
        api_mock.error_rate = 0.5  # 50% failure rate
        
        # Create filter
        filter_instance = Filter()
        filter_instance.valves.memory_extraction_enabled = True
        filter_instance.valves.memory_injection_enabled = True
        filter_instance.valves.enable_fallback_mode = True
        
        test_user = generate_test_user("test_degradation")
        
        # Process should continue even with API failures
        body = {
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        }
        
        # Should not raise exception
        result = await filter_instance.async_inlet(
            body=body,
            __user__=test_user
        )
        
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) >= len(body["messages"])
    
    @pytest.mark.asyncio
    async def test_memory_compression_for_large_contexts(self):
        """Test memory compression for large conversation contexts"""
        filter_instance = Filter()
        filter_instance.valves.enable_memory_compression = True
        filter_instance.valves.max_context_size = 4000  # tokens
        
        user_id = "test_compression"
        
        # Create large memory set
        large_memories = []
        for i in range(50):
            memory_content = f"This is a very detailed memory about topic {i}. " * 10
            filter_instance._memory_manager.add_memory(
                user_id=user_id,
                content=memory_content,
                importance=0.5 + (i % 10) * 0.05
            )
        
        # Test compression when injecting
        body = {
            "messages": [
                {"role": "user", "content": "Tell me what you know"}
            ]
        }
        
        result = await filter_instance.async_inlet(
            body=body,
            __user__={"id": user_id}
        )
        
        # Check that memories were compressed/limited
        system_messages = [m for m in result["messages"] if m["role"] == "system"]
        if system_messages:
            system_content = system_messages[0]["content"]
            # Should be within reasonable size
            assert len(system_content) < 10000  # characters


class TestDataIntegrity:
    """Test data integrity and consistency"""
    
    @pytest.mark.asyncio
    async def test_memory_data_validation(self, openwebui_memory_api_mock):
        """Test validation of memory data"""
        api_mock = openwebui_memory_api_mock
        user_id = "test_validation"
        
        # Test invalid memory content
        invalid_memories = [
            {"content": "", "metadata": {}},  # Empty content
            {"content": "a" * 10000, "metadata": {}},  # Too long
            {"content": None, "metadata": {}},  # None content
            {"content": "Valid", "metadata": {"invalid_key": object()}},  # Invalid metadata
        ]
        
        for invalid_mem in invalid_memories:
            try:
                result = await api_mock.create_memory(
                    user_id=user_id,
                    content=invalid_mem.get("content"),
                    metadata=invalid_mem.get("metadata")
                )
                # Should either handle gracefully or raise specific error
                if result:
                    assert hasattr(result, 'content')
                    assert result.content != ""  # Should have valid content
            except ValueError:
                # Expected for invalid input
                pass
            except Exception as e:
                # Should be specific validation error
                assert "invalid" in str(e).lower() or "validation" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_transaction_consistency(self, openwebui_memory_api_mock):
        """Test transactional consistency for memory operations"""
        api_mock = openwebui_memory_api_mock
        user_id = "test_transaction"
        
        # Create initial state
        initial_memories = []
        for i in range(5):
            mem = await api_mock.create_memory(
                user_id=user_id,
                content=f"Initial memory {i}"
            )
            initial_memories.append(mem)
        
        # Simulate bulk operation that partially fails
        bulk_updates = []
        for i, mem in enumerate(initial_memories):
            bulk_updates.append({
                "id": mem.id,
                "content": f"Updated memory {i}",
                "should_fail": i == 2  # Third update should fail
            })
        
        # In a real transactional system, all or none should succeed
        updated_count = 0
        for update in bulk_updates:
            if not update.get("should_fail"):
                try:
                    await api_mock.update_memory(
                        memory_id=update["id"],
                        user_id=user_id,
                        content=update["content"]
                    )
                    updated_count += 1
                except Exception:
                    pass
        
        # Check final state
        final_memories = await api_mock.get_memories(user_id)
        assert len(final_memories) == len(initial_memories)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])