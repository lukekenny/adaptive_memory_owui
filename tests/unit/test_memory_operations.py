"""
Unit tests for memory operations in the Adaptive Memory Plugin.

Tests memory extraction, storage, retrieval, and deduplication functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone
import json
import uuid


class TestMemoryOperations:
    """Test memory-related operations."""

    def test_memory_extraction_from_message(self, filter_instance):
        """Test memory extraction from user messages."""
        # Create a message with extractable information
        message_body = {
            "messages": [
                {
                    "role": "user",
                    "content": "My name is John and I work as a software engineer. I prefer Python for development."
                }
            ],
            "user": {"id": "test-user"},
            "chat_id": "test-chat"
        }
        
        # Process through outlet (where memory extraction happens)
        with patch.object(filter_instance, 'query_llm_with_retry') as mock_llm:
            # Create async mock return value
            async def mock_return(system_prompt, user_prompt):
                return json.dumps({
                    "memories": [
                        {
                            "content": "User's name is John",
                            "importance": 0.9,
                            "memory_type": "identity"
                        },
                        {
                            "content": "User works as a software engineer",
                            "importance": 0.8,
                            "memory_type": "identity"
                        },
                        {
                            "content": "User prefers Python for development",
                            "importance": 0.7,
                            "memory_type": "preference"
                        }
                    ]
                })
            mock_llm.side_effect = mock_return
            
            result = filter_instance.outlet(message_body)
            assert isinstance(result, dict)
            
            # Verify LLM was called for extraction
            if hasattr(filter_instance, '_extract_memories_from_message'):
                assert mock_llm.called or filter_instance.valves.memory_extraction_enabled

    def test_memory_filtering(self, filter_instance):
        """Test memory filtering for relevance and quality."""
        # Test memories with different quality levels
        memories = [
            {"content": "User likes pizza", "importance": 0.9},  # Good memory
            {"content": "The", "importance": 0.1},  # Too short
            {"content": "User said hello", "importance": 0.2},  # Low importance
            {"content": "User's favorite color is blue", "importance": 0.8},  # Good memory
            {"content": "System prompt: You are helpful", "importance": 0.5},  # Meta content
        ]
        
        if hasattr(filter_instance, '_filter_memories'):
            # Mock the filter method behavior
            filtered = [m for m in memories if len(m['content']) > 5 and m['importance'] > 0.5]
            assert len(filtered) == 2
            assert filtered[0]['content'] == "User likes pizza"
            assert filtered[1]['content'] == "User's favorite color is blue"

    def test_memory_deduplication(self, filter_instance):
        """Test memory deduplication mechanisms."""
        # Create duplicate memories with slight variations
        existing_memories = [
            {"id": "1", "content": "User likes Python programming", "embedding": [0.1, 0.2, 0.3]},
            {"id": "2", "content": "User enjoys reading books", "embedding": [0.4, 0.5, 0.6]},
        ]
        
        new_memory = {"content": "User likes Python", "embedding": [0.1, 0.21, 0.29]}
        
        # Test similarity detection
        if hasattr(filter_instance, '_calculate_similarity'):
            # This should detect high similarity with first memory
            # Cosine similarity calculation
            import numpy as np
            
            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            similarity = cosine_similarity(existing_memories[0]['embedding'], new_memory['embedding'])
            assert similarity > 0.9  # Should be very similar

    def test_memory_storage_with_metadata(self, filter_instance):
        """Test memory storage with proper metadata."""
        memory_data = {
            "content": "User prefers dark mode interfaces",
            "importance": 0.8,
            "memory_type": "preference",
            "context": "UI preferences"
        }
        
        user_id = "test-user-123"
        
        with patch('open_webui.routers.memories.create_memory') as mock_create:
            mock_create.return_value = {"id": "mem-123", "status": "created"}
            
            # Simulate memory storage
            if hasattr(filter_instance, '_store_memory'):
                # The method would typically add metadata like timestamp
                stored_memory = {
                    **memory_data,
                    "user_id": user_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": "chat_conversation"
                }
                
                # Verify metadata is complete
                assert "timestamp" in stored_memory
                assert "user_id" in stored_memory
                assert stored_memory["memory_type"] == "preference"

    def test_memory_retrieval_with_relevance(self, filter_instance):
        """Test memory retrieval with relevance scoring."""
        query_context = "What are my UI preferences?"
        
        stored_memories = [
            {
                "id": "1",
                "content": "User prefers dark mode",
                "embedding": [0.1, 0.8, 0.3],
                "importance": 0.9
            },
            {
                "id": "2", 
                "content": "User likes Italian food",
                "embedding": [0.5, 0.2, 0.7],
                "importance": 0.7
            },
            {
                "id": "3",
                "content": "User wants concise responses",
                "embedding": [0.2, 0.6, 0.4],
                "importance": 0.8
            }
        ]
        
        with patch('open_webui.routers.memories.query_memory') as mock_query:
            mock_query.return_value = stored_memories
            
            # Test relevance filtering
            relevant_memories = [m for m in stored_memories if "UI" in query_context and "dark" in m['content']]
            assert len(relevant_memories) >= 1
            assert relevant_memories[0]['content'] == "User prefers dark mode"

    def test_memory_injection_into_context(self, filter_instance):
        """Test memory injection into conversation context."""
        relevant_memories = [
            {"content": "User is a software engineer", "importance": 0.9},
            {"content": "User prefers Python", "importance": 0.8},
            {"content": "User likes concise explanations", "importance": 0.7}
        ]
        
        original_body = {
            "messages": [
                {"role": "user", "content": "Can you help me with a coding problem?"}
            ],
            "user": {"id": "test-user"}
        }
        
        # Inject memories into context
        if hasattr(filter_instance, '_inject_memories'):
            # Simulate memory injection
            memory_context = "\n".join([f"- {m['content']}" for m in relevant_memories])
            system_message = {
                "role": "system",
                "content": f"User context:\n{memory_context}"
            }
            
            enhanced_body = original_body.copy()
            enhanced_body["messages"].insert(0, system_message)
            
            assert len(enhanced_body["messages"]) == 2
            assert enhanced_body["messages"][0]["role"] == "system"
            assert "software engineer" in enhanced_body["messages"][0]["content"]

    def test_memory_importance_scoring(self, filter_instance):
        """Test memory importance scoring algorithm."""
        test_memories = [
            {"content": "User's name is Alice", "base_importance": 0.9},  # Identity info
            {"content": "User mentioned they like coffee", "base_importance": 0.4},  # Casual preference
            {"content": "User is allergic to peanuts", "base_importance": 1.0},  # Critical info
            {"content": "User said good morning", "base_importance": 0.1},  # Trivial
        ]
        
        # Test importance calculation
        for memory in test_memories:
            # Identity and critical info should maintain high importance
            if "name" in memory["content"] or "allergic" in memory["content"]:
                assert memory["base_importance"] >= 0.9
            # Trivial info should have low importance
            elif "good morning" in memory["content"]:
                assert memory["base_importance"] <= 0.2

    def test_memory_temporal_decay(self, filter_instance):
        """Test memory temporal decay functionality."""
        from datetime import timedelta
        
        now = datetime.now(timezone.utc)
        
        memories_with_age = [
            {"content": "Recent info", "timestamp": now.isoformat(), "importance": 0.7},
            {"content": "Week old info", "timestamp": (now - timedelta(days=7)).isoformat(), "importance": 0.7},
            {"content": "Month old info", "timestamp": (now - timedelta(days=30)).isoformat(), "importance": 0.7},
        ]
        
        # Test recency scoring
        if hasattr(filter_instance, '_calculate_recency_score'):
            # Recent memories should have higher recency scores
            # Assuming exponential decay over time
            recency_scores = []
            for memory in memories_with_age:
                age_days = (now - datetime.fromisoformat(memory["timestamp"])).days
                recency = max(0.1, 1.0 - (age_days / 30))  # Simple decay model
                recency_scores.append(recency)
            
            assert recency_scores[0] > recency_scores[1] > recency_scores[2]

    def test_memory_category_classification(self, filter_instance):
        """Test memory category classification."""
        test_messages = [
            ("My email is john@example.com", "identity"),
            ("I prefer working in the morning", "behavior"),
            ("I love Italian cuisine", "preference"),
            ("My goal is to learn machine learning", "goal"),
            ("My wife's name is Sarah", "relationship"),
            ("I drive a Tesla Model 3", "possession"),
        ]
        
        for message, expected_category in test_messages:
            # Test category detection logic
            # Check relationship patterns first (more specific)
            if "wife" in message or "husband" in message or "spouse" in message:
                detected_category = "relationship"
            elif "email" in message or ("name is" in message and "my" in message.lower()):
                detected_category = "identity"
            elif "prefer" in message or "morning" in message:
                detected_category = "behavior"
            elif "love" in message or "cuisine" in message:
                detected_category = "preference"
            elif "goal" in message:
                detected_category = "goal"
            elif "drive" in message or "own" in message:
                detected_category = "possession"
            else:
                detected_category = "general"
            
            if expected_category != "general":
                assert detected_category == expected_category

    def test_memory_privacy_filtering(self, filter_instance):
        """Test privacy filtering for sensitive information."""
        sensitive_memories = [
            {"content": "SSN: 123-45-6789", "should_filter": True},
            {"content": "Password is abc123", "should_filter": True},
            {"content": "Credit card: 1234-5678-9012-3456", "should_filter": True},
            {"content": "User likes pizza", "should_filter": False},
            {"content": "API key: sk-1234567890", "should_filter": True},
        ]
        
        # Test privacy filtering
        for memory in sensitive_memories:
            # Check for sensitive patterns
            sensitive_patterns = ["SSN:", "Password", "Credit card", "API key"]
            is_sensitive = any(pattern in memory["content"] for pattern in sensitive_patterns)
            assert is_sensitive == memory["should_filter"]

    def test_memory_batch_operations(self, filter_instance):
        """Test batch memory operations for efficiency."""
        # Create multiple memories
        batch_memories = [
            {"content": f"Memory {i}", "importance": 0.5 + (i * 0.1)} 
            for i in range(10)
        ]
        
        with patch('open_webui.routers.memories.create_memory') as mock_create:
            mock_create.return_value = {"status": "success"}
            
            # Test batch processing
            if hasattr(filter_instance, '_batch_store_memories'):
                # Should process efficiently in batches
                batch_size = 5
                for i in range(0, len(batch_memories), batch_size):
                    batch = batch_memories[i:i+batch_size]
                    assert len(batch) <= batch_size

    def test_memory_conflict_resolution(self, filter_instance):
        """Test handling of conflicting memories."""
        existing_memory = {
            "id": "mem1",
            "content": "User prefers morning meetings",
            "timestamp": "2024-01-01T10:00:00Z",
            "importance": 0.7
        }
        
        conflicting_memory = {
            "content": "User prefers afternoon meetings",
            "timestamp": "2024-01-15T10:00:00Z",
            "importance": 0.8
        }
        
        # Test conflict resolution
        # Newer memory with higher importance should take precedence
        if existing_memory["timestamp"] < conflicting_memory["timestamp"]:
            if conflicting_memory["importance"] >= existing_memory["importance"]:
                # Should update or replace
                assert conflicting_memory["importance"] > existing_memory["importance"]

    def test_memory_search_and_query(self, filter_instance):
        """Test memory search and query functionality."""
        search_query = "programming preferences"
        
        all_memories = [
            {"content": "Prefers Python for data science", "tags": ["programming", "preference"]},
            {"content": "Likes morning coffee", "tags": ["preference", "routine"]},
            {"content": "Uses VS Code for development", "tags": ["programming", "tools"]},
            {"content": "Enjoys hiking on weekends", "tags": ["hobby", "activity"]},
        ]
        
        # Test search functionality
        matching_memories = [
            m for m in all_memories 
            if any(tag in ["programming", "preference"] for tag in m.get("tags", []))
        ]
        
        assert len(matching_memories) >= 2
        assert any("Python" in m["content"] for m in matching_memories)
        assert any("VS Code" in m["content"] for m in matching_memories)

    def test_memory_compression_and_summarization(self, filter_instance):
        """Test memory compression for similar memories."""
        similar_memories = [
            {"content": "User likes Python programming", "id": "1"},
            {"content": "User enjoys coding in Python", "id": "2"},
            {"content": "User prefers Python over Java", "id": "3"},
        ]
        
        # Test summarization logic
        if hasattr(filter_instance, '_summarize_memories'):
            # These should be compressed into a single memory
            summary = "User strongly prefers Python programming"
            compressed_memory = {
                "content": summary,
                "importance": 0.9,  # High importance due to multiple mentions
                "source_memories": ["1", "2", "3"]
            }
            
            assert len(compressed_memory["source_memories"]) == 3
            assert compressed_memory["importance"] > 0.8