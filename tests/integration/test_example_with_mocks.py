"""
Example integration tests demonstrating the use of mock servers.

This module shows how to use the mock servers and fixtures
for integration testing of the adaptive memory plugin.
"""

import pytest
import asyncio
from typing import Dict, Any
from datetime import datetime, timezone

from tests.integration.mocks import (
    APIError,
    EventType,
    LLMProvider
)


class TestMemoryAPIIntegration:
    """Test memory API integration with mocks"""
    
    @pytest.mark.asyncio
    async def test_memory_creation_success(self, openwebui_memory_api_mock):
        """Test successful memory creation"""
        # Create a memory
        result = await openwebui_memory_api_mock.create_memory(
            user_id="test_user",
            content="User prefers dark mode",
            metadata={"category": "preferences"}
        )
        
        # Verify response
        assert "id" in result
        assert result["user_id"] == "test_user"
        assert result["content"] == "User prefers dark mode"
        assert "timestamp" in result
        
        # Verify memory was stored
        memories = await openwebui_memory_api_mock.get_memories("test_user")
        assert memories["total"] == 1
        assert memories["memories"][0]["content"] == "User prefers dark mode"
    
    @pytest.mark.asyncio
    async def test_memory_search(self, memory_search_scenario):
        """Test memory search functionality"""
        user_id = memory_search_scenario["user_id"]
        api_mock = memory_search_scenario["api_mock"]
        
        # Search for programming-related memories
        results = await api_mock.query_memory(
            user_id=user_id,
            query="programming",
            limit=5
        )
        
        # Should find Python and Rust memories
        assert len(results["results"]) >= 2
        contents = [r["content"] for r in results["results"]]
        assert any("Python" in c for c in contents)
        assert any("Rust" in c for c in contents)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, openwebui_memory_api_mock_with_errors):
        """Test rate limiting behavior"""
        api_mock = openwebui_memory_api_mock_with_errors
        user_id = "rate_limit_test"
        
        # Make requests until rate limited
        for i in range(15):
            result = await api_mock.create_memory(
                user_id=user_id,
                content=f"Memory {i}"
            )
            
            if "error" in result and result["error"] == APIError.RATE_LIMIT.value:
                # Should be rate limited after 10 requests
                assert i >= 10
                assert result["retry_after"] == 60
                break
        else:
            pytest.fail("Rate limiting not triggered")
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self, batch_memory_scenario):
        """Test bulk memory operations"""
        user_id = batch_memory_scenario["user_id"]
        memories = batch_memory_scenario["memories"]
        api_mock = batch_memory_scenario["api_mock"]
        
        # Bulk create memories
        result = await api_mock.bulk_create_memories(
            user_id=user_id,
            memories=memories[:10]  # Create first 10
        )
        
        assert result["total_requested"] == 10
        assert result["total_created"] == 10
        assert len(result["errors"]) == 0
        
        # Verify they were created
        stored = await api_mock.get_memories(user_id, limit=20)
        assert stored["total"] == 10


class TestLLMAPIIntegration:
    """Test LLM API integration with mocks"""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("llm_client", [
        LLMProvider.OPENAI,
        LLMProvider.OLLAMA,
        LLMProvider.ANTHROPIC
    ], indirect=True)
    async def test_chat_completion(self, llm_client):
        """Test chat completion across different providers"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ]
        
        response = await llm_client.post(
            "/v1/chat/completions",
            json={
                "messages": messages,
                "model": "test-model",
                "temperature": 0.7
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Common response structure
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert len(data["choices"][0]["message"]["content"]) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, openai_api_mock):
        """Test streaming chat completion"""
        messages = [{"role": "user", "content": "Tell me a story"}]
        
        # Get streaming response
        response = await openai_api_mock.create_chat_completion(
            messages=messages,
            model="gpt-3.5-turbo",
            stream=True
        )
        
        # Collect chunks
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
        
        # Should have multiple chunks
        assert len(chunks) > 2
        
        # First chunk should have role
        assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
        
        # Last chunk should have finish reason
        assert chunks[-1]["choices"][0]["finish_reason"] is not None
    
    @pytest.mark.asyncio
    async def test_custom_response_patterns(self, openai_api_mock):
        """Test custom response patterns"""
        # Set custom response
        openai_api_mock.set_response_pattern(
            "weather",
            "The weather today is sunny with a high of 75°F."
        )
        
        response = await openai_api_mock.create_chat_completion(
            messages=[{"role": "user", "content": "What's the weather like?"}],
            model="gpt-4"
        )
        
        content = response["choices"][0]["message"]["content"]
        assert "sunny" in content
        assert "75°F" in content


class TestEmbeddingAPIIntegration:
    """Test embedding API integration with mocks"""
    
    @pytest.mark.asyncio
    async def test_embedding_generation(self, embedding_api_mock):
        """Test embedding generation"""
        texts = [
            "Machine learning is fascinating",
            "Neural networks are complex",
            "Deep learning requires GPUs"
        ]
        
        result = await embedding_api_mock.create_embeddings(
            input=texts,
            model="text-embedding-ada-002"
        )
        
        assert result["object"] == "list"
        assert len(result["data"]) == 3
        
        # Check embedding structure
        for i, embedding_data in enumerate(result["data"]):
            assert embedding_data["object"] == "embedding"
            assert embedding_data["index"] == i
            assert len(embedding_data["embedding"]) == 1536  # Ada-002 dimensions
    
    @pytest.mark.asyncio
    async def test_embedding_similarity(self, embedding_api_mock):
        """Test embedding similarity computation"""
        # Create embeddings for similar texts
        result1 = await embedding_api_mock.create_embeddings(
            input="I love programming in Python",
            model="text-embedding-ada-002"
        )
        
        result2 = await embedding_api_mock.create_embeddings(
            input="Python programming is my favorite",
            model="text-embedding-ada-002"
        )
        
        result3 = await embedding_api_mock.create_embeddings(
            input="I enjoy cooking Italian food",
            model="text-embedding-ada-002"
        )
        
        # Extract embeddings
        emb1 = result1["data"][0]["embedding"]
        emb2 = result2["data"][0]["embedding"]
        emb3 = result3["data"][0]["embedding"]
        
        # Compute similarities
        sim_12 = embedding_api_mock.compute_similarity(emb1, emb2)
        sim_13 = embedding_api_mock.compute_similarity(emb1, emb3)
        
        # Similar texts should have higher similarity
        assert sim_12 > sim_13
    
    @pytest.mark.asyncio
    async def test_local_embedding_model(self, local_embedding_model):
        """Test local embedding model mock"""
        sentences = [
            "This is a test sentence",
            "Another example text",
            "Machine learning is powerful"
        ]
        
        # Generate embeddings
        embeddings = local_embedding_model.encode(sentences)
        
        assert embeddings.shape == (3, 384)  # MiniLM dimensions
        
        # Test single sentence
        single_emb = local_embedding_model.encode("Single sentence")
        assert single_emb.shape == (1, 384)


class TestWebSocketIntegration:
    """Test WebSocket integration with mocks"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, websocket_server):
        """Test WebSocket connection lifecycle"""
        # Create connection
        conn_id = await websocket_server.create_connection("test_user")
        
        # Verify connection
        info = websocket_server.get_connection_info(conn_id)
        assert info["user_id"] == "test_user"
        assert info["state"] == "open"
        
        # Send message
        success = await websocket_server.send_message(
            conn_id,
            EventType.SYSTEM_MESSAGE,
            {"content": "Test message"}
        )
        assert success
        
        # Close connection
        await websocket_server.close_connection(conn_id)
        
        # Verify closed
        info = websocket_server.get_connection_info(conn_id)
        assert info is None
    
    @pytest.mark.asyncio
    async def test_conversation_streaming(self, conversation_scenario):
        """Test streaming conversation updates"""
        conn_id = conversation_scenario["connection_id"]
        chat_id = conversation_scenario["chat_id"]
        server = conversation_scenario["server"]
        
        # Simulate streaming response
        await server.simulate_chat_stream(
            conn_id,
            chat_id,
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            chunks=5
        )
        
        # Get sent messages
        messages = server.get_messages(conn_id, "sent")
        
        # Should have initial messages + streaming chunks + final
        stream_messages = [
            m for m in messages 
            if m.type == EventType.ASSISTANT_MESSAGE and m.chat_id == chat_id
        ]
        
        # At least 5 chunks + 1 complete message
        assert len(stream_messages) >= 6
        
        # Last message should be complete
        last_msg = stream_messages[-1]
        assert last_msg.data.get("complete") is True
    
    @pytest.mark.asyncio
    async def test_websocket_client(self, websocket_server):
        """Test WebSocket client functionality"""
        client = MockWebSocketClient(websocket_server, "client_user")
        
        # Connect
        connected = await client.connect()
        assert connected
        assert client.is_connected
        
        # Set up message handler
        received_messages = []
        client.on_message(lambda msg: received_messages.append(msg))
        
        # Send message
        await client.send(
            EventType.USER_MESSAGE,
            {"content": "Hello from client"}
        )
        
        # Server should receive it
        await asyncio.sleep(0.2)  # Wait for async processing
        
        server_messages = websocket_server.get_messages(
            client.connection_id, 
            "received"
        )
        assert len(server_messages) > 0
        assert server_messages[-1].data["content"] == "Hello from client"
        
        # Disconnect
        await client.disconnect()
        assert not client.is_connected


class TestIntegrationWithFilter:
    """Test full integration with the adaptive memory filter"""
    
    @pytest.mark.asyncio
    async def test_filter_with_mocked_apis(self, mock_filter_with_apis, 
                                         openwebui_memory_api_mock,
                                         basic_message_body):
        """Test filter operations with mocked external APIs"""
        filter_instance = mock_filter_with_apis
        
        # Process message through filter
        result = await filter_instance.inlet(
            body=basic_message_body,
            __user__={"id": basic_message_body["user"]["id"]}
        )
        
        # Should process successfully
        assert result is not None
        assert "messages" in result
        
        # Check if memory operations were attempted
        # (This would depend on the filter's implementation)
    
    @pytest.mark.asyncio
    async def test_error_handling_cascade(self, mock_filter_with_apis,
                                        openwebui_memory_api_mock_with_errors,
                                        basic_message_body):
        """Test error handling with unreliable APIs"""
        filter_instance = mock_filter_with_apis
        
        # Set up error sequence
        openwebui_memory_api_mock_with_errors.set_error_sequence([
            APIError.SERVER_ERROR,
            APIError.TIMEOUT,
            None,  # Success on third try
        ])
        
        # Process should handle errors gracefully
        result = await filter_instance.inlet(
            body=basic_message_body,
            __user__={"id": basic_message_body["user"]["id"]}
        )
        
        # Should still return a result despite errors
        assert result is not None


class TestRecordingAndReplay:
    """Test request/response recording functionality"""
    
    @pytest.mark.asyncio
    async def test_recording_session(self, request_recorder, 
                                   openwebui_memory_api_mock):
        """Test recording API interactions"""
        # Make some API calls
        await openwebui_memory_api_mock.create_memory(
            user_id="record_test",
            content="Test memory for recording"
        )
        
        await openwebui_memory_api_mock.get_memories(
            user_id="record_test"
        )
        
        # Get recordings
        recordings = request_recorder["openwebui"].get_recording()
        
        assert recordings["total_requests"] >= 2
        assert len(recordings["requests"]) >= 2
        assert len(recordings["responses"]) >= 2
        
        # Verify request details
        create_req = recordings["requests"][0]
        assert create_req["endpoint"] == "/memories/create"
        assert create_req["method"] == "POST"
        assert create_req["data"]["content"] == "Test memory for recording"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])