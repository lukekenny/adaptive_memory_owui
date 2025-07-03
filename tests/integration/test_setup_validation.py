"""
Setup validation tests for integration testing framework.

This module validates that the integration test framework is properly
configured and all components are working correctly.
"""

import pytest
import asyncio
import sys
import os
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.integration.fixtures import (
    openwebui_memory_api_mock,
    openwebui_client,
    generate_test_user,
    generate_test_memories
)
from tests.integration.mocks.openwebui_api_mock import APIError


class TestSetupValidation:
    """Validate integration test setup"""
    
    @pytest.mark.asyncio
    async def test_openwebui_mock_basic_functionality(self, openwebui_memory_api_mock):
        """Test that OpenWebUI mock server works correctly"""
        api_mock = openwebui_memory_api_mock
        
        # Test user creation
        user_id = "setup_test_user"
        
        # Test memory creation
        memory = await api_mock.create_memory(
            user_id=user_id,
            content="Setup validation memory"
        )
        
        assert memory is not None
        assert memory.user_id == user_id
        assert memory.content == "Setup validation memory"
        assert hasattr(memory, 'id')
        assert hasattr(memory, 'timestamp')
    
    @pytest.mark.asyncio
    async def test_memory_crud_operations(self, openwebui_memory_api_mock):
        """Test basic CRUD operations work"""
        api_mock = openwebui_memory_api_mock
        user_id = "crud_test_user"
        
        # Create
        memory = await api_mock.create_memory(
            user_id=user_id,
            content="CRUD test memory",
            metadata={"test": "crud"}
        )
        memory_id = memory.id
        
        # Read
        retrieved = await api_mock.get_memory(memory_id, user_id)
        assert retrieved.content == "CRUD test memory"
        assert retrieved.metadata["test"] == "crud"
        
        # Update
        updated = await api_mock.update_memory(
            memory_id=memory_id,
            user_id=user_id,
            content="Updated CRUD memory"
        )
        assert updated.content == "Updated CRUD memory"
        
        # Delete
        await api_mock.delete_memory(memory_id, user_id)
        
        # Verify deletion
        memories = await api_mock.get_memories(user_id)
        assert not any(m.id == memory_id for m in memories)
    
    @pytest.mark.asyncio
    async def test_error_injection_works(self, openwebui_memory_api_mock):
        """Test that error injection mechanisms work"""
        api_mock = openwebui_memory_api_mock
        
        # Enable errors
        api_mock.enable_random_errors = True
        api_mock.error_sequence = [APIError.SERVER_ERROR, None]
        
        user_id = "error_test_user"
        
        # First call should fail
        try:
            await api_mock.create_memory(
                user_id=user_id,
                content="Should fail"
            )
            # If we get here, error injection didn't work
            assert False, "Expected error was not raised"
        except Exception as e:
            # Expected error
            assert "server_error" in str(e).lower() or "500" in str(e)
        
        # Second call should succeed
        memory = await api_mock.create_memory(
            user_id=user_id,
            content="Should succeed"
        )
        assert memory is not None
    
    @pytest.mark.asyncio
    async def test_test_data_generators(self):
        """Test that test data generators work correctly"""
        # Test user generation
        user = generate_test_user("test_user_123")
        assert user["id"] == "test_user_123"
        assert "name" in user
        assert "email" in user
        assert user["email"].endswith("@test.example.com")
        
        # Test memory generation
        memories = generate_test_memories(5, "test_user")
        assert len(memories) == 5
        assert all(m["user_id"] == "test_user" for m in memories)
        assert all("content" in m for m in memories)
        assert all("timestamp" in m for m in memories)
    
    @pytest.mark.asyncio
    async def test_async_operation_support(self, openwebui_memory_api_mock):
        """Test that async operations work correctly"""
        api_mock = openwebui_memory_api_mock
        user_id = "async_test_user"
        
        # Test concurrent operations
        async def create_memory(index: int):
            return await api_mock.create_memory(
                user_id=user_id,
                content=f"Async memory {index}"
            )
        
        # Run multiple operations concurrently
        tasks = [create_memory(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify all succeeded
        assert len(results) == 5
        assert all(r is not None for r in results)
        assert all(f"Async memory {i}" in r.content for i, r in enumerate(results))
    
    @pytest.mark.asyncio
    async def test_fixture_isolation(self, openwebui_memory_api_mock):
        """Test that fixtures provide proper isolation between tests"""
        api_mock = openwebui_memory_api_mock
        
        # Check initial state is clean
        all_users_memories = []
        for user_num in range(3):
            user_id = f"isolation_test_user_{user_num}"
            memories = await api_mock.get_memories(user_id)
            all_users_memories.extend(memories)
        
        # Should start clean (or only have test data from fixtures)
        # The exact count depends on fixture setup
        initial_count = len(all_users_memories)
        
        # Add some memories
        user_id = "isolation_test_user_new"
        await api_mock.create_memory(
            user_id=user_id,
            content="Isolation test memory"
        )
        
        # Verify only this test's data is present for this user
        user_memories = await api_mock.get_memories(user_id)
        assert len(user_memories) == 1
        assert user_memories[0].content == "Isolation test memory"
    
    def test_import_structure(self):
        """Test that all required modules can be imported"""
        # Test fixture imports
        from tests.integration.fixtures import (
            openwebui_memory_api_mock,
            generate_test_user,
            generate_test_memories
        )
        
        # Test mock imports
        from tests.integration.mocks.openwebui_api_mock import (
            OpenWebUIMemoryAPIMock,
            APIError
        )
        
        # Test config imports
        from tests.integration.test_config import (
            IntegrationTestConfig,
            EnvironmentConfig
        )
        
        # If we get here, all imports work
        assert True
    
    def test_configuration_loading(self):
        """Test that configuration loads correctly"""
        from tests.integration.test_config import IntegrationTestConfig
        
        # Test scenario loading
        scenarios = IntegrationTestConfig.SCENARIOS
        assert len(scenarios) > 0
        
        # Test specific scenario
        happy_path = IntegrationTestConfig.get_scenario("happy_path")
        assert happy_path.name == "happy_path"
        assert happy_path.description is not None
        
        # Test API config
        api_config = IntegrationTestConfig.get_api_config("v1")
        assert "base_url" in api_config
        assert "memory_endpoint" in api_config


class TestFrameworkResilience:
    """Test framework resilience and error handling"""
    
    @pytest.mark.asyncio
    async def test_mock_recovery_from_errors(self, openwebui_memory_api_mock):
        """Test that mocks can recover from error conditions"""
        api_mock = openwebui_memory_api_mock
        user_id = "recovery_test_user"
        
        # Force an error condition
        api_mock.enable_random_errors = True
        api_mock.error_rate = 1.0  # 100% error rate
        
        # Should fail
        with pytest.raises(Exception):
            await api_mock.create_memory(
                user_id=user_id,
                content="Should fail"
            )
        
        # Disable errors
        api_mock.enable_random_errors = False
        api_mock.error_rate = 0.0
        
        # Should now succeed
        memory = await api_mock.create_memory(
            user_id=user_id,
            content="Should succeed"
        )
        assert memory is not None
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, openwebui_memory_api_mock):
        """Test timeout handling in mocks"""
        api_mock = openwebui_memory_api_mock
        user_id = "timeout_test_user"
        
        # Set long delay
        original_delay = api_mock.response_delay_ms
        api_mock.response_delay_ms = 2000  # 2 seconds
        
        try:
            # Should timeout
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    api_mock.create_memory(
                        user_id=user_id,
                        content="Timeout test"
                    ),
                    timeout=0.5  # 0.5 second timeout
                )
        finally:
            # Restore original delay
            api_mock.response_delay_ms = original_delay


if __name__ == "__main__":
    # Run validation tests
    pytest.main([__file__, "-v"])