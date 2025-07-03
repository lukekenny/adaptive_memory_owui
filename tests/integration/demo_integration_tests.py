#!/usr/bin/env python3
"""
Demonstration script for OpenWebUI Adaptive Memory Integration Tests.

This script provides a guided tour of the integration testing capabilities,
showing various test scenarios and usage patterns.
"""

import asyncio
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.integration.fixtures import (
    generate_test_user,
    generate_test_memories,
    generate_test_messages
)
from tests.integration.mocks.openwebui_api_mock import OpenWebUIMemoryAPIMock, APIError
from tests.integration.test_config import IntegrationTestConfig


async def demo_basic_mock_usage():
    """Demonstrate basic mock server usage"""
    print("=== Demo: Basic Mock Server Usage ===")
    
    # Create mock server
    api_mock = OpenWebUIMemoryAPIMock(
        enable_rate_limiting=False,
        enable_random_errors=False,
        response_delay_ms=10
    )
    
    # Create test user
    user = generate_test_user("demo_user")
    user_id = user["id"]
    print(f"Created test user: {user_id}")
    
    # Create memories
    memories = []
    for i in range(3):
        memory = await api_mock.create_memory(
            user_id=user_id,
            content=f"Demo memory {i}: User likes {['coffee', 'tea', 'water'][i]}",
            metadata={"category": "preferences", "demo": True}
        )
        memories.append(memory)
        print(f"Created memory {i}: {memory.content}")
    
    # Retrieve memories
    retrieved = await api_mock.get_memories(user_id)
    print(f"Retrieved {len(retrieved)} memories")
    
    # Search memories
    search_results = await api_mock.search_memories(
        user_id=user_id,
        query="coffee",
        limit=10
    )
    print(f"Search for 'coffee' found {len(search_results)} results")
    
    print("✓ Basic mock usage demo completed\n")


async def demo_error_injection():
    """Demonstrate error injection capabilities"""
    print("=== Demo: Error Injection ===")
    
    # Create mock with error injection
    api_mock = OpenWebUIMemoryAPIMock(
        enable_random_errors=True,
        error_rate=0.5,  # 50% error rate
        response_delay_ms=50
    )
    
    user_id = "error_demo_user"
    
    # Try operations with errors
    successful_operations = 0
    failed_operations = 0
    
    for i in range(10):
        try:
            memory = await api_mock.create_memory(
                user_id=user_id,
                content=f"Error demo memory {i}"
            )
            successful_operations += 1
            print(f"  ✓ Operation {i} succeeded")
        except Exception as e:
            failed_operations += 1
            print(f"  ✗ Operation {i} failed: {type(e).__name__}")
    
    print(f"Results: {successful_operations} succeeded, {failed_operations} failed")
    print("✓ Error injection demo completed\n")


async def demo_rate_limiting():
    """Demonstrate rate limiting"""
    print("=== Demo: Rate Limiting ===")
    
    # Create mock with rate limiting
    api_mock = OpenWebUIMemoryAPIMock(
        enable_rate_limiting=True,
        enable_random_errors=False
    )
    
    # Configure rate limiter
    api_mock.rate_limiter.max_requests = 5
    api_mock.rate_limiter.window_seconds = 10
    
    user_id = "rate_limit_demo_user"
    
    # Make requests until rate limited
    for i in range(8):
        try:
            memory = await api_mock.create_memory(
                user_id=user_id,
                content=f"Rate limit test {i}"
            )
            print(f"  ✓ Request {i} succeeded")
        except Exception as e:
            print(f"  ✗ Request {i} rate limited: {type(e).__name__}")
            break
    
    print("✓ Rate limiting demo completed\n")


async def demo_concurrent_operations():
    """Demonstrate concurrent operations"""
    print("=== Demo: Concurrent Operations ===")
    
    # Create mock server
    api_mock = OpenWebUIMemoryAPIMock(
        enable_rate_limiting=False,
        enable_random_errors=False,
        response_delay_ms=100  # Add some delay to make concurrency visible
    )
    
    user_id = "concurrent_demo_user"
    
    # Define concurrent tasks
    async def create_memory_task(index: int):
        return await api_mock.create_memory(
            user_id=user_id,
            content=f"Concurrent memory {index}",
            metadata={"index": index, "concurrent": True}
        )
    
    # Run concurrent operations
    import time
    start_time = time.time()
    
    tasks = [create_memory_task(i) for i in range(10)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    # Analyze results
    successful = [r for r in results if not isinstance(r, Exception)]
    errors = [r for r in results if isinstance(r, Exception)]
    
    print(f"Concurrent operations completed in {end_time - start_time:.2f} seconds")
    print(f"Successful: {len(successful)}, Errors: {len(errors)}")
    
    # Verify data integrity
    all_memories = await api_mock.get_memories(user_id)
    indexes = [m.metadata.get("index") for m in all_memories if "index" in m.metadata]
    unique_indexes = set(indexes)
    
    print(f"Created {len(all_memories)} memories with {len(unique_indexes)} unique indexes")
    print("✓ Concurrent operations demo completed\n")


async def demo_batch_operations():
    """Demonstrate batch operations"""
    print("=== Demo: Batch Operations ===")
    
    api_mock = OpenWebUIMemoryAPIMock(
        enable_rate_limiting=False,
        enable_random_errors=False,
        response_delay_ms=10
    )
    
    user_id = "batch_demo_user"
    
    # Generate test memories
    memories_data = []
    for i in range(50):
        memories_data.append({
            "content": f"Batch memory {i}: Information about topic {i % 10}",
            "metadata": {"batch": True, "topic": i % 10}
        })
    
    # Batch create
    import time
    start_time = time.time()
    
    created_memories = await api_mock.bulk_create_memories(
        user_id=user_id,
        memories=memories_data
    )
    
    creation_time = time.time() - start_time
    
    print(f"Created {len(created_memories)} memories in {creation_time:.2f} seconds")
    print(f"Average time per memory: {(creation_time / len(created_memories)) * 1000:.1f}ms")
    
    # Test pagination
    page_size = 10
    all_retrieved = []
    page = 0
    
    while True:
        batch = await api_mock.get_memories(
            user_id=user_id,
            limit=page_size,
            offset=page * page_size
        )
        if not batch:
            break
        all_retrieved.extend(batch)
        page += 1
        print(f"  Retrieved page {page}: {len(batch)} memories")
    
    print(f"Total retrieved: {len(all_retrieved)} memories")
    print("✓ Batch operations demo completed\n")


def demo_test_configuration():
    """Demonstrate test configuration"""
    print("=== Demo: Test Configuration ===")
    
    # Show available scenarios
    scenarios = IntegrationTestConfig.SCENARIOS
    print(f"Available test scenarios ({len(scenarios)}):")
    for scenario in scenarios:
        print(f"  - {scenario.name}: {scenario.description}")
    
    # Show specific scenario details
    happy_path = IntegrationTestConfig.get_scenario("happy_path")
    print(f"\nHappy path scenario details:")
    print(f"  API config: {happy_path.api_configs}")
    print(f"  Error injection: {happy_path.error_injection}")
    print(f"  Data volumes: {happy_path.data_volumes}")
    
    # Show API configurations
    api_configs = IntegrationTestConfig.OPENWEBUI_API_CONFIGS
    print(f"\nAvailable API configurations:")
    for version, config in api_configs.items():
        print(f"  {version}: {config['base_url']}{config['memory_endpoint']}")
    
    print("✓ Test configuration demo completed\n")


def demo_test_data_generation():
    """Demonstrate test data generation"""
    print("=== Demo: Test Data Generation ===")
    
    # Generate users
    users = [generate_test_user(f"demo_user_{i}") for i in range(3)]
    print(f"Generated {len(users)} test users:")
    for user in users:
        print(f"  {user['id']}: {user['name']} ({user['email']})")
    
    # Generate memories
    memories = generate_test_memories(10, users[0]["id"])
    print(f"\nGenerated {len(memories)} test memories:")
    for i, memory in enumerate(memories[:3]):  # Show first 3
        print(f"  {memory['id']}: {memory['content'][:50]}...")
    print(f"  ... and {len(memories) - 3} more")
    
    # Generate messages
    messages = generate_test_messages(5, "demo_chat")
    print(f"\nGenerated {len(messages)} test messages:")
    for message in messages:
        print(f"  {message['role']}: {message['content']}")
    
    print("✓ Test data generation demo completed\n")


async def run_all_demos():
    """Run all demonstration scenarios"""
    print("OpenWebUI Adaptive Memory Integration Tests - Demo\n")
    print("This demo shows the capabilities of the integration testing framework.\n")
    
    demos = [
        demo_basic_mock_usage,
        demo_error_injection,
        demo_rate_limiting,
        demo_concurrent_operations,
        demo_batch_operations,
    ]
    
    for demo in demos:
        try:
            await demo()
        except Exception as e:
            print(f"Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Non-async demos
    demo_test_configuration()
    demo_test_data_generation()
    
    print("=== Demo Summary ===")
    print("The integration testing framework provides:")
    print("✓ Mock OpenWebUI API server with full CRUD operations")
    print("✓ Error injection for testing resilience")
    print("✓ Rate limiting simulation")
    print("✓ Concurrent operation support")
    print("✓ Batch operation testing")
    print("✓ Flexible test configuration")
    print("✓ Test data generation utilities")
    print("✓ Performance monitoring")
    print("✓ Request/response recording")
    print("\nUse these tools to build comprehensive integration tests!")


if __name__ == "__main__":
    asyncio.run(run_all_demos())