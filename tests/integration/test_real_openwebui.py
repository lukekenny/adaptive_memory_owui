"""
Real OpenWebUI Integration Tests

These tests run against an actual OpenWebUI instance to validate
the plugin works correctly in the real environment.
"""

import pytest
import asyncio
import aiohttp
import time
import json
from pathlib import Path


class RealOpenWebUITester:
    """Test against real OpenWebUI instance"""
    
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def wait_for_openwebui(self, timeout: int = 60):
        """Wait for OpenWebUI to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                async with self.session.get(f"{self.base_url}/api/v1/auths") as response:
                    if response.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(2)
        return False
    
    async def install_plugin(self, plugin_file: str):
        """Install the adaptive memory plugin"""
        plugin_path = Path(plugin_file)
        if not plugin_path.exists():
            raise FileNotFoundError(f"Plugin file not found: {plugin_file}")
            
        with open(plugin_path, 'r') as f:
            plugin_content = f.read()
        
        # Upload plugin via API
        async with self.session.post(
            f"{self.base_url}/api/v1/functions/create",
            json={
                "name": "adaptive_memory_v4.0",
                "content": plugin_content,
                "is_active": True
            }
        ) as response:
            return response.status == 200
    
    async def send_message(self, message: str, model: str = "gpt-3.5-turbo"):
        """Send a message through OpenWebUI"""
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": message}
            ],
            "stream": False
        }
        
        async with self.session.post(
            f"{self.base_url}/api/v1/chat/completions",
            json=payload
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception(f"Message failed: {response.status}")
    
    async def get_memories(self, user_id: str):
        """Get user memories"""
        async with self.session.get(
            f"{self.base_url}/api/v1/memories",
            params={"user_id": user_id}
        ) as response:
            if response.status == 200:
                return await response.json()
            return []


@pytest.mark.asyncio
@pytest.mark.real_openwebui
class TestRealOpenWebUIIntegration:
    """Test against real OpenWebUI instance"""
    
    async def test_openwebui_connection(self):
        """Test basic connection to OpenWebUI"""
        async with RealOpenWebUITester() as tester:
            is_ready = await tester.wait_for_openwebui()
            assert is_ready, "OpenWebUI is not responding"
    
    async def test_plugin_installation(self):
        """Test installing the adaptive memory plugin"""
        async with RealOpenWebUITester() as tester:
            await tester.wait_for_openwebui()
            
            # Install plugin
            success = await tester.install_plugin("../../adaptive_memory_v4.0.py")
            assert success, "Failed to install plugin"
    
    async def test_memory_extraction_real_conversation(self):
        """Test memory extraction in real conversation"""
        async with RealOpenWebUITester() as tester:
            await tester.wait_for_openwebui()
            await tester.install_plugin("../../adaptive_memory_v4.0.py")
            
            # Send message with extractable information
            response = await tester.send_message(
                "Hi, my name is Alice and I'm a software engineer. I love Python programming."
            )
            
            assert response is not None
            assert "choices" in response
            
            # Check if memories were created
            memories = await tester.get_memories("test_user")
            
            # Should have extracted some memories
            assert len(memories) > 0
            
            # Should contain name and profession
            memory_content = " ".join([m.get("content", "") for m in memories])
            assert "Alice" in memory_content
            assert "software engineer" in memory_content.lower()
    
    async def test_memory_injection_real_conversation(self):
        """Test memory injection in subsequent conversations"""
        async with RealOpenWebUITester() as tester:
            await tester.wait_for_openwebui()
            await tester.install_plugin("../../adaptive_memory_v4.0.py")
            
            # First conversation - establish context
            await tester.send_message(
                "My favorite programming language is Python and I work on web applications."
            )
            
            # Second conversation - should have injected context
            response = await tester.send_message(
                "What should I learn next for my career?"
            )
            
            # The response should be contextually aware
            assert response is not None
            response_text = response["choices"][0]["message"]["content"].lower()
            
            # Should reference programming context
            assert any(keyword in response_text for keyword in [
                "python", "web", "programming", "developer"
            ])
    
    async def test_filter_orchestration_real_environment(self):
        """Test filter orchestration in real OpenWebUI"""
        async with RealOpenWebUITester() as tester:
            await tester.wait_for_openwebui()
            await tester.install_plugin("../../adaptive_memory_v4.0.py")
            
            # Send message and ensure no conflicts
            response = await tester.send_message(
                "I prefer concise explanations and work in the AI field."
            )
            
            assert response is not None
            
            # Should process without orchestration conflicts
            assert "error" not in str(response).lower()
    
    async def test_performance_real_load(self):
        """Test performance under real load"""
        async with RealOpenWebUITester() as tester:
            await tester.wait_for_openwebui()
            await tester.install_plugin("../../adaptive_memory_v4.0.py")
            
            # Send multiple messages concurrently
            tasks = []
            for i in range(5):
                task = tester.send_message(f"This is message {i} for performance testing.")
                tasks.append(task)
            
            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # All requests should succeed
            successful_responses = [r for r in responses if not isinstance(r, Exception)]
            assert len(successful_responses) >= 4, "Most requests should succeed"
            
            # Should handle 5 concurrent requests in reasonable time
            assert total_time < 30, f"Took too long: {total_time}s"
    
    async def test_error_handling_real_environment(self):
        """Test error handling in real environment"""
        async with RealOpenWebUITester() as tester:
            await tester.wait_for_openwebui()
            await tester.install_plugin("../../adaptive_memory_v4.0.py")
            
            # Send problematic messages
            problematic_messages = [
                "",  # Empty message
                "x" * 10000,  # Very long message
                "SELECT * FROM users; DROP TABLE users;",  # SQL injection attempt
                "<script>alert('xss')</script>",  # XSS attempt
            ]
            
            for message in problematic_messages:
                try:
                    response = await tester.send_message(message)
                    # Should handle gracefully
                    assert response is not None
                except Exception as e:
                    # Should not crash OpenWebUI
                    assert "Internal Server Error" not in str(e)


@pytest.mark.asyncio  
@pytest.mark.manual
class TestManualRealWorldValidation:
    """Manual tests that require human validation"""
    
    async def test_manual_conversation_quality(self):
        """Manual test: Verify conversation quality with memory"""
        print("\n" + "="*60)
        print("MANUAL TEST: Conversation Quality with Memory")
        print("="*60)
        
        async with RealOpenWebUITester() as tester:
            await tester.wait_for_openwebui()
            await tester.install_plugin("../../adaptive_memory_v4.0.py")
            
            print("\nStep 1: Visit http://localhost:3000")
            print("Step 2: Start a conversation:")
            print("  - Say: 'Hi, I'm a Python developer working on AI projects'")
            print("  - Continue conversation naturally")
            print("Step 3: Start a new conversation:")
            print("  - Ask: 'What programming topics should I focus on?'")
            print("  - Verify the response shows awareness of your background")
            
            print("\nExpected: Second conversation should reference your Python/AI background")
            print("Manual verification required!")
            
            # Just ensure the system is ready
            assert await tester.wait_for_openwebui()


if __name__ == "__main__":
    # Quick validation script
    async def main():
        print("Testing real OpenWebUI integration...")
        
        async with RealOpenWebUITester() as tester:
            if await tester.wait_for_openwebui():
                print("✅ OpenWebUI is running")
                
                if await tester.install_plugin("../../adaptive_memory_v4.0.py"):
                    print("✅ Plugin installed successfully")
                else:
                    print("❌ Plugin installation failed")
                    
            else:
                print("❌ OpenWebUI is not running")
                print("Start with: docker-compose -f docker-compose.test.yml up")
    
    asyncio.run(main())