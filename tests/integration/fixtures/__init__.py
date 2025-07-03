"""
Fixtures for integration testing.
"""

from .llm_fixtures import *

# Import fixtures from the main fixtures.py file
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fixtures import (
        generate_test_user,
        generate_test_chat,
        generate_test_messages,
        generate_test_memories,
        openwebui_memory_api_mock,
        openwebui_memory_api_mock_with_errors,
        openwebui_client,
        mock_httpx_client,
        memory_search_scenario,
        batch_memory_scenario,
        error_scenarios,
    )
except ImportError:
    # Fallback implementations
    import uuid
    from datetime import datetime, timezone
    from typing import Dict, List, Any, Optional
    
    def generate_test_user(user_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate test user data"""
        user_id = user_id or f"test_user_{uuid.uuid4().hex[:8]}"
        return {
            "id": user_id,
            "name": f"Test User {user_id[-4:]}",
            "email": f"{user_id}@test.example.com",
            "role": "user",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    
    def generate_test_memories(count: int = 10, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate test memory data"""
        user_id = user_id or f"test_user_{uuid.uuid4().hex[:8]}"
        memories = []
        
        contexts = ["preferences", "knowledge", "experience", "relationships", "goals"]
        
        for i in range(count):
            memories.append({
                "id": f"mem_{uuid.uuid4().hex[:8]}",
                "user_id": user_id,
                "content": f"Test memory {i+1}: User information about {contexts[i % len(contexts)]}",
                "importance": 0.5 + (i % 5) * 0.1,
                "context": contexts[i % len(contexts)],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "source": "test",
                    "confidence": 0.8,
                    "tags": [f"tag_{i}", "test"]
                }
            })
        
        return memories
    
    def generate_test_messages(count: int = 5, chat_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate test message data"""
        chat_id = chat_id or f"test_chat_{uuid.uuid4().hex[:8]}"
        messages = []
        
        for i in range(count):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "role": role,
                "content": f"Test message {i+1} from {role}",
                "chat_id": chat_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        return messages