"""
Mock WebSocket server for event testing.

This module provides a mock WebSocket server implementation
for testing real-time event handling in the adaptive memory plugin.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import random
from unittest.mock import Mock, AsyncMock
import threading
from collections import defaultdict


class WebSocketState(Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class EventType(Enum):
    """Types of WebSocket events"""
    MESSAGE = "message"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_MESSAGE = "system_message"
    MEMORY_UPDATE = "memory_update"
    FILTER_EVENT = "filter_event"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    id: str
    type: EventType
    data: Dict[str, Any]
    timestamp: str
    user_id: Optional[str] = None
    chat_id: Optional[str] = None


@dataclass
class WebSocketConnection:
    """Mock WebSocket connection"""
    id: str
    user_id: str
    state: WebSocketState = WebSocketState.OPEN
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    received_messages: List[WebSocketMessage] = field(default_factory=list)
    sent_messages: List[WebSocketMessage] = field(default_factory=list)


class WebSocketServerMock:
    """Mock WebSocket server for testing"""
    
    def __init__(self,
                 enable_auto_ping: bool = True,
                 ping_interval: int = 30,
                 enable_random_disconnects: bool = False,
                 disconnect_rate: float = 0.05,
                 enable_message_delays: bool = False,
                 message_delay_ms: int = 0):
        self.enable_auto_ping = enable_auto_ping
        self.ping_interval = ping_interval
        self.enable_random_disconnects = enable_random_disconnects
        self.disconnect_rate = disconnect_rate
        self.enable_message_delays = enable_message_delays
        self.message_delay_ms = message_delay_ms
        
        self.connections: Dict[str, WebSocketConnection] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.broadcast_groups: Dict[str, Set[str]] = defaultdict(set)
        
        self._lock = threading.Lock()
        self._running = False
        self._ping_task = None
        
        # Message queue for async processing
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Recording
        self.enable_recording = False
        self.recorded_events: List[Dict[str, Any]] = []
    
    async def start(self):
        """Start the mock server"""
        self._running = True
        
        if self.enable_auto_ping:
            self._ping_task = asyncio.create_task(self._ping_loop())
    
    async def stop(self):
        """Stop the mock server"""
        self._running = False
        
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        with self._lock:
            for conn_id in list(self.connections.keys()):
                await self.close_connection(conn_id)
    
    async def _ping_loop(self):
        """Send periodic ping messages"""
        while self._running:
            await asyncio.sleep(self.ping_interval)
            
            with self._lock:
                for conn_id, conn in list(self.connections.items()):
                    if conn.state == WebSocketState.OPEN:
                        await self.send_ping(conn_id)
    
    async def _simulate_delay(self):
        """Simulate message delay"""
        if self.enable_message_delays and self.message_delay_ms > 0:
            await asyncio.sleep(self.message_delay_ms / 1000.0)
    
    def _should_disconnect(self) -> bool:
        """Determine if connection should randomly disconnect"""
        return (self.enable_random_disconnects and 
                random.random() < self.disconnect_rate)
    
    def _record_event(self, event_type: str, data: Any):
        """Record event for testing"""
        if self.enable_recording:
            self.recorded_events.append({
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "data": data
            })
    
    async def create_connection(self, user_id: str) -> str:
        """Create a new WebSocket connection"""
        conn_id = f"ws_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            conn = WebSocketConnection(
                id=conn_id,
                user_id=user_id
            )
            self.connections[conn_id] = conn
        
        self._record_event("connection_created", {
            "connection_id": conn_id,
            "user_id": user_id
        })
        
        # Send connection established message
        await self.send_message(conn_id, EventType.SYSTEM_MESSAGE, {
            "content": "WebSocket connection established",
            "connection_id": conn_id
        })
        
        return conn_id
    
    async def close_connection(self, conn_id: str, code: int = 1000, 
                             reason: str = "Normal closure"):
        """Close a WebSocket connection"""
        with self._lock:
            if conn_id not in self.connections:
                return
            
            conn = self.connections[conn_id]
            conn.state = WebSocketState.CLOSED
            
            # Remove from broadcast groups
            for group in self.broadcast_groups.values():
                group.discard(conn_id)
        
        self._record_event("connection_closed", {
            "connection_id": conn_id,
            "code": code,
            "reason": reason
        })
        
        # Notify handlers
        await self._trigger_handlers(EventType.SYSTEM_MESSAGE, {
            "content": "WebSocket connection closed",
            "connection_id": conn_id,
            "code": code,
            "reason": reason
        })
        
        with self._lock:
            del self.connections[conn_id]
    
    async def send_message(self, conn_id: str, event_type: EventType,
                         data: Dict[str, Any], chat_id: Optional[str] = None) -> bool:
        """Send a message to a specific connection"""
        await self._simulate_delay()
        
        with self._lock:
            if conn_id not in self.connections:
                return False
            
            conn = self.connections[conn_id]
            
            if conn.state != WebSocketState.OPEN:
                return False
            
            # Check for random disconnect
            if self._should_disconnect():
                await self.close_connection(conn_id, 1006, "Abnormal closure")
                return False
            
            # Create message
            message = WebSocketMessage(
                id=f"msg_{uuid.uuid4().hex[:8]}",
                type=event_type,
                data=data,
                timestamp=datetime.utcnow().isoformat(),
                user_id=conn.user_id,
                chat_id=chat_id
            )
            
            conn.sent_messages.append(message)
            conn.last_activity = time.time()
        
        self._record_event("message_sent", {
            "connection_id": conn_id,
            "message": message.__dict__
        })
        
        # Trigger handlers
        await self._trigger_handlers(event_type, data, conn_id)
        
        return True
    
    async def receive_message(self, conn_id: str, event_type: EventType,
                            data: Dict[str, Any]) -> bool:
        """Simulate receiving a message from a connection"""
        with self._lock:
            if conn_id not in self.connections:
                return False
            
            conn = self.connections[conn_id]
            
            if conn.state != WebSocketState.OPEN:
                return False
            
            # Create message
            message = WebSocketMessage(
                id=f"msg_{uuid.uuid4().hex[:8]}",
                type=event_type,
                data=data,
                timestamp=datetime.utcnow().isoformat(),
                user_id=conn.user_id
            )
            
            conn.received_messages.append(message)
            conn.last_activity = time.time()
        
        self._record_event("message_received", {
            "connection_id": conn_id,
            "message": message.__dict__
        })
        
        # Add to message queue
        await self.message_queue.put((conn_id, message))
        
        return True
    
    async def broadcast_message(self, event_type: EventType, data: Dict[str, Any],
                              group: Optional[str] = None,
                              exclude_connections: Optional[List[str]] = None) -> int:
        """Broadcast a message to multiple connections"""
        exclude_connections = exclude_connections or []
        sent_count = 0
        
        with self._lock:
            if group:
                target_connections = list(self.broadcast_groups.get(group, []))
            else:
                target_connections = list(self.connections.keys())
        
        for conn_id in target_connections:
            if conn_id not in exclude_connections:
                success = await self.send_message(conn_id, event_type, data)
                if success:
                    sent_count += 1
        
        return sent_count
    
    async def send_ping(self, conn_id: str) -> bool:
        """Send a ping message"""
        return await self.send_message(conn_id, EventType.PING, {
            "timestamp": time.time()
        })
    
    async def send_pong(self, conn_id: str, ping_data: Dict[str, Any]) -> bool:
        """Send a pong message in response to ping"""
        return await self.send_message(conn_id, EventType.PONG, {
            "timestamp": time.time(),
            "ping_timestamp": ping_data.get("timestamp")
        })
    
    def add_to_group(self, conn_id: str, group: str):
        """Add connection to a broadcast group"""
        with self._lock:
            if conn_id in self.connections:
                self.broadcast_groups[group].add(conn_id)
    
    def remove_from_group(self, conn_id: str, group: str):
        """Remove connection from a broadcast group"""
        with self._lock:
            self.broadcast_groups[group].discard(conn_id)
    
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register an event handler"""
        self.event_handlers[event_type].append(handler)
    
    def unregister_handler(self, event_type: EventType, handler: Callable):
        """Unregister an event handler"""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
    
    async def _trigger_handlers(self, event_type: EventType, data: Dict[str, Any],
                              conn_id: Optional[str] = None):
        """Trigger registered event handlers"""
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data, conn_id)
                else:
                    handler(event_type, data, conn_id)
            except Exception as e:
                self._record_event("handler_error", {
                    "event_type": event_type.value,
                    "error": str(e)
                })
    
    def get_connection_info(self, conn_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a connection"""
        with self._lock:
            if conn_id not in self.connections:
                return None
            
            conn = self.connections[conn_id]
            return {
                "id": conn.id,
                "user_id": conn.user_id,
                "state": conn.state.value,
                "created_at": conn.created_at,
                "last_activity": conn.last_activity,
                "messages_sent": len(conn.sent_messages),
                "messages_received": len(conn.received_messages)
            }
    
    def get_all_connections(self) -> List[str]:
        """Get all active connection IDs"""
        with self._lock:
            return list(self.connections.keys())
    
    def get_messages(self, conn_id: str, 
                    direction: str = "both") -> List[WebSocketMessage]:
        """Get messages for a connection"""
        with self._lock:
            if conn_id not in self.connections:
                return []
            
            conn = self.connections[conn_id]
            
            if direction == "sent":
                return conn.sent_messages.copy()
            elif direction == "received":
                return conn.received_messages.copy()
            else:  # both
                all_messages = conn.sent_messages + conn.received_messages
                return sorted(all_messages, key=lambda m: m.timestamp)
    
    async def simulate_chat_stream(self, conn_id: str, chat_id: str,
                                 message: str, chunks: int = 10) -> bool:
        """Simulate streaming chat response"""
        if conn_id not in self.connections:
            return False
        
        # Split message into chunks
        words = message.split()
        chunk_size = max(1, len(words) // chunks)
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            
            await self.send_message(conn_id, EventType.ASSISTANT_MESSAGE, {
                "chat_id": chat_id,
                "content": chunk,
                "streaming": True,
                "chunk_index": i // chunk_size,
                "total_chunks": chunks
            }, chat_id=chat_id)
            
            await asyncio.sleep(0.05)  # Simulate streaming delay
        
        # Send completion message
        await self.send_message(conn_id, EventType.ASSISTANT_MESSAGE, {
            "chat_id": chat_id,
            "content": message,
            "streaming": False,
            "complete": True
        }, chat_id=chat_id)
        
        return True
    
    def reset(self):
        """Reset mock to initial state"""
        with self._lock:
            # Close all connections
            for conn_id in list(self.connections.keys()):
                asyncio.create_task(self.close_connection(conn_id))
            
            self.connections.clear()
            self.broadcast_groups.clear()
            self.recorded_events.clear()
            self.event_handlers.clear()
    
    def get_recording(self) -> Dict[str, Any]:
        """Get recorded events"""
        return {
            "events": self.recorded_events.copy(),
            "total_events": len(self.recorded_events),
            "connections": len(self.connections),
            "active_groups": list(self.broadcast_groups.keys())
        }


class MockWebSocketClient:
    """Mock WebSocket client for testing"""
    
    def __init__(self, server: WebSocketServerMock, user_id: str):
        self.server = server
        self.user_id = user_id
        self.connection_id: Optional[str] = None
        self.received_messages: List[WebSocketMessage] = []
        self._message_handler: Optional[Callable] = None
        self._closed = False
    
    async def connect(self) -> bool:
        """Connect to the mock server"""
        if self.connection_id:
            return False
        
        self.connection_id = await self.server.create_connection(self.user_id)
        self._closed = False
        
        # Start message receiver
        asyncio.create_task(self._receive_loop())
        
        return True
    
    async def disconnect(self, code: int = 1000, reason: str = "Normal closure"):
        """Disconnect from the server"""
        if not self.connection_id:
            return
        
        await self.server.close_connection(self.connection_id, code, reason)
        self.connection_id = None
        self._closed = True
    
    async def send(self, event_type: EventType, data: Dict[str, Any]) -> bool:
        """Send a message to the server"""
        if not self.connection_id:
            return False
        
        return await self.server.receive_message(self.connection_id, event_type, data)
    
    async def _receive_loop(self):
        """Receive messages from the server"""
        while not self._closed and self.connection_id:
            try:
                # Check for messages
                messages = self.server.get_messages(self.connection_id, "sent")
                
                # Process new messages
                if len(messages) > len(self.received_messages):
                    new_messages = messages[len(self.received_messages):]
                    self.received_messages.extend(new_messages)
                    
                    # Trigger handler for new messages
                    if self._message_handler:
                        for msg in new_messages:
                            if asyncio.iscoroutinefunction(self._message_handler):
                                await self._message_handler(msg)
                            else:
                                self._message_handler(msg)
                
                await asyncio.sleep(0.1)  # Polling interval
                
            except Exception:
                break
    
    def on_message(self, handler: Callable):
        """Set message handler"""
        self._message_handler = handler
    
    @property
    def is_connected(self) -> bool:
        """Check if connected"""
        return self.connection_id is not None and not self._closed