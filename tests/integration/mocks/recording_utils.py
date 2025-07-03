"""
Utilities for recording and replaying API requests/responses.

This module provides tools for capturing real API interactions
and converting them to test fixtures.
"""

import json
import os
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pickle
import gzip
import asyncio
from pathlib import Path


class RecordingMode(Enum):
    """Recording modes for API interactions"""
    RECORD = "record"      # Record new interactions
    REPLAY = "replay"      # Replay recorded interactions
    PASSTHROUGH = "passthrough"  # Pass through to real API
    MIXED = "mixed"        # Record new, replay existing


@dataclass
class RecordedRequest:
    """Recorded API request"""
    timestamp: float
    method: str
    url: str
    headers: Dict[str, str]
    body: Optional[Any] = None
    params: Optional[Dict[str, Any]] = None
    request_id: str = field(default_factory=lambda: f"req_{time.time()}")


@dataclass
class RecordedResponse:
    """Recorded API response"""
    timestamp: float
    status_code: int
    headers: Dict[str, str]
    body: Any
    elapsed_ms: float
    request_id: str
    error: Optional[str] = None


@dataclass
class RecordingSession:
    """Recording session containing multiple request/response pairs"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    mode: RecordingMode = RecordingMode.RECORD
    interactions: List[Tuple[RecordedRequest, RecordedResponse]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecordingManager:
    """Manages recording and replay of API interactions"""
    
    def __init__(self, 
                 storage_dir: str = "./test_recordings",
                 mode: RecordingMode = RecordingMode.RECORD,
                 compression: bool = True):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.compression = compression
        self.current_session: Optional[RecordingSession] = None
        self.loaded_sessions: Dict[str, RecordingSession] = {}
        
    def start_session(self, session_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start a new recording session"""
        session_id = session_id or f"session_{int(time.time())}"
        
        self.current_session = RecordingSession(
            session_id=session_id,
            start_time=time.time(),
            mode=self.mode,
            metadata=metadata or {}
        )
        
        return session_id
    
    def end_session(self) -> Optional[str]:
        """End current recording session and save"""
        if not self.current_session:
            return None
        
        self.current_session.end_time = time.time()
        session_id = self.current_session.session_id
        
        # Save session
        self.save_session(self.current_session)
        
        self.current_session = None
        return session_id
    
    def record_interaction(self, request: RecordedRequest, 
                         response: RecordedResponse):
        """Record a request/response interaction"""
        if not self.current_session:
            self.start_session()
        
        self.current_session.interactions.append((request, response))
    
    def find_matching_response(self, request: RecordedRequest) -> Optional[RecordedResponse]:
        """Find a matching recorded response for a request"""
        # Create request signature
        signature = self._create_request_signature(request)
        
        # Search in loaded sessions
        for session in self.loaded_sessions.values():
            for req, resp in session.interactions:
                if self._create_request_signature(req) == signature:
                    return resp
        
        return None
    
    def _create_request_signature(self, request: RecordedRequest) -> str:
        """Create a unique signature for a request"""
        # Combine key request attributes
        sig_data = {
            "method": request.method,
            "url": request.url,
            "body": request.body,
            "params": request.params
        }
        
        # Create hash
        sig_str = json.dumps(sig_data, sort_keys=True)
        return hashlib.sha256(sig_str.encode()).hexdigest()
    
    def save_session(self, session: RecordingSession):
        """Save a recording session to disk"""
        filename = f"{session.session_id}.{'pkl.gz' if self.compression else 'pkl'}"
        filepath = self.storage_dir / filename
        
        # Convert to serializable format
        session_data = asdict(session)
        
        if self.compression:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(session_data, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(session_data, f)
    
    def load_session(self, session_id: str) -> Optional[RecordingSession]:
        """Load a recording session from disk"""
        # Try compressed first
        filename = f"{session_id}.pkl.gz"
        filepath = self.storage_dir / filename
        
        if not filepath.exists():
            # Try uncompressed
            filename = f"{session_id}.pkl"
            filepath = self.storage_dir / filename
            
            if not filepath.exists():
                return None
        
        try:
            if filename.endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    session_data = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    session_data = pickle.load(f)
            
            # Reconstruct session
            session = RecordingSession(**session_data)
            self.loaded_sessions[session_id] = session
            
            return session
            
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None
    
    def load_all_sessions(self):
        """Load all available recording sessions"""
        for filepath in self.storage_dir.glob("*.pkl*"):
            session_id = filepath.stem.replace('.pkl', '')
            if session_id not in self.loaded_sessions:
                self.load_session(session_id)
    
    def export_to_json(self, session_id: str, output_file: str):
        """Export a session to JSON format for inspection"""
        session = self.loaded_sessions.get(session_id)
        if not session:
            session = self.load_session(session_id)
            
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Convert to JSON-serializable format
        export_data = {
            "session_id": session.session_id,
            "start_time": datetime.fromtimestamp(session.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(session.end_time).isoformat() if session.end_time else None,
            "mode": session.mode.value,
            "metadata": session.metadata,
            "interactions": []
        }
        
        for req, resp in session.interactions:
            interaction = {
                "request": {
                    "timestamp": datetime.fromtimestamp(req.timestamp).isoformat(),
                    "method": req.method,
                    "url": req.url,
                    "headers": req.headers,
                    "body": req.body,
                    "params": req.params
                },
                "response": {
                    "timestamp": datetime.fromtimestamp(resp.timestamp).isoformat(),
                    "status_code": resp.status_code,
                    "headers": resp.headers,
                    "body": resp.body,
                    "elapsed_ms": resp.elapsed_ms,
                    "error": resp.error
                }
            }
            export_data["interactions"].append(interaction)
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def get_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about recorded sessions"""
        if session_id:
            sessions = [self.loaded_sessions.get(session_id)]
            if not sessions[0]:
                session = self.load_session(session_id)
                sessions = [session] if session else []
        else:
            sessions = list(self.loaded_sessions.values())
        
        if not sessions:
            return {"error": "No sessions found"}
        
        stats = {
            "total_sessions": len(sessions),
            "total_interactions": sum(len(s.interactions) for s in sessions),
            "by_method": {},
            "by_status": {},
            "average_response_time_ms": 0,
            "error_count": 0
        }
        
        total_time = 0
        interaction_count = 0
        
        for session in sessions:
            for req, resp in session.interactions:
                # Count by method
                stats["by_method"][req.method] = stats["by_method"].get(req.method, 0) + 1
                
                # Count by status
                status_group = f"{resp.status_code // 100}xx"
                stats["by_status"][status_group] = stats["by_status"].get(status_group, 0) + 1
                
                # Response time
                total_time += resp.elapsed_ms
                interaction_count += 1
                
                # Errors
                if resp.error or resp.status_code >= 400:
                    stats["error_count"] += 1
        
        if interaction_count > 0:
            stats["average_response_time_ms"] = total_time / interaction_count
        
        return stats


class RecordingInterceptor:
    """Interceptor for recording HTTP client calls"""
    
    def __init__(self, recording_manager: RecordingManager):
        self.recording_manager = recording_manager
        self.original_client = None
    
    async def intercept_request(self, method: str, url: str, **kwargs) -> Any:
        """Intercept and record/replay HTTP requests"""
        # Create recorded request
        request = RecordedRequest(
            timestamp=time.time(),
            method=method.upper(),
            url=url,
            headers=kwargs.get("headers", {}),
            body=kwargs.get("json") or kwargs.get("data"),
            params=kwargs.get("params")
        )
        
        start_time = time.time()
        
        if self.recording_manager.mode == RecordingMode.REPLAY:
            # Try to find matching response
            recorded_response = self.recording_manager.find_matching_response(request)
            if recorded_response:
                # Simulate delay
                await asyncio.sleep(recorded_response.elapsed_ms / 1000.0)
                
                # Create mock response
                mock_response = MockHTTPResponse(
                    status_code=recorded_response.status_code,
                    headers=recorded_response.headers,
                    body=recorded_response.body
                )
                return mock_response
        
        # Fall through to real request or error
        if self.recording_manager.mode == RecordingMode.RECORD or \
           self.recording_manager.mode == RecordingMode.MIXED:
            # Make real request (would need actual client)
            if self.original_client:
                response = await self.original_client.request(method, url, **kwargs)
                
                # Record the interaction
                elapsed_ms = (time.time() - start_time) * 1000
                recorded_response = RecordedResponse(
                    timestamp=time.time(),
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    body=response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    elapsed_ms=elapsed_ms,
                    request_id=request.request_id
                )
                
                self.recording_manager.record_interaction(request, recorded_response)
                
                return response
        
        # No recorded response and not recording
        raise ValueError(f"No recorded response found for {method} {url}")


class MockHTTPResponse:
    """Mock HTTP response for replay"""
    
    def __init__(self, status_code: int, headers: Dict[str, str], body: Any):
        self.status_code = status_code
        self.headers = headers
        self._body = body
    
    def json(self):
        """Return JSON body"""
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body
    
    @property
    def text(self):
        """Return text body"""
        if isinstance(self._body, str):
            return self._body
        return json.dumps(self._body)
    
    async def aread(self):
        """Async read for compatibility"""
        return self.text.encode()


# Utility functions for test data generation from recordings
def generate_test_fixtures_from_recording(session_id: str, 
                                        storage_dir: str = "./test_recordings",
                                        output_file: str = "generated_fixtures.py"):
    """Generate pytest fixtures from recorded session"""
    manager = RecordingManager(storage_dir)
    session = manager.load_session(session_id)
    
    if not session:
        raise ValueError(f"Session {session_id} not found")
    
    fixture_code = '''"""
Generated test fixtures from recording: {session_id}
Generated at: {timestamp}
"""

import pytest
from typing import Dict, Any

'''
    
    fixture_code = fixture_code.format(
        session_id=session_id,
        timestamp=datetime.now().isoformat()
    )
    
    # Group interactions by endpoint
    endpoints = {}
    for req, resp in session.interactions:
        endpoint_key = f"{req.method}_{req.url.split('/')[-1]}"
        if endpoint_key not in endpoints:
            endpoints[endpoint_key] = []
        endpoints[endpoint_key].append((req, resp))
    
    # Generate fixtures
    for endpoint_key, interactions in endpoints.items():
        fixture_name = f"recorded_{endpoint_key.lower()}_responses"
        
        fixture_code += f'''
@pytest.fixture
def {fixture_name}():
    """Recorded responses for {endpoint_key}"""
    return [
'''
        
        for req, resp in interactions[:5]:  # Limit to 5 examples
            fixture_code += f'''        {{
            "request": {{
                "method": "{req.method}",
                "url": "{req.url}",
                "body": {json.dumps(req.body, indent=16)}
            }},
            "response": {{
                "status_code": {resp.status_code},
                "body": {json.dumps(resp.body, indent=16)}
            }}
        }},
'''
        
        fixture_code += '''    ]
'''
    
    with open(output_file, 'w') as f:
        f.write(fixture_code)
    
    print(f"Generated fixtures written to {output_file}")