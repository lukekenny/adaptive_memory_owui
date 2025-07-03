"""
OpenWebUI Adaptive Memory Plugin - Minimal Test Version
This is a simplified version for testing installation and basic functionality.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Filter:
    """
    Minimal Adaptive Memory Filter for OpenWebUI
    
    This simplified version tests basic installation and functionality.
    """
    
    class Valves(BaseModel):
        """Configuration options for the filter."""
        
        enable_memory: bool = Field(
            default=True,
            description="Enable memory functionality"
        )
        test_mode: bool = Field(
            default=True,
            description="Run in test mode with minimal processing"
        )
        debug_logging: bool = Field(
            default=True,
            description="Enable debug logging"
        )
    
    def __init__(self):
        """Initialize the filter."""
        try:
            self.valves = self.Valves()
            logger.info("Minimal Adaptive Memory Filter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize filter: {e}")
            # Create default valves if initialization fails
            self.valves = type('DefaultValves', (), {
                'enable_memory': True,
                'test_mode': True,
                'debug_logging': True
            })()
    
    def inlet(self, body: dict) -> dict:
        """Process user input."""
        try:
            if self.valves.debug_logging:
                logger.info("Inlet called - processing user input")
            
            if not self.valves.enable_memory:
                return body
            
            # Simple test processing - just log the input
            if isinstance(body, dict) and "messages" in body:
                logger.info(f"Processing {len(body['messages'])} messages")
            
            return body
            
        except Exception as e:
            logger.error(f"Error in inlet: {e}")
            return body
    
    def outlet(self, body: dict) -> dict:
        """Process model output."""
        try:
            if self.valves.debug_logging:
                logger.info("Outlet called - processing model output")
            
            if not self.valves.enable_memory:
                return body
            
            # Simple test processing - just log the output
            if isinstance(body, dict) and "messages" in body:
                logger.info(f"Processing {len(body['messages'])} messages in outlet")
            
            return body
            
        except Exception as e:
            logger.error(f"Error in outlet: {e}")
            return body
    
    def stream(self, event: dict) -> dict:
        """Process streaming events."""
        try:
            if self.valves.debug_logging:
                logger.debug("Stream event processed")
            
            return event
            
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            return event