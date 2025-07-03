"""
Dependency Fallback for OWUI Adaptive Memory Plugin

This module provides fallback implementations when heavy dependencies are not available.
"""

import logging
import numpy as np
from typing import List, Optional, Any

logger = logging.getLogger(__name__)

class FallbackEmbeddings:
    """Fallback embedding implementation using simple text hashing."""
    
    def __init__(self, model_name: str = "fallback"):
        self.model_name = model_name
        logger.warning("Using fallback embeddings - install sentence-transformers for better results")
        
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Create simple hash-based embeddings."""
        embeddings = []
        for text in texts:
            # Simple character frequency-based embedding
            char_counts = [0] * 256
            for char in text.lower():
                char_counts[ord(char) % 256] += 1
            
            # Normalize to create embedding vector
            total = sum(char_counts) or 1
            embedding = [count / total for count in char_counts]
            embeddings.append(embedding)
        
        return np.array(embeddings)

def get_fallback_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """Get fallback embeddings when sentence-transformers is not available."""
    return FallbackEmbeddings(model_name)

# Test if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using fallback")
