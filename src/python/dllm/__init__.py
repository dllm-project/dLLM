"""
dLLM - Distributed CPU AI Inference Engine

A high-performance distributed inference engine with SIMD optimization
and OpenAI-compatible API.
"""

__version__ = "1.0.0"
__author__ = "dLLM Team"
__license__ = "MIT"

from .backend_connector import BackendConnector, ModelFormat
from .api_routes import chat, completions, embeddings, models

__all__ = [
    "BackendConnector",
    "ModelFormat",
    "chat",
    "completions",
    "embeddings",
    "models",
]
