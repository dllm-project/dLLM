"""
dLLM API Routes

OpenAI-compatible API endpoints for chat, completions, embeddings, and models.
"""

from .chat import router as chat_router
from .completions import router as completions_router
from .embeddings import router as embeddings_router
from .models import router as models_router

__all__ = [
    "chat_router",
    "completions_router",
    "embeddings_router",
    "models_router",
]
