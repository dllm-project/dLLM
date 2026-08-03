"""
dLLM API Routes Package

This package contains modular FastAPI routes for the dLLM backend service.
Each module implements a specific OpenAI-compatible endpoint group.

Modules:
    chat - Chat completions endpoints (/v1/chat/completions)
    completions - Text completion endpoints (/v1/completions)
    embeddings - Embedding generation endpoints (/v1/embeddings)
    models - Model management endpoints (/v1/models)
"""

from .chat import router as chat_router
from .completions import router as completions_router
from .embeddings import router as embeddings_router
from .models import router as models_router

__all__ = [
    'chat_router',
    'completions_router', 
    'embeddings_router',
    'models_router'
]