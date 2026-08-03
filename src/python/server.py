"""
dLLM - Distributed CPU AI Inference Engine
OpenAI-compatible API Server

This is a modular FastAPI server that uses route modules for better organization.
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Import modular routes
from api_routes.chat import router as chat_router
from api_routes.completions import router as completions_router
from api_routes.embeddings import router as embeddings_router
from api_routes.models import router as models_router

app = FastAPI(
    title="dLLM API",
    description="Distributed CPU AI inference engine with OpenAI-compatible API",
    version="1.0.0"
)

# Include routers with appropriate prefixes
app.include_router(chat_router, prefix="/api")
app.include_router(completions_router, prefix="/api")
app.include_router(embeddings_router, prefix="/api")
app.include_router(models_router, prefix="/api")

# Keep original endpoints for backward compatibility
class Message(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "backend": "connected"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "dLLM API",
        "version": "1.0.0",
        "description": "Distributed CPU AI inference engine",
        "endpoints": [
            "/api/chat/completions",
            "/api/completions",
            "/api/embeddings", 
            "/api/models"
        ]
    }


if __name__ == "__main__":
    port = int(os.environ.get("DLLM_PORT", 8000))
    host = os.environ.get("DLLM_HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )