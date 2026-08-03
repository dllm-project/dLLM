"""
dLLM - Distributed CPU AI Inference Engine
OpenAI-compatible API Server
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Import backend connector
try:
    from backend_connector import BackendConnector
except ImportError as e:
    print(f"Warning: Could not import backend_connector: {e}")
    BackendConnector = None


app = FastAPI(
    title="dLLM API",
    description="Distributed CPU AI inference engine with OpenAI-compatible API",
    version="1.0.0"
)

# Initialize backend connector
if BackendConnector:
    backend = BackendConnector()
else:
    backend = None


# ============================================
# Request/Response Models (Pydantic)
# ============================================

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

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]

class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "dLLM"

class ModelsListResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]


# ============================================
# API Endpoints
# ============================================

@app.get("/v1/models")
async def list_models():
    """List available models"""
    if not backend:
        raise HTTPException(status_code=503, detail="Backend not available")
    
    models = backend.list_models()
    return ModelsListResponse(data=models)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Create chat completion"""
    if not backend:
        raise HTTPException(status_code=503, detail="Backend not available")
    
    try:
        # Prepare messages for inference
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        result = backend.chat(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        return {
            "id": f"chatcmpl-{hash(str(messages)) % 100000}",
            "object": "chat.completion",
            "created": 0,  # Timestamp would be set here in production
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(messages[0]["content"].split()),
                "completion_tokens": len(result.split()),
                "total_tokens": 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Create text completion"""
    if not backend:
        raise HTTPException(status_code=503, detail="Backend not available")
    
    try:
        result = backend.infer(
            model=request.model,
            prompt=request.prompt,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        
        return {
            "id": f"cmpl-{hash(request.prompt) % 100000}",
            "object": "text_completion",
            "created": 0,
            "model": request.model,
            "choices": [{
                "index": 0,
                "text": result,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(request.prompt.split()),
                "completion_tokens": len(result.split()),
                "total_tokens": 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Create embeddings"""
    if not backend:
        raise HTTPException(status_code=503, detail="Backend not available")
    
    try:
        # Get embedding for input
        embedding = backend.get_embedding(
            input_text=request.input,
            model=request.model
        )
        
        return {
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": embedding,
                "index": 0
            }],
            "model": request.model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Utility Endpoints
# ============================================

@app.get("/health")
async def health():
    """Health check endpoint"""
    if backend and backend.is_ready():
        return {"status": "healthy", "backend": "connected"}
    else:
        return {"status": "unhealthy", "backend": "disconnected"}


if __name__ == "__main__":
    port = int(os.environ.get("DLLM_PORT", 8000))
    host = os.environ.get("DLLM_HOST", "0.0.0.0")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )