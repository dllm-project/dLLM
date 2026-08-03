"""
dLLM Chat Completions Route
OpenAI-compatible /v1/chat/completions endpoint
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time

from ..backend_connector import BackendConnector

router = APIRouter()

# Request/Response models matching OpenAI API
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


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Create chat completion
    
    This endpoint is compatible with OpenAI's /v1/chat/completions API.
    It processes chat conversations and generates responses using the dLLM backend.
    """
    if not BackendConnector:
        raise HTTPException(status_code=503, detail="Backend not available")
    
    try:
        # Initialize backend connector
        backend = BackendConnector()
        
        if not backend.is_ready():
            raise HTTPException(status_code=503, detail="Backend not initialized")
        
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
            "id": f"chatcmpl-{int(time.time() * 1000) % 100000}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(messages[0]["content"].split()) if messages else 0,
                "completion_tokens": len(result.split()),
                "total_tokens": 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/chat/models")
async def list_chat_models():
    """
    List available chat models
    
    Returns a list of models that can be used with the chat completions endpoint.
    """
    if not BackendConnector:
        raise HTTPException(status_code=503, detail="Backend not available")
    
    try:
        backend = BackendConnector()
        
        if not backend.is_ready():
            raise HTTPException(status_code=503, detail="Backend not initialized")
        
        models = backend.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": m["id"],
                    "object": "model",
                    "created": 0,
                    "owned_by": m.get("owned_by", "dLLM")
                }
                for m in models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))