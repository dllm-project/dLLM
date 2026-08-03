"""
dLLM Completions Route
OpenAI-compatible /v1/completions endpoint
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time

from ..backend_connector import BackendConnector

router = APIRouter()


class CompletionRequest(BaseModel):
    model: str
    prompt: str | List[str]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    Create text completion
    
    This endpoint is compatible with OpenAI's /v1/completions API.
    It generates text completions based on the provided prompt.
    """
    if not BackendConnector:
        raise HTTPException(status_code=503, detail="Backend not available")
    
    try:
        backend = BackendConnector()
        
        if not backend.is_ready():
            raise HTTPException(status_code=503, detail="Backend not initialized")
        
        # Handle single prompt or multiple prompts
        prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
        
        results = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for i, prompt in enumerate(prompts):
            result = backend.infer(
                model=request.model,
                prompt=prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
            )
            
            results.append({
                "index": i,
                "text": result,
                "finish_reason": "stop"
            })
            
            total_prompt_tokens += len(prompt.split())
            total_completion_tokens += len(result.split())
        
        return {
            "id": f"cmpl-{int(time.time() * 1000) % 100000}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": results,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/completions/models")
async def list_completion_models():
    """
    List available completion models
    
    Returns a list of models that can be used with the completions endpoint.
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