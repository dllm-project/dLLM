"""
dLLM Embeddings Route
OpenAI-compatible /v1/embeddings endpoint
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Union
import time

from ..backend_connector import BackendConnector

router = APIRouter()


class EmbeddingRequest(BaseModel):
    input: str | List[str]
    model: str


class EmbeddingData(BaseModel):
    object: str
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]


@router.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """
    Create embeddings for input text
    
    This endpoint is compatible with OpenAI's /v1/embeddings API.
    It generates vector representations of text that can be used for
    semantic search, clustering, and other ML applications.
    """
    if not BackendConnector:
        raise HTTPException(status_code=503, detail="Backend not available")
    
    try:
        backend = BackendConnector()
        
        if not backend.is_ready():
            raise HTTPException(status_code=503, detail="Backend not initialized")
        
        # Handle single input or multiple inputs
        inputs = [request.input] if isinstance(request.input, str) else request.input
        
        embeddings_data = []
        total_tokens = 0
        
        for i, text in enumerate(inputs):
            embedding = backend.get_embedding(
                input_text=text,
                model=request.model
            )
            
            embeddings_data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })
            
            # Count tokens (rough estimate: 1 token ~= 4 characters)
            total_tokens += len(text) // 4
        
        return {
            "object": "list",
            "data": embeddings_data,
            "model": request.model,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/embeddings/models")
async def list_embedding_models():
    """
    List available embedding models
    
    Returns a list of models that can be used with the embeddings endpoint.
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