"""
dLLM Models Route
OpenAI-compatible /v1/models endpoint
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import time

from ..backend_connector import BackendConnector

router = APIRouter()


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsListResponse(BaseModel):
    object: str = "list"
    data: List[ModelObject]


@router.get("/v1/models")
async def list_models():
    """
    List available models
    
    This endpoint is compatible with OpenAI's /v1/models API.
    It returns a list of all models that can be used with other endpoints.
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


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """
    Get information about a specific model
    
    This endpoint is compatible with OpenAI's /v1/models/{model_id} API.
    """
    if not BackendConnector:
        raise HTTPException(status_code=503, detail="Backend not available")
    
    try:
        backend = BackendConnector()
        
        if not backend.is_ready():
            raise HTTPException(status_code=503, detail="Backend not initialized")
        
        models = backend.list_models()
        
        # Find the requested model
        model_info = None
        for m in models:
            if m["id"] == model_id:
                model_info = m
                break
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        return {
            "id": model_info["id"],
            "object": "model",
            "created": 0,
            "owned_by": model_info.get("owned_by", "dLLM"),
            **{k: v for k, v in model_info.items() if k not in ["id", "object", "created", "owned_by"]}
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))