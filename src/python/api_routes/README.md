# API Routes Module

This module contains modular FastAPI route implementations for the dLLM backend service.

## Structure

```
api_routes/
├── __init__.py        # Package initialization and router imports
├── chat.py            # Chat completions endpoints
├── completions.py     # Text completion endpoints
├── embeddings.py      # Embedding generation endpoints
└── models.py          # Model management endpoints
```

## Usage

To use these routes in your FastAPI application:

```python
from fastapi import FastAPI
from api_routes.chat import router as chat_router
from api_routes.completions import router as completions_router

app = FastAPI()
app.include_router(chat_router, prefix="/api")
app.include_router(completions_router, prefix="/api")
```

## API Endpoints

### Chat Completions (`/api/chat/completions`)
- POST `/v1/chat/completions` - Create chat completion

### Completions (`/api/completions`)
- POST `/v1/completions` - Create text completion

### Embeddings (`/api/embeddings`)
- POST `/v1/embeddings` - Create embeddings for input text

### Models (`/api/models`)
- GET `/v1/models` - List available models
- GET `/v1/models/{model_id}` - Get model information

## Dependencies

- fastapi
- pydantic
- backend_connector (local)