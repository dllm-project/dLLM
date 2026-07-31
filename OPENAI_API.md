# OpenAI API Compatibility in dLLM

## Overview

dLLM provides full OpenAI API compatibility through a Python-based HTTP server that translates OpenAI-compatible requests to the C++ inference backend. All features and endpoints from the official OpenAI API are supported.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    dLLM Cluster                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  +------------------+    +------------------+              │
│  |   Python Front   │    |     C++ Backend  │              │
│  |   - FastAPI      ├────┤   - Tensor Core  │              │
│  |   - OpenAI Routes│    |   - Inference    │              │
│  |   - Auth/Rate    │    |   - Distribution │              │
│  +------------------+    +------------------+              │
│        │                          │                        │
│        ▼                          ▼                        │
│  OpenAI SDK compatible      Distributed Nodes              │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Chat Completions (`/v1/chat/completions`)

```json
POST /v1/chat/completions
{
    "model": "llama-7b",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "top_p": 1.0,
    "n": 1,
    "stream": false,
    "max_tokens": 256,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
}
```

**Response:**
```json
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "llama-7b",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 9,
        "total_tokens": 21
    }
}
```

### Completions (`/v1/completions`)

```json
POST /v1/completions
{
    "model": "gpt2-small",
    "prompt": "The quick brown fox",
    "temperature": 0.8,
    "max_tokens": 50,
    "stream": false
}
```

### Embeddings (`/v1/embeddings`)

```json
POST /v1/embeddings
{
    "model": "bert-base",
    "input": ["Hello world", "How are you?"]
}
```

**Response:**
```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [0.1, -0.2, 0.3, ...],
            "index": 0
        }
    ],
    "model": "bert-base",
    "usage": {
        "prompt_tokens": 5,
        "total_tokens": 5
    }
}
```

### Models (`/v1/models`)

```bash
GET /v1/models

Response:
{
    "object": "list",
    "data": [
        {
            "id": "llama-7b",
            "object": "model",
            "created": 1677858242,
            "owned_by": "dLLM"
        }
    ]
}
```

## Python Frontend Implementation

### FastAPI Server

```python
# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="dLLM OpenAI-compatible API")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 256

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # Translate to C++ backend
    result = cpp_backend.infer(request.model, request.messages)
    return format_openai_response(result)
```

### C++ Backend Bridge

```cpp
// python_bridge.cpp
#include <pybind11/pybind11.h>
#include "engine/inference_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dllm_cpp, m) {
    py::class_<InferenceEngine>(m, "InferenceEngine")
        .def(py::init<>())
        .def("load_model", &InferenceEngine::load_model)
        .def("infer", &InferenceEngine::infer)
        .def("chat", &InferenceEngine::chat);
}
```

## Feature Compatibility

| OpenAI Feature | Status |
|---------------|--------|
| Chat Completions API | ✓ Fully supported |
| Completions API | ✓ Fully supported |
| Embeddings API | ✓ Fully supported |
| Models listing | ✓ Fully supported |
| Streaming responses | ✓ Supported via SSE |
| Temperature control | ✓ Supported |
| Top-p sampling | ✓ Supported |
| Max tokens | ✓ Supported |
| Logprobs | ✓ Supported |
| Seed parameter | ✓ Supported |

## Configuration

```yaml
# openai_api.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

backend:
  cpp_library_path: "./build/libdllm.so"
  instruction_set: auto

model_paths:
  llama-7b: "/models/llama-7b"
  gpt2-small: "/models/gpt2-small"
```

## Usage with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not used, but required by SDK
)

response = client.chat.completions.create(
    model="llama-7b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)