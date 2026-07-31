# dLLM Python Integration Guide

## Overview

This guide explains how to integrate dLLM with your Python applications using the OpenAI-compatible API.

## Quick Start

### Installation

```bash
pip install openai fastapi uvicorn pydantic numpy
```

### Starting the Server

```bash
cd src/python
python server.py --config ../config.yaml
```

The server will start on `http://0.0.0.0:8000` with:
- Full OpenAI API compatibility
- Auto-generated documentation at `/docs`
- Streaming support

## Python Client Examples

### Basic Chat Completion

```python
from openai import OpenAI

# Initialize client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# Create a chat completion
response = client.chat.completions.create(
    model="llama-7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AI?"}
    ]
)
print(response.choices[0].message.content)
```

### Streaming Responses

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

stream = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Embeddings

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Create embeddings
response = client.embeddings.create(
    model="bert-base",
    input=["Hello world", "How are you?"]
)

for embedding in response.data:
    print(f"Embedding dimension: {len(embedding.embedding)}")
```

### Text Generation

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

response = client.completions.create(
    model="gpt2-small",
    prompt="The quick brown fox",
    max_tokens=50,
    temperature=0.7
)
print(response.choices[0].text)
```

## Advanced Usage

### Batch Processing

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

responses = client.chat.completions.create(
    model="llama-7b",
    messages=[
        {"role": "user", "content": "Question 1"},
        {"role": "user", "content": "Question 2"}
    ],
    n=2
)
```

### Custom Parameters

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
    presence_penalty=0.1,
    frequency_penalty=0.1
)
```

### Error Handling

```python
from openai import OpenAI, APIError, RateLimitError

client = OpenAI(base_url="http://localhost:8000/v1")

try:
    response = client.chat.completions.create(
        model="llama-7b",
        messages=[{"role": "user", "content": "Hello"}]
    )
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except APIError as e:
    print(f"API error: {e}")
```

### Async Usage

```python
import asyncio
from openai import AsyncOpenAI

async def main():
    client = AsyncOpenAI(base_url="http://localhost:8000/v1")
    
    response = await client.chat.completions.create(
        model="llama-7b",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

## Server Configuration

### config.yaml

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins: ["*"]

backend:
  cpp_library_path: "./build/libdllm.so"
  instruction_set: auto

models:
  llama-7b: "/models/llama-7b"
  gpt2-small: "/models/gpt2-small"

distribution:
  mode: local
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DLLM_HOST | 0.0.0.0 | Server host address |
| DLLM_PORT | 8000 | Server port |
| DLLM_CPP_PATH | ./build/libdllm.so | C++ library path |

## pybind11 Integration

### Creating a Custom Python Module

```cpp
// custom_bridge.cpp
#include <pybind11/pybind11.h>
#include "engine/inference_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(dllm_custom, m) {
    py::class_<InferenceEngine>(m, "InferenceEngine")
        .def(py::init<>())
        .def("load_model", &InferenceEngine::load_model)
        .def("infer", &InferenceEngine::infer);
}

// Compile: g++ -O3 -shared -std=c++17 custom_bridge.cpp \
//     -I/usr/include/python3.8 `python3-config --cflags` \
//     -o dllm_custom.so
```

### Using the Custom Module

```python
import dllm_custom

engine = dllm_custom.InferenceEngine()
engine.load_model("/models/llama-7b")
result = engine.infer("Hello!")
print(result)