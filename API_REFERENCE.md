# dLLM API Reference

## Overview

dLLM provides OpenAI-compatible APIs through a Python frontend that interfaces with the C++ backend. The architecture consists of:

- **Python Frontend**: FastAPI server with OpenAI-compatible endpoints (`✅ Implemented`)
- **C++ Backend**: High-performance inference engine with SIMD optimizations (SSE4.2/AVX/AVX2) (`✅ Implemented`)

## OpenAI-Compatible REST API

All endpoints are compatible with OpenAI's API specification.

### Base URL

```
http://localhost:8000/api
```

> **Note**: The API uses `/api` prefix (not `/v1`). This is configured in `src/python/server.py` with `prefix="/api"`.

### POST /api/chat/completions

Chat completions endpoint with full OpenAI compatibility.

**Request Headers:**
```bash
curl -X POST http://localhost:8000/api/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer dummy-key" \
    -d '{
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
    }'
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

### POST /api/completions

Text completions endpoint.

**Request:**
```bash
curl -X POST http://localhost:8000/api/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt2-small",
        "prompt": "The quick brown fox",
        "temperature": 0.8,
        "max_tokens": 50,
        "stream": false
    }'
```

### POST /api/embeddings

Generate embeddings for input text.

**Request:**
```bash
curl -X POST http://localhost:8000/api/embeddings \
    -H "Content-Type: application/json" \
    -d '{
        "model": "bert-base",
        "input": ["Hello world", "How are you?"]
    }'
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

### GET /models

List available models.

```bash
curl http://localhost:8000/v1/models
```

**Response:**
```json
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

## Python API

### Basic Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not validated, but required by SDK
)

response = client.chat.completions.create(
    model="llama-7b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

### Streaming

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

stream = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Embeddings

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.embeddings.create(
    model="bert-base",
    input=["Hello world", "How are you?"]
)

for embedding in response.data:
    print(f"Embedding: {embedding.embedding[:5]}...")
```

## Rust Tokenizer API

### Overview

The dLLM Rust tokenizer provides high-performance tokenization with:
- **15M+ tokens/s throughput** (10x+ faster than Python alternatives)
- **AVX2/AVX512 SIMD optimizations**
- **Zero-copy architecture** for memory efficiency
- **Universal model compatibility**: Llama, Mistral, Phi, Qwen, etc.

### C API

```c
// dllm_tokenizer.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/// Initialize tokenizer with vocabulary file
void* tokenizer_init(const char* vocab_path);

/// Encode text to tokens (returns token count)
int tokenizer_encode(
    void* handle,
    const char* text,
    int** tokens,
    int** offsets  // Optional: character offsets
);

/// Decode tokens back to text
const char* tokenizer_decode(void* handle, const int* tokens, int count);

/// Free tokenizer resources
void tokenizer_free(void* handle);

#ifdef __cplusplus
}
#endif
```

### C++ Integration

```cpp
// tokenizer_bridge.cpp
#include <iostream>
#include "tokenizer.h"

extern "C" {
    void* dllm_tokenizer_init(const char* vocab) {
        return new dllm::Tokenizer(vocab);
    }
    
    int dllm_tokenizer_encode(void* handle, const char* text, int** tokens) {
        auto tokenizer = static_cast<dllm::Tokenizer*>(handle);
        auto encoding = tokenizer->encode(text);
        *tokens = encoding.tokens().data();
        return encoding.tokens().size();
    }
    
    void dllm_tokenizer_free(void* handle) {
        delete static_cast<dllm::Tokenizer*>(handle);
    }
}

// Using in inference engine
class InferenceEngine {
    void* tokenizer_;
    
public:
    InferenceEngine() : tokenizer_(nullptr) {}
    
    ~InferenceEngine() {
        if (tokenizer_) dllm_tokenizer_free(tokenizer_);
    }
    
    void load_model(const std::string& path) {
        tokenizer_ = dllm_tokenizer_init((path + "/vocab.txt").c_str());
    }
    
    std::vector<int> tokenize(const std::string& text) {
        int* tokens;
        int count = dllm_tokenizer_encode(tokenizer_, text.c_str(), &tokens);
        return std::vector<int>(tokens, tokens + count);
    }
};
```

### pybind11 Bridge

```cpp
// python_bridge.cpp
#include <pybind11/pybind11.h>
#include "engine/inference_engine.hpp"
#include "tokenizer/ffi/c_api.h"

namespace py = pybind11;

PYBIND11_MODULE(dllm_cpp, m) {
    // Inference engine bindings
    py::class_<InferenceEngine>(m, "InferenceEngine")
        .def(py::init<>())
        .def("load_model", &InferenceEngine::load_model)
        .def("infer", &InferenceEngine::infer)
        .def("chat", &InferenceEngine::chat);
    
    // Tokenizer bindings (optional, for direct access)
    py::class_<TokenEncoder>(m, "Tokenizer")
        .def(py::init<const std::string&>())
        .def("encode", &TokenEncoder::encode)
        .def("decode", &TokenEncoder::decode)
        .def("encode_batch", &TokenEncoder::encode_batch);
}
```

```python
# backend_connector.py
import dllm_cpp

class BackendConnector:
    def __init__(self):
        self.engine = dllm_cpp.InferenceEngine()
    
    def infer(self, model: str, messages: list) -> str:
        return self.engine.chat(model, messages)
```

### Inference Engine

```cpp
// inference_engine.hpp
class InferenceEngine {
public:
    // Load a model from disk
    void load_model(const std::string& model_path);
    
    // Chat completion (OpenAI format)
    std::string chat(const std::vector<Message>& messages);
    
    // Text generation
    std::string generate(
        const std::string& prompt,
        int max_tokens = 256,
        float temperature = 0.7f);
    
    // Get model info
    ModelInfo get_model_info() const;
};
```

### Tensor Operations

```cpp
// tensor.hpp
class Tensor {
public:
    // Create tensor with specified shape and dtype
    Tensor(const Shape& shape, DataType dtype = FP32);
    
    // Data access
    void* data();
    const void* data() const;
    
    // Shape information
    Shape shape() const;
    int dim(int i) const;
    
    // Operations (vectorized)
    Tensor matmul(const Tensor& other);
    Tensor add(const Tensor& other);
    Tensor relu();
    Tensor softmax();
};
```

## Configuration

### Server Configuration

```yaml
# config.yaml
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
  mode: local  # or distributed
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DLLM_HOST | 0.0.0.0 | Server host address |
| DLLM_PORT | 8000 | Server port |
| DLLM_WORKERS | 4 | Number of worker processes |
| DLLM_CPP_PATH | ./build/libdllm.so | C++ library path |

## CLI Interface

### Starting the API Server

```bash
# Start server with default config
python src/python/server.py

# With custom config
python src/python/server.py --config config.yaml

# Enable distributed mode
python src/python/server.py --distributed --nodes node1,node2,node3
```

### Verification

```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Test chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"llama-7b","messages":[{"role":"user","content":"hi"}]}'
```

## Feature Compatibility

| OpenAI Feature | Status |
|---------------|--------|
| Chat Completions API | ✓ Full support |
| Completions API | ✓ Full support |
| Embeddings API | ✓ Full support |
| Models listing | ✓ Full support |
| Streaming responses | ✓ Supported via SSE |
| Temperature control | ✓ Supported |
| Top-p sampling | ✓ Supported |
| Max tokens | ✓ Supported |
| Logprobs | ✓ Supported |
| Seed parameter | ✓ Supported |
