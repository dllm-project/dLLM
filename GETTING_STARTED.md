# Getting Started with dLLM

## Quick Start Guide

dLLM uses a two-tier architecture:
- **Python Frontend**: OpenAI-compatible API server (FastAPI) (`✅ Implemented`)
- **C++ Backend**: High-performance inference engine with SIMD optimizations (SSE4.2/AVX/AVX2) (`✅ Implemented`)

### Option 1: CPU Mode (Default)

```bash
# Clone the repository
git clone https://github.com/dllm-project/dLLM.git
cd dLLM

# Install Python dependencies
pip install -r requirements.txt

# Build C++ backend
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_SSE42=ON \
      -DUSE_AVX=ON \
      -DUSE_AVX2=ON \
      ..
make -j$(nproc)

# Start the API server
cd ../src/python
python server.py
```

### Option 2: GPU Mode (🔲 Planned)

> **Note**: GPU acceleration is declared as CMake options but no GPU backend source code exists yet.

```bash
# Install GPU drivers and toolkit
# NVIDIA: CUDA 11.8+, AMD: ROCm 5.3+, Intel: OneAPI 2023.1+

# Build with GPU support (planned)
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_CUDA=ON \      # or USE_HIP=ON, USE_SYCL=ON
      ..
make -j$(nproc)

# Start the API server with GPU acceleration (planned)
cd ../src/python
python server.py --use-gpu
```

**Note:** See [GPU Hardware Support](GPU_HARDWARE_SUPPORT.md) for complete installation instructions by vendor.

The server will start on `http://0.0.0.0:8000` with:
- Auto-generated Swagger UI at `/docs`
- Full OpenAI API compatibility

### Step 1 (CPU): Test Your Installation

```bash
# Test models endpoint (note: /api prefix, not /v1)
curl http://localhost:8000/api/models

# Test chat completions
curl -X POST http://localhost:8000/api/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-7b",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

### Step 2: Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="llama-7b",
    messages=[
        {"role": "user", "content": "What is AI?"}
    ]
)
print(response.choices[0].message.content)
```

## Common Use Cases

### Text Generation (LLM)

**Using curl:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-7b",
        "messages": [
            {"role": "user", "content": "Write a poem about AI"}
        ],
        "temperature": 0.7,
        "max_tokens": 256
    }'
```

**Using Python:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Write a poem"}],
    temperature=0.7
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

response = client.embeddings.create(
    model="bert-base",
    input=["Hello world", "How are you?"]
)

for embedding in response.data:
    print(f"Embedding dim: {len(embedding.embedding)}")
```

## Configuration

### CPU Mode Configuration
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

backend:
  cpp_library_path: "./build/libdllm.so"
  instruction_set: auto   # sse42, avx, avx2, avx512

models:
  llama-7b: "/models/llama-7b"

distribution:
  mode: local
```

### GPU Mode Configuration (NVIDIA)
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

backend:
  cpp_library_path: "./build/libdllm.so"
  gpu_backend: cuda       # cuda, hip, sycl
  device_id: 0

gpu:
  enabled: true
  memory_pool_size: 8GB

models:
  llama-7b-gpu: "/models/llama-7b"

distribution:
  mode: local
```

### GPU Mode Configuration (AMD/ATI)
```yaml
backend:
  gpu_backend: hip        # ROCm/HIP backend

gpu:
  enabled: true
  device_id: all          # Use all available GPUs
```

### GPU Mode Configuration (Intel)
```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

backend:
  cpp_library_path: "./build/libdllm.so"
  gpu_backend: sycl       # OneAPI/SYCL backend
  device_id: 0

gpu:
  enabled: true
  memory_pool_size: 4GB

models:
  llama-7b-gpu-intel: "/models/llama-7b"

distribution:
  mode: local
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DLLM_HOST | 0.0.0.0 | Server host address |
| DLLM_PORT | 8000 | Server port |
| DLLM_CPP_PATH | ./build/libdllm.so | C++ library path |

## Distributed Setup

### Cluster Configuration

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8000

backend:
  cpp_library_path: "./build/libdllm.so"

distribution:
  mode: distributed
  nodes:
    - host: node1.example.com
      rank: 0
    - host: node2.example.com
      rank: 1
    - host: node3.example.com
      rank: 2

parallelism:
  tensor_degree: 4
  pipeline_stages: 2
```

### Starting the Cluster

```bash
# On head node (node1)
python server.py --config config.yaml

# Access from any client
curl http://node1:8000/v1/chat/completions
```

## Performance Optimization Tips

### CPU Optimization
```yaml
backend:
  instruction_set: avx512  # Use highest available
  threads: 0               # 0 = all cores
```

### Memory Optimization
```yaml
backend:
  memory_pool_size: 32GB
  quantization: int8       # Reduced precision
```

## Monitoring

### Server Metrics

```bash
# Check server health
curl http://localhost:8000/health

# View API docs (Swagger UI)
open http://localhost:8000/docs
```

### Debug Mode

```bash
# Start with verbose logging
python server.py --log-level debug
```
