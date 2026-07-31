# Getting Started with dLLM

## Quick Start Guide

dLLM uses a two-tier architecture:
- **Python Frontend**: OpenAI-compatible API server (FastAPI)
- **C++ Backend**: High-performance inference engine with SIMD optimizations

### Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/dark0venom/dLLM.git
cd dLLM

# Install Python dependencies
pip install -r requirements.txt

# Build C++ backend
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_AVX512=ON \
      ..
make -j$(nproc)

# Start the API server
cd ../src/python
python server.py
```

The server will start on `http://0.0.0.0:8000` with:
- Auto-generated Swagger UI at `/docs`
- Full OpenAI API compatibility

### Step 2: Test Your Installation

```bash
# Test models endpoint
curl http://localhost:8000/v1/models

# Test chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-7b",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

### Step 3: Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
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

### Server Configuration (config.yaml)

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

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
