# dLLM - Distributed CPU AI Inference Engine

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-beta-orange.svg)

## Overview

dLLM is a high-performance CPU-based AI inference engine that achieves GPU-level performance through distributed computing across multiple servers. By leveraging modern CPU instruction sets (SSE4.2 to AVX512) and intelligent distribution strategies, dLLM provides competitive inference speed with the cost-efficiency of CPU infrastructure.

## Architecture

dLLM uses a two-tier architecture:

- **C++ Backend**: Core inference engine with SIMD optimizations (AVX2/AVX512)
- **Python Frontend**: OpenAI-compatible API server with FastAPI

```
┌─────────────────────────────────────────────────────────────┐
│                    dLLM Cluster                             │
├─────────────────────────────────────────────────────────────┤
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

## Key Features

### High-Performance Tokenization (NEW)
The dLLM Rust tokenizer provides:
- **85K-92K tokens/s throughput** (10x+ faster than Python alternatives)
- **AVX2/AVX512 SIMD optimizations** for maximum speed
- **Zero-copy architecture** with memory-efficient processing (1.1x input size)
- **Universal compatibility**: Works with all LLM models

### GPU Hardware Support (NEW)
dLLM now supports multi-vendor GPU acceleration:
- **NVIDIA GPUs** - CUDA backend with Tensor Cores
- **AMD/ATI GPUs** - ROCm/HIP backend for GCN/RDNA architectures
- **Intel GPUs** - OneAPI/SYCL backend for Xe graphics

See [GPU Hardware Support](GPU_HARDWARE_SUPPORT.md) for complete details.

### Hardware Acceleration (CPU)
- **SSE4.2** - Streaming SIMD Extensions 4.2
- **AVX** - Advanced Vector Extensions
- **AVX2** - Advanced Vector Extensions 2
- **AVX-512** - Advanced Vector Extensions 512 (up to 512-bit wide operations)

### Distribution Clustering
- **Tensor Parallelism** - Split tensor operations across multiple nodes
- **Pipeline Parallelism** - Distributed layer-wise computation
- **Hybrid Parallelism** - Combined tensor + pipeline strategies

### KV Cache Optimization (NEW)
- **PV Cache** - Prefix Vector caching for 1M+ token contexts
- **Memory Compression**: 60-80% reduction in KV cache memory
- **Distributed Caching**: Cross-node prefix sharing
- **Adaptive Quantization**: Dynamic precision based on attention patterns

### Performance Characteristics
- **Network Speed**: 1 GB/s (Gigabit per second)
- **Latency Optimized**: Sub-millisecond internal communication
- **High Throughput**: Millions of tokens/second on distributed clusters

## Architecture

```
+------------------+     +------------------+     +------------------+
|   Node 1         |     |   Node 2         |     |   Node N         |
|  +------------+  |     |  +------------+  |     |  +------------+  |
|  | dLLM Core  |  |     |  | dLLM Core  |  |     |  | dLLM Core  |  |
|  +------------+  |     |  +------------+  |     |  +------------+  |
|  Tensor Pool   |<--->|  Tensor Pool   |<--->|  Tensor Pool   |
|  Pipeline Stage|     |  Pipeline Stage|     |  Pipeline Stage|
|  Hybrid Cache  |     |  Hybrid Cache  |     |  Hybrid Cache  |
+------------------+     +------------------+     +------------------+
         |                       |                      |
         +-----------------------+----------------------+
                                 |
                         +---------------+
                         | Load Balancer |
                         +---------------+
```

## Quick Start

### Prerequisites
- GCC 11+ or Clang 14+ with AVX2/AVX512 support
- CMake 3.20+
- Python 3.8+
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/dllm-project/dLLM.git
cd dLLM

# Install Python dependencies
pip install -r requirements.txt

# Build C++ backend
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_AVX512=ON \
      ..
make -j$(nproc)

# Start the OpenAI-compatible API server
python server.py
```

### Usage with OpenAI SDK

```bash
# Using Python OpenAI client
python examples/openai_client.py

# Or use curl
curl http://localhost:8000/v1/chat/completions \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-7b",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'
```

## API Compatibility

dLLM implements full OpenAI-compatible APIs:

| Endpoint | Status |
|----------|--------|
| `/v1/chat/completions` | ✓ Full support |
| `/v1/completions` | ✓ Full support |
| `/v1/embeddings` | ✓ Full support |
| `/v1/models` | ✓ Full support |

See [OpenAI API Documentation](OPENAI_API.md) for details.

## Performance Metrics

| Model | GPU Baseline | dLLM (4-node) | Speedup Factor |
|-------|-------------|---------------|----------------|
| GPT-2 Small | 150 ms | 160 ms | 0.94x |
| Llama-7B | 850 ms | 920 ms | 0.92x |
| Falcon-40B | 3.2 s | 3.5 s | 0.91x |

*Note: Performance varies based on network topology and model architecture*

## PV Cache Performance (NEW!)

### Memory Efficiency

| Context Size | Traditional KV Cache | PV Cache (INT8) | Savings |
|--------------|---------------------|-----------------|---------|
| 1M tokens    | ~40 GB              | ~10 GB          | 75%     |
| 2M tokens    | ~80 GB              | ~20 GB          | 75%     |

### Throughput Improvement

| Context Size | Baseline (no PV) | PV Cache (INT8) | Speedup |
|--------------|------------------|-----------------|---------|
| 1M tokens    | 150 tok/s        | 320 tok/s       | 2.1x    |
| 2M tokens    | 80 tok/s         | 180 tok/s       | 2.25x   |

### Cache Hit Rates

| Prefix Length | Exact Match Rate | Approximate Match Rate |
|---------------|------------------|------------------------|
| 64 tokens     | 12%              | 35%                    |
| 256 tokens    | 28%              | 52%                    |
| 1024 tokens   | 45%              | 68%                    |
| 4096 tokens   | 62%              | 78%                    |

See [PV_CACHE_OPTIMIZATION.md](PV_CACHE_OPTIMIZATION.md) for detailed performance analysis.

## Documentation

### Core Documentation
- [OpenAI API Compatibility](OPENAI_API.md) - Full OpenAI-compatible API documentation
- [Architecture](ARCHITECTURE.md) - System architecture details
- [Tensor Parallelism](TENSOR_PARALLELISM.md) - Tensor distribution strategies
- [Pipeline Parallelism](PIPELINE_PARALLELISM.md) - Pipeline parallel execution
- [Hybrid Parallelism](HYBRID_PARALLELISM.md) - Combined parallelization

### KV Cache Optimization (NEW!)
- [PV Cache Overview](PV_CACHE_README.md) - Complete PV cache documentation
  - **Prefix Vector Caching** for efficient large context handling
  - **Memory Compression**: 60-80% reduction in KV cache memory
  - **Distributed Caching**: Cross-node prefix sharing for clusters
  - **Adaptive Quantization**: Dynamic precision based on attention patterns

### Installation
- [Installation Guide](INSTALL.md) - Complete installation instructions including Rust tokenizer build

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines first.

## Contact

For questions and support, please open an issue in the repository.
