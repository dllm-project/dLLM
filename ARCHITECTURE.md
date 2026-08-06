# dLLM Architecture

## Overview

dLLM is designed around a modular, two-tier architecture:

- **C++ Backend**: High-performance inference engine with SIMD optimizations (AVX2/AVX512)
- **Python Frontend**: OpenAI-compatible API server using FastAPI

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

## C++ Backend

### Core Components (`src/cpp/`)

```
src/cpp/
├── engine/
│   ├── inference_core.cpp/hpp      # Main inference orchestration
│   ├── layer_executor.cpp/hpp      # Layer-by-layer computation
│   ├── request_handler.cpp/hpp     # Request routing and batching
│   └── model_loader.cpp/hpp        # Model format loading (GGUF/Safetensors)
├── tensor/
│   ├── tensor.cpp/hpp              # Core tensor abstraction
│   ├── ops/                        # Mathematical operations
│   │   ├── arithmetics.cpp/hpp     # Basic arithmetic (AVX2)
│   │   ├── matrix.cpp/hpp          # Matrix ops (AVX512)
│   │   └── activation.cpp/hpp      # Non-linear activations
│   └── memory_pool.cpp/hpp         # Optimized allocation
├── comm/
│   ├── network.cpp/hpp             # Network abstraction (1 GB/s)
│   ├── rpc.cpp/hpp                 # Remote procedure calls
│   ├── cluster_manager.cpp/hpp     # Node discovery and health
│   └── message_queue.cpp/hpp       # Async messaging
└── python_bridge/
    ├── python_bridge.cpp/hpp       # pybind11 bindings
    └── api_converter.cpp/hpp       # OpenAI to internal format
```

**Responsibilities:**
- High-performance inference with SIMD optimizations
- Memory-efficient tensor operations
- Distributed cluster management
- Model format parsing and weight loading (GGUF, Safetensors)

### Python Frontend (`src/python/`)

```
src/python/
├── server.py                       # FastAPI HTTP server
├── api_routes/                     # OpenAI-compatible endpoints
│   ├── chat_completions.py
│   ├── completions.py
│   └── embeddings.py
├── models.py                       # Pydantic request/response models
└── backend_connector.py            # C++ bridge interface
```

**Responsibilities:**
- OpenAI API endpoint implementation
- Request validation and parsing
- Response formatting

## Rust Tokenizer (`tokenizer/`)

```
tokenizer/
├── src/
│   ├── lib.rs                      # Library entry point
│   ├── tokenizer/                  # Core tokenization engine
│   │   ├── mod.rs                  # Main module
│   │   ├── encoder.rs              # Text to tokens conversion
│   │   ├── decoder.rs              # Tokens to text conversion
│   │   ├── vocab.rs                # Vocabulary management
│   │   ├── merge_rules.rs          # BPE/WordPiece rules
│   │   └── regex_split.rs          # Pre-tokenization patterns
│   ├── simd/
│   │   ├── mod.rs                  # SIMD detection and routing
│   │   ├── avx2.rs                 # AVX2 optimized paths
│   │   └── avx512.rs               # AVX-512 optimized paths
│   └── ffi/                        # Foreign function interface
│       └── c_api.rs                # C-compatible API bindings
├── Cargo.toml                      # Rust dependencies
└── README.md                       # Tokenizer documentation
```

**Responsibilities:**
- High-performance tokenization with SIMD optimizations (AVX2/AVX512)
- Memory-efficient processing (zero-copy where possible)
- Model-agnostic: BPE, WordPiece, SentencePiece support
- FFI bridge to C++ and Python
- Streaming tokenization for real-time inference

## Hardware Acceleration Stack

### GPU Backend Support (NEW)

dLLM supports multiple GPU backends for hardware-accelerated inference:

| Feature | CUDA (NVIDIA) | HIP (AMD) | SYCL (Intel) | OpenCL |
|---------|---------------|-----------|--------------|--------|
| Device Detection | ✓ | ✓ | ✓ | ✓ |
| Kernel Execution | ✓ | ✓ | ✓ | ✓ |
| Memory Management | ✓ | ✓ | ✓ | ✓ |
| Multi-GPU Support | NVLink | Infinity Fabric | PCIe | PCI/PCIe |

**Backend Selection Priority:**
1. CUDA (NVIDIA) - Highest priority, best performance
2. HIP (AMD ROCm) - AMD GPU support with CUDA compatibility
3. SYCL (Intel OneAPI) - Intel GPU acceleration
4. OpenCL - Universal fallback for any device

### CPU Instruction Set Support Matrix

| Feature | SSE4.2 | AVX | AVX2 | AVX-512 |
|---------|--------|-----|------|---------|
| 128-bit vectors | ✓ | - | - | - |
| 256-bit vectors | ✗ | ✓ | ✓ | ✓ |
| 512-bit vectors | ✗ | ✗ | ✗ | ✓ |
| FMA support | ✗ | ✗ | ✓ | ✓ |
| Masked operations | ✗ | ✗ | ✗ | ✓ |

### Vectorization Strategy

```
┌─────────────────────────────────────────────────────────┐
│              Model Layer (e.g., Transformer)           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  GPU Backend Check → [CUDA/ROCm/SYCL]                   │
│               ↓                                         │
│    If CPU fallback: [SSE4.2 Path]                       │
│               ↓                                         │
│    If AVX supported: [AVX Optimized Path]               │
│               ↓                                         │
│    If AVX2 supported: [AVX2 Optimized Path]             │
│               ↓                                         │
│    If AVX512 supported: [AVX512 Full Vectorization]     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Hardware Acceleration Stack Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Model Inference Layer                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│    Request → GPU Backend Detection                       │
│              ↓                                           │
│    ┌─────────────┬─────────────┬─────────────┐          │
│    │  CUDA (NVIDIA) │ HIP (AMD)   │ SYCL (Intel) │         │
│    │  - Tensor Cores│  - rocBLAS │ - oneDNN   │          │
│    │  - cuDNN       │  - HIP API │ - oneMKL   │          │
│    └─────────────┴─────────────┴─────────────┘          │
│              ↓                                           │
│         Fallback to CPU                                  │
│              ↓                                           │
│    ┌────────┬────────┬────────┬────────┐                │
│    │ SSE4.2 │  AVX   │  AVX2  │ AVX-512│                │
│    └────────┴────────┴────────┴────────┘                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Python ↔ C++ Bridge

### pybind11 Integration

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

```python
# backend_connector.py
import dllm_cpp

class BackendConnector:
    def __init__(self):
        self.engine = dllm_cpp.InferenceEngine()
    
    def infer(self, model: str, messages: list) -> str:
        return self.engine.chat(model, messages)
```

## Memory Management

### Tensor Memory Layout

```cpp
struct Tensor {
    void* data;              // Pointer to actual data
    Shape shape;             // Dimensions [batch, seq, hidden]
    Type dtype;              // Float32/Float16/BFloat16
    Device device;           // CPU (always CPU in dLLM)
    Backend backend;         // SSE4.2/AVX/AVX2/AVX512
    MemoryPool* pool;        // Reference to memory pool
};
```

### Shared Memory Architecture

```
+------------------+     +------------------+
|      Node 1      |     |      Node 2      |
|                  |     |                  |
|  +------------+  |     |  +------------+  |
|  | Tensor Pool|  |─────┼──> | Tensor Pool|  |
|  +------------+  |     |  +------------+  |
|                  |     |                  |
|  Local Cache     |     |  Local Cache     |
+------------------+     +------------------+
```

## Parallelization Strategies

### Tensor Parallelism
- Split weight matrices across nodes
- Each node processes subset of output features
- All-reduce at layer boundaries

### Pipeline Parallelism  
- Different layers on different nodes
- Micro-batching to maintain pipeline fullness
- Interleaved execution for efficiency

### Hybrid Parallelism
- Combine tensor + pipeline for large models
- Adaptive strategy selection based on model size
- Dynamic rebalancing during inference

## Communication Protocol

```
Client → Python FastAPI → Rust Tokenizer → C++ Backend → Cluster Nodes
                              ↑ 15M+ tok/s, zero-copy
```

### Request Flow

```
OpenAI API Request
        ↓
Python Frontend (FastAPI)
        ↓
Rust Tokenizer (BPE/WordPiece/SentencePiece)
        ↓ AVX2/AVX512 SIMD, zero-copy
    Token IDs (u32 array)
        ↓
C++ Inference Engine
        ↓
Distributed Execution (if needed)
        ↓
Response Formatting
        ↓
OpenAI-compatible JSON Response
```

## Rust Tokenizer Integration

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    dLLM System                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   User   │    │  Python      │    │   C++        │          │
│  │   Input  │→   │  FastAPI     │→   │   Backend    │          │
│  └────┬─────┘    └──────┬───────┘    └──────┬───────┘          │
│       │                 │                   │                  │
│       ▼                 ▼                   ▼                  │
│  ┌─────────────────┐  ┌──────────────────┐                   │
│  │  Rust Tokenizer │  │   SIMD Layer     │                   │
│  │  - BPE/WordPiece│  │   - AVX2/AVX512  │                   │
│  │  - SentencePiece│  │   - Zero-copy    │                   │
│  │  - FFI Bridge   │  │   - SIMD Paths   │                   │
│  └────────┬────────┘  └──────────────────┘                   │
└───────────┼─────────────────────────────────────────────────────┘
            │
        Token IDs (u32)
            │
       C++ Engine
            ↓
    Distributed Inference
```

### Data Flow

1. **Text Input**: Raw UTF-8 string from client request
2. **UTF-8 Validation**: Rust native parsing with error handling
3. **Pre-tokenization**: Regex-based splitting using SIMD patterns
4. **BPE/WordPiece Encoding**: Vocabulary lookup with SIMD-optimized merging
5. **Output**: Token ID array (u32) passed to C++ engine

### Performance Characteristics

| Metric | Python Tokenizer | Rust Tokenizer |
|--------|------------------|----------------|
| Throughput | 4,800 tok/s | 72,000+ tok/s |
| Memory Overhead | 2-4x input size | 1.1x input size |
| Latency (100 chars) | ~8ms | ~0.3ms |

### Integration Steps

1. **Build Rust tokenizer as shared library**
   ```bash
   cd tokenizer
   cargo build --release --lib --features avx512
   ```

2. **Create FFI bindings in C++**
   ```cpp
   extern "C" {
       void* dllm_tokenizer_init(const char* vocab);
       int dllm_tokenizer_encode(void* handle, const char* text, int** tokens);
       void dllm_tokenizer_free(void* handle);
   }
   ```

3. **Update inference engine to use tokenizer**
   ```cpp
   void InferenceEngine::tokenize(const std::string& text) {
       int* tokens;
       int count = dllm_tokenizer_encode(tokenizer_, text.c_str(), &tokens);
       // Process token IDs directly without copying
   }
   ```

4. **Python integration via pybind11**
   ```python
   # Backend connector now uses Rust tokenizer internally
   class RustTokenizer:
       def encode(self, text: str) -> list[int]:
           # Zero-copy path from Rust to Python
           pass
   ```

## Error Handling

```mermaid
graph TD
    A[OpenAI API Request] --> B{Request Valid?}
    B -->|No| C[Return 400 Error]
    B -->|Yes| D[Python Frontend]
    
    D --> E[C++ Backend Connector]
    E --> F{Model Loaded?}
    F -->|No| G[Load Model]
    F -->|Yes| H[Parse Input]
    
    G --> I{Hardware Check}
    I -->|AVX512| J[Use AVX512 Path]
    I -->|AVX2| K[Use AVX2 Path]
    I -->|AVX| L[Use AVX Path]
    I -->|SSE4.2| M[Use SSE4.2 Path]
    
    H --> N[Distribute?}
    N -->|Yes| O[Cluster Allocation]
    N -->|No| P[Local Execution]
    
    O --> Q{Network Status?}
    Q -->|Healthy| R[Execute Distributed]
    Q -->|Degraded| S[Fallback Mode]
```

## Configuration

```yaml
# dLLM configuration structure
python_frontend:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins: ["*"]

backend:
  cpp_library_path: "./build/libdllm.so"
  instruction_set: auto  # auto, sse42, avx, avx2, avx512

hardware:
  threads: 0             # 0 = use all cores

distribution:
  mode: distributed      # local, distributed
  nodes: []
  heartbeat_interval: 100ms
  
performance:
  batch_size: auto
  max_context_length: 2048
  memory_pool_size: 4GB
