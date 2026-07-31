# Installation Guide

## System Requirements

### Minimum Requirements
- **CPU**: Intel Core i5 3rd gen (SSE4.2) or AMD equivalent
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Network**: Gigabit Ethernet (1 Gbps)
- **Python**: 3.8+

### Recommended Requirements (for AVX512)
- **CPU**: Intel Ice Lake/Ivy Bridge-EP or AMD Zen4+ with AVX-512
- **RAM**: 64 GB+
- **Storage**: NVMe SSD, 50 GB free space
- **Network**: 10 Gbps Ethernet (for distributed mode)

## Prerequisites

### Compiler Requirements
| Compiler | Version | Features |
|----------|---------|----------|
| GCC | 11+ | Full C++17 support |
| Clang | 14+ | Full C++17 support + AVX512 |
| MSVC | 19.30+ | Visual Studio 2022 |

### Python Dependencies

```bash
pip install fastapi uvicorn pydantic numpy pybind11 openai
```

### Rust Toolchain (for Tokenizer)

The dLLM Rust tokenizer requires:

- **Rust 1.70+**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- **Cargo**: Cargo package manager (included with Rust)
- **CMake 3.20+**: For FFI bindings

```bash
# Check Rust installation
rustc --version
cargo --version

# Update to latest stable
rustup update stable
```

### Build the Rust Tokenizer

```bash
cd dLLM/tokenizer

# Build with maximum SIMD acceleration (AVX-512)
cargo build --release --features avx512

# Or for compatibility with most modern CPUs:
cargo build --release --features avx2

# Output will be at: target/release/libdllm_tokenizer.so
```

### Hardware Detection Tools
- `cpuid` (Linux) or `wmic cpu get name` (Windows)
- Check for instruction set flags in `/proc/cpuinfo`

## Quick Start

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

# Start the API server
cd ../src/python
python server.py
```

## Build from Source

### Linux/macOS Build Steps

```bash
# Install C++ dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install -y \
    build-essential cmake git libssl-dev \
    libboost-all-dev libnuma-dev python3-dev

# Clone repository
git clone https://github.com/dllm-project/dLLM.git
cd dLLM

# Build C++ backend
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_SSE42=ON \
      -DUSE_AVX=OFF \
      -DUSE_AVX2=ON \
      -DUSE_AVX512=ON \
      ..
make -j$(nproc)

# Install Python package
pip install .
```

### Windows Build Steps

```cmd
:: Using Visual Studio Command Prompt
git clone https://github.com/dllm-project/dLLM.git
cd dLLM
mkdir build && cd build

cmake -G "Visual Studio 17 2022" ^
      -A x64 ^
      -DUSE_AVX512=ON ^
      ..

cmake --build . --config Release

:: Install Python dependencies
pip install fastapi uvicorn pydantic numpy
```

### Build the Rust Tokenizer for C++ Integration

```bash
cd dLLM/tokenizer

# Build shared library with FFI bindings
cargo build --release --lib --features avx512

# The shared library will be at:
# Linux: target/release/libdllm_tokenizer.so
# macOS: target/release/libdllm_tokenizer.dylib
# Windows: target\release\dllm_tokenizer.dll
```

### C++ Integration Example

```cpp
// tokenizer_bridge.cpp
#include <iostream>
#include "tokenizer/ffi/c_api.h"

extern "C" {
    void* dllm_tokenizer_init(const char* vocab);
    int dllm_tokenizer_encode(void* handle, const char* text, int** tokens);
    void dllm_tokenizer_free(void* handle);
}

int main() {
    // Initialize tokenizer
    void* tokenizer = dllm_tokenizer_init("vocab.txt");
    
    // Encode text
    int* tokens;
    int count = dllm_tokenizer_encode(tokenizer, "Hello world", &tokens);
    
    std::cout << "Encoded " << count << " tokens" << std::endl;
    for (int i = 0; i < count; i++) {
        std::cout << tokens[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    dllm_tokenizer_free(tokenizer);
    return 0;
}
```

Compile:
```bash
g++ -O3 tokenizer_bridge.cpp \
    -I./tokenizer/src/ffi \
    -L./tokenizer/target/release \
    -ldllm_tokenizer \
    -o tokenizer_example
```

## Configuration

### Basic Configuration (Single Node)

```yaml
# config.yaml
python_frontend:
  host: "0.0.0.0"
  port: 8000
  workers: 4

backend:
  cpp_library_path: "./build/libdllm.so"
  instruction_set: auto      # or sse42, avx, avx2, avx512
  
hardware:
  threads: 0                 # 0 = all cores

inference:
  max_context_length: 2048
  batch_size: auto
```

### Distributed Configuration

```yaml
# config.yaml
python_frontend:
  host: "0.0.0.0"
  port: 8000
  workers: 4

backend:
  cpp_library_path: "./build/libdllm.so"

distribution:
  mode: distributed
  nodes:
    - host: node1.example.com
      port: 5000
      rank: 0
    - host: node2.example.com  
      port: 5000
      rank: 1
    - host: node3.example.com
      port: 5000
      rank: 2
  
network:
  protocol: tcp
  bandwidth_target: 1GB/s
```

## Hardware Detection

### Check CPU Features (Linux)

```bash
# SSE4.2 support
grep sse4_2 /proc/cpuinfo

# AVX support
grep avx /proc/cpuinfo

# AVX2 support  
grep avx2 /proc/cpuinfo

# AVX-512 support
grep avx512f /proc/cpuinfo
```

### Check CPU Features (macOS)

```bash
# All instruction sets available
sysctl -a | grep machdep.cpu.features
sysctl -a | grep machdep.cpu.leaf7_features
```

## Docker Installation

### Build Docker Image

```dockerfile
FROM ubuntu:22.04

RUN apt update && apt install -y \
    build-essential cmake git libssl-dev \
    libboost-all-dev libnuma-dev python3-dev

WORKDIR /app
COPY . .

# Build C++ backend
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DUSE_SSE42=ON .. && \
    make -j$(nproc)

# Install Python dependencies
RUN pip install fastapi uvicorn pydantic numpy

CMD ["python", "src/python/server.py"]
```

```bash
docker build -t dllm:latest .
docker run --rm -p 8000:8000 dllm:latest
```

## Running the API Server

### Single Node Mode

```bash
cd src/python
python server.py --config ../config.yaml
```

### Distributed Mode

```bash
# Start on head node
python server.py \
    --distributed \
    --nodes node1,node2,node3

# Access at http://head-node:8000
```

## Verification

### Test Build

```bash
# Run unit tests
./build/bin/dllm-test --gtest_filter=Tensor.* 

# Check instruction set support
./build/bin/dllm-info
```

Expected output:
```
dLLM Hardware Info
==================
SSE4.2:     SUPPORTED
AVX:        SUPPORTED  
AVX2:       SUPPORTED
AVX-512:    SUPPORTED (512-bit, 32 registers)

Optimal instruction set: AVX-512
Recommended mode: distributed
```

### Test API Server

```bash
curl http://localhost:8000/v1/models

# Expected output:
# {"object":"list","data":[{"id":"llama-7b",...}]}
```

## Troubleshooting

### "Instruction not supported" Error
```bash
# Force fallback to lower instruction set
./dllm-infer --instruction-set sse42
```

### Distributed Mode Connection Issues
```bash
# Check node connectivity
./dllm-cluster-check --nodes host1,host2,host3

# View network diagnostics
./dllm-network-test --bandwidth 1GB/s
```

### OpenAI API Server Not Starting
```bash
# Ensure port 8000 is not in use
lsof -i :8000