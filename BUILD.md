# Build Environment Setup Guide

This document describes how to set up the dLLM build environment on different platforms.

## Build Environment Overview

### Target Build Server
| Component | Specification |
|-----------|--------------|
| Server IP | 192.168.10.7 |
| User | saszel |
| OS | Linux (Ubuntu/Fedora) |
| CPU | x86_64 with SSE4.2/AVX2 support |
| GPU | None (CPU-only builds) |
| Instruction Set | SSE4.2 → AVX2 |

> **Note**: GPU acceleration (CUDA/ROCm/SYCL) is declared as CMake options but not yet implemented.

## Building on Target Server (192.168.10.7)

### Prerequisites
- SSH access to the server (`ssh saszel@192.168.10.7`)
- sudo privileges

### Installation Steps

```bash
# Connect to build server
ssh saszel@192.168.10.7

# Install system dependencies (if not already installed)
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git libssl-dev \
    python3-dev python3-pip \
    gcc g++ make

# Verify installations
cmake --version  # Should be 3.20+
gcc --version    # Should be GCC 11+

# Build with AVX2 support
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_SSE42=ON \
      -DUSE_AVX=ON \
      -DUSE_AVX2=ON \
      ..
make -j$(nproc)

# Install Python dependencies
pip install -r requirements.txt

# Test the build
ls -la lib/
```

## Building with Different Instruction Sets

### SSE4.2 Only (Baseline)
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_SSE42=ON \
      -DUSE_AVX=OFF \
      -DUSE_AVX2=OFF \
      ..
```

### AVX + AVX2 (Recommended)
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_SSE42=ON \
      -DUSE_AVX=ON \
      -DUSE_AVX2=ON \
      ..
```

### With CUDA Support (Planned)
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_SSE42=ON \
      -DUSE_AVX=ON \
      -DUSE_AVX2=ON \
      -DUSE_CUDA=ON \
      ..
```

> **Note**: CUDA support is declared as a CMake option but no GPU backend source code exists yet.

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug/Release/RelWithDebInfo) |
| `USE_SSE42` | ON | Enable SSE4.2 SIMD operations |
| `USE_AVX` | ON | Enable AVX SIMD operations |
| `USE_AVX2` | ON | Enable AVX2 SIMD operations |
| `USE_CUDA` | OFF | Enable NVIDIA CUDA GPU acceleration (planned) |
| `USE_HIP` | OFF | Enable AMD ROCm GPU acceleration (planned) |
| `USE_SYCL` | OFF | Enable Intel OneAPI GPU acceleration (planned) |

## Build Output

After a successful build, the following artifacts are produced:

| Artifact | Location | Description |
|----------|----------|-------------|
| `libdllm_tensor.a` | `build/src/cpp/tensor/` | Tensor library (static) |
| `libdllm_engine.a` | `build/src/cpp/engine/` | Inference engine library (static) |
| `libdllm_comm.a` | `build/src/cpp/comm/` | Communication library (static) |
| `dllm_cpp.so` | `build/src/python/` | Python extension module |
| `setup.py` | `build/src/python/` | Generated Python setup script |

### CMake Configuration Options

| Option | Description | Recommended Use |
|--------|-------------|-----------------|
| `USE_SSE42` | Enable SSE4.2 support (baseline) | All systems, default ON |
| `USE_AVX` | Enable AVX support | CPUs with AVX instruction set |
| `USE_AVX2` | Enable AVX2 support | CPUs with AVX2/FMA instructions |
| `USE_CUDA` | Enable NVIDIA CUDA acceleration | Systems with NVIDIA GPU |

### Example CMake Commands

```bash
# SSE4.2 only (maximum compatibility)
cmake -DUSE_SSE42=ON -DUSE_AVX=OFF -DUSE_AVX2=OFF ..

# AVX enabled (most modern CPUs)
cmake -DUSE_SSE42=ON -DUSE_AVX=ON -DUSE_AVX2=OFF ..

# AVX2 with FMA (best for modern Intel/AMD CPUs)
cmake -DUSE_SSE42=ON -DUSE_AVX=ON -DUSE_AVX2=ON ..

# With CUDA support (NVIDIA GPUs)
cmake -DUSE_SSE42=ON -DUSE_AVX2=ON -DUSE_CUDA=ON ..
```

## Docker Build

### Build Docker Image

```bash
docker build -t dllm:latest \
    --build-arg USE_CUDA=true \
    .
```

### Run with CUDA Support

```bash
docker run --gpus all -p 8000:8000 dllm:latest
```

## CI/CD Pipeline

The project uses GitHub Actions for CI/CD. The pipeline:

1. Builds on Ubuntu with SSE4.2 and AVX2 configurations
2. Builds on Fedora with CUDA support
3. Tests the compiled binaries
4. Deploys to production node (192.168.10.125) on main branch push

### Triggering a Build

```bash
# Push to main or develop branches
git push origin main
```

## Verification

After building, verify the installation:

```bash
# Check library files exist
ls -la build/lib/

# Run tests (if implemented)
cd build && ctest --output-on-failure

# Start the API server
cd src/python && python server.py