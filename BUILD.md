# Build Environment Setup Guide

This document describes how to set up the dLLM build environment on different platforms.

## Build Environment Overview

### Production Node (192.168.10.125)
| Component | Specification |
|-----------|--------------|
| Server IP | 192.168.10.125 |
| OS | Fedora Linux 44 (Server Edition) |
| CPU | Intel Core i9-14900KF |
| GPU | NVIDIA GeForce GTX 1060 6GB (Pascal, CC 6.1) |
| Instruction Set | SSE4.2 → AVX2 + CUDA |

### SSE4.2 Node (192.168.10.5)
| Component | Specification |
|-----------|--------------|
| Server IP | 192.168.10.5 |
| OS | Ubuntu 26.04 LTS |
| CPU | Intel Xeon X5570 |
| GPU | None |
| Instruction Set | SSE4.2 only |

## Building on Production Node (192.168.10.125)

### Prerequisites
- SSH access to the server
- sudo privileges

### Installation Steps

```bash
# Connect to production node
ssh saszel@192.168.10.125

# Install system dependencies (if not already installed)
sudo dnf install -y \
    build-essential cmake git libssl-dev \
    libboost-all-dev libnuma-dev python3-dev \
    gcc gcc-c++ make

# Verify installations
cmake --version  # Should be 3.20+
gcc --version    # Should be GCC 11+

# Build with CUDA support (GTX 1060 Pascal)
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_AVX2=ON \
      -DUSE_CUDA=ON \
      ..
make -j$(nproc)

# Install Python dependencies
pip install -r requirements.txt

# Test the build
ls -la lib/
```

## Building on SSE4.2 Node (192.168.10.5)

### Prerequisites
- SSH access to the server
- sudo privileges

### Installation Steps

```bash
# Connect to SSE4.2 node
ssh saszel@192.168.10.5

# Update package list and install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git libssl-dev \
    python3-pip python3-venv

# Verify installations
cmake --version  # Should be 3.20+
gcc --version    # Should be GCC 11+

# Build with SSE4.2 only (no AVX/AVX2)
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_SSE42=ON \
      -DUSE_AVX=OFF \
      -DUSE_AVX2=OFF \
      ..
make -j$(nproc)

# Install Python dependencies
pip install -r requirements.txt

# Test the build
ls -la lib/
```

## Building with Different Instruction Sets

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