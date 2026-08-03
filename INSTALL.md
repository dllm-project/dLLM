# Installation Guide

## System Requirements

### Minimum Requirements (CPU)
- **CPU**: Intel Core i5 3rd gen (SSE4.2) or AMD equivalent
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Network**: Gigabit Ethernet (1 Gbps)
- **Python**: 3.8+

### GPU Minimum Requirements

#### NVIDIA GPUs
- **CUDA Version**: 11.4+
- **Compute Capability**: 6.0+ (Pascal and newer)
- **GPU Drivers**: NVIDIA Driver 470.82.01+
- **RAM**: 4 GB VRAM minimum, 8 GB recommended

#### AMD/ATI GPUs
- **ROCm Version**: 5.3+
- **Hardware**: GCN 1st gen or newer (RX 400 series+)
- **RAM**: 4 GB VRAM minimum, 8 GB recommended

#### Intel GPUs
- **OneAPI Version**: 2023.1+
- **Hardware**: Gen9+ graphics (Iris Xe or Arc A-Series)
- **RAM**: 4 GB VRAM minimum, 8 GB recommended

### Recommended Requirements (for AVX2 CPU)
- **CPU**: Intel Haswell/Broadwell or AMD Zen+ with AVX2 support
- **RAM**: 32 GB+
- **Storage**: NVMe SSD, 50 GB free space
- **Network**: 10 Gbps Ethernet (for distributed mode)

### Note on Instruction Set Support - NO AVX-512
This project does NOT use AVX-512 instruction set. All builds are optimized for:
- SSE4.2: Universal fallback, works on all modern CPUs
- AVX/AVX2: Available where supported, not required for basic operation

**No AVX-512 support**: This project intentionally excludes AVX-512 to maintain compatibility with production homelab systems.

## Production Build Environments

### Node 1 - Production (192.168.10.125)
| Component | Specification |
|-----------|--------------|
| OS | Fedora Linux 44 (Server Edition) |
| CPU | Intel Core i9-14900KF (SSE4.2 → AVX2, no AVX-512 enabled) |
| GPU | NVIDIA GeForce GTX 1060 6GB (Pascal, CC 6.1) |
| Build Type | Release with CUDA support |

### Node 2 - SSE4.2 Only (192.168.10.5)
| Component | Specification |
|-----------|--------------|
| OS | Ubuntu 26.04 LTS |
| CPU | Intel Xeon X5570 (SSE4.2 only, no AVX/AVX2) |
| GPU | None |
| Build Type | Release with SSE4.2 only |

### CI/CD Pipeline
- **Host**: GitHub Actions
- **Runner Environment**: Ubuntu latest container
- **Build Matrix**: SSE4.2, AVX2 configurations
- **Deployment**: Production node via SSH

For detailed build instructions for each environment, see [BUILD.md](BUILD.md).

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

## GPU Hardware Detection

### Check GPU Devices

```bash
# NVIDIA CUDA
nvidia-smi

# AMD ROCm
rocm-smi

# Intel OneAPI
intel_gpu_top

# OpenCL (universal fallback)
clinfo
```

### Verify GPU Support
```bash
# Test CUDA availability
nvcc --version

# Test HIP availability  
hipcc --version

# Test SYCL availability
dpcpp --version
```

## Quick Start - CPU Mode

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

## GPU Installation (Optional)

### Ubuntu/Debian - NVIDIA CUDA
```bash
# Install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2204-11-8-local/cuda-archive-keyring.gpg
sudo apt update
sudo apt install -y cuda-toolkit-11-8

# Set environment variables
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Ubuntu/Debian - AMD ROCm
```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/deb/amdgpu-install_7.0.70000-1_all.deb
sudo dpkg -i amdgpu-install_7.0.70000-1_all.deb

# Install ROCm with HIP support
sudo apt update
sudo apt install -y rocm-hip-sdk rocm-dev

# Add user to video group (required for GPU access)
sudo usermod -a -G video $USER
```

### Ubuntu/Debian - Intel OneAPI
```bash
# Install OneAPI Base Toolkit
wget https://apt.repos.intel.com/intel-gpu-setup.pub
sudo apt-key add intel-gpu-setup.pub
echo "deb [signed-by=intel-gpu-setup.pub] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update
sudo apt install -y intel-oneapi-dev-util intel-oneapi-compiler-dpcpp-cpp
```

### Windows - NVIDIA CUDA
1. Download CUDA Toolkit 11.8 from [NVIDIA website](https://developer.nvidia.com/cuda-toolkit)
2. Run installer with Developer Drivers option
3. Set environment variables:
   ```
   CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
   PATH=%CUDA_PATH%\bin;%PATH%
   ```

**Note:** GPU support is optional. CPU-only mode works without any GPU drivers installed.

## Build from Source

### Linux/macOS Build Steps (CPU)

```bash
# Install C++ dependencies (Ubuntu/Debian)
sudo apt update && sudo apt install -y \
    build-essential cmake git libssl-dev \
    libboost-all-dev libnuma-dev python3-dev

# Clone repository
git clone https://github.com/dllm-project/dLLM.git
cd dLLM

# Build C++ backend (CPU only)
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

### Linux/macOS Build Steps (GPU with CUDA)
```bash
# Build C++ backend with CUDA support
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_CUDA=ON \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 \
      ..
make -j$(nproc)
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

### Basic Configuration (Single Node - CPU)
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

### GPU Configuration (NVIDIA)
```yaml
# config.yaml with GPU support
python_frontend:
  host: "0.0.0.0"
  port: 8000
  workers: 4

backend:
  cpp_library_path: "./build/libdllm.so"
  gpu_backend: cuda         # cuda, hip, sycl
  device_id: 0              # GPU device ID (0 = first GPU)

gpu:
  enabled: true
  memory_pool_size: 8GB
  batch_size: auto

inference:
  max_context_length: 2048
```

### AMD/ATI GPU Configuration
```yaml
backend:
  cpp_library_path: "./build/libdllm.so"
  gpu_backend: hip          # ROCm/HIP backend
  
gpu:
  enabled: true
  device_id: all            # Use all available GPUs
```

### Intel GPU Configuration
```yaml
backend:
  cpp_library_path: "./build/libdllm.so"
  gpu_backend: sycl         # OneAPI/SYCL backend

gpu:
  enabled: true
  device_id: 0
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

### Check GPU Features (Linux)

#### NVIDIA CUDA
```bash
# List all GPUs with driver info
nvidia-smi

# Query compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

#### AMD ROCm
```bash
# List available GPUs
rocm-smi

# Check GPU info
rocminfo | grep -A 5 "GPU"
```

#### Intel OneAPI
```bash
# List GPU devices
intel_gpu_top

# Check GPU information
lspci | grep -i vga
```

### GPU Detection on macOS
- **Note:** macOS uses Metal for Apple Silicon GPUs via ROCm compatibility layer.
- Requires macOS 12+ and Xcode 14+
```bash
# Check GPU model
system_profiler SPHardwareDataType | grep "Chip\|GPU"
```

## Docker Installation with CPU Only (Default)

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

### Docker with NVIDIA GPU Support

```dockerfile
FROM ubuntu:22.04

# Install NVIDIA Container Toolkit prerequisites
RUN apt update && apt install -y \
    build-essential cmake git libssl-dev \
    libboost-all-dev libnuma-dev python3-dev \
    curl gnupg2

# Install CUDA toolkit for GPU support
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
    -o /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
    apt update && apt install -y cuda-toolkit-11-8

WORKDIR /app
COPY . .

# Build C++ backend with GPU support
RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON .. && \
    make -j$(nproc)

# Install Python dependencies
RUN pip install fastapi uvicorn pydantic numpy

CMD ["python", "src/python/server.py"]
```

```bash
# For NVIDIA GPU support, use nvidia-container-toolkit
docker run --gpus all -p 8000:8000 dllm:latest
```

## Running the API Server

### Single Node Mode (CPU)

```bash
cd src/python
python server.py --config ../config.yaml
```

### GPU Mode

```bash
# Use GPU acceleration automatically detected
python server.py --use-gpu

# Or specify backend explicitly
python server.py --gpu-backend cuda  # or hip, sycl
```

### Distributed Mode (CPU or GPU)

```bash
# Start on head node
python server.py \
    --distributed \
    --nodes node1,node2,node3

# Access at http://head-node:8000
```

## Verification

### Test Build (CPU)

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

### Test GPU Installation

```bash
# Verify CUDA is accessible
./build/bin/dllm-gpu-check --backend cuda

# Verify ROCm/HIP is accessible  
./build/bin/dllm-gpu-check --backend hip

# Verify OneAPI/SYCL is accessible
./build/bin/dllm-gpu-check --backend sycl
```

### Test API Server

```bash
curl http://localhost:8000/v1/models

# Expected output:
# {"object":"list","data":[{"id":"llama-7b",...}]}
```

## Troubleshooting

### "Instruction not supported" Error (CPU)
```bash
# Force fallback to lower instruction set
./dllm-infer --instruction-set sse42
```

### CUDA Issues
**Error: "CUDA driver version is insufficient"**
```bash
# Update NVIDIA drivers
sudo apt update && sudo apt upgrade nvidia-driver-525
```

**Error: "CUDA out of memory"**
```yaml
gpu:
  memory_pool_size: 4GB  # Reduce pool size
  batch_size: auto       # Let system determine optimal batch size
```

### ROCm Issues
**Error: "ROCm not found"**
```bash
# Verify ROCm installation
/opt/rocm/bin/clinfo --version

# Check GPU visibility
/opt/rocm/bin/rocm-smi
```

### OneAPI Issues
**Error: "SYCL backend unavailable"**
```bash
# Verify Intel GPU driver
sudo apt install intel-opencl-icd

# Check device detection
intel_gpu_top
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