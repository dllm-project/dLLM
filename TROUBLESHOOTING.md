# Troubleshooting dLLM

## Common Issues

### "Instruction set not supported" Error

**Problem**: CPU doesn't support the requested instruction set.

**Solutions**:
```bash
# Auto-detect and use highest available
./dllm-infer --instruction-set auto

# Force fallback to lower instruction set
./dllm-infer --instruction-set avx2

# Check supported instruction sets
./dllm-info
```

### Python Frontend Not Starting

**Problem**: Port 8000 is already in use.

**Solutions**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python server.py --port 8001
```

### C++ Backend Not Loading

**Problem**: Cannot find or load libdllm.so.

**Solutions**:
```yaml
# Verify library path in config
backend:
  cpp_library_path: "./build/libdllm.so"

# Check library exists
ls -la build/libdllm.so

# Rebuild if necessary
cd build && make
```

### OpenAI API Compatibility Issues

**Problem**: OpenAI SDK doesn't work with the server.

**Solutions**:
```python
# Verify the base_url is correct (note: /api prefix, not /v1)
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/api",  # Use /api prefix
    api_key="dummy"
)

# Test connection
print(client.models.list())  # Should return list of models
```

### Connection Timeout in Distributed Mode

> **Note**: Distributed computing features are planned but not yet implemented. The `src/cpp/comm/` directory contains CMakeLists.txt references but no source files.

**Solutions**:
```yaml
# Increase timeout
network:
  connect_timeout: 30s
  read_timeout: 60s
  
# Check firewall
ufw allow 5000/tcp

# Verify connectivity
./dllm-cluster-check --nodes node1,node2,node3
```

### Out of Memory Error

**Problem**: Model too large for available memory.

**Solutions**:
```yaml
# Enable quantization
quantization:
  enabled: true
  dtype: int8  # or fp16, bf16

# Reduce batch size
inference:
  max_batch_size: 8
  
# Use tensor parallelism to split model across nodes (planned)
tensor_parallelism:
  enabled: true
```

## Performance Issues

### Slow Inference

**Diagnosis**:
```bash
./dllm-benchmark --model /models/llama-7b
```

**Solutions**:
1. Enable AVX2 instruction set (AVX-512 is not supported)
2. Increase thread count
3. Use larger batch size (if memory permits)
4. Reduce model precision

### High Network Latency

> **Note**: Distributed computing features are planned but not yet implemented.

**Diagnosis**:
```bash
./dllm-network-test --bandwidth-target 1GB/s
```

**Solutions**:
```yaml
network:
  protocol: tcp
  compression_threshold: 8192  # Compress larger messages
  
# Enable connection pooling
pool_size: 256
keepalive_time: 60s
```

### Slow Python Server Response

**Diagnosis**: High latency from OpenAI API endpoint.

**Solutions**:
```yaml
server:
  workers: 16              # Increase worker count
  timeout_keep_alive: 120  # Keep connections alive
  
# Check C++ backend connection
backend:
  cpp_library_path: "./build/libdllm.so"
```

## Debug Mode

### Verbose Logging - Python Server
```bash
python server.py \
    --log-level trace \
    --verbose
```

### Verbose Logging - C++ Backend
```bash
# Set environment variable for debug logging
export DLLM_LOG_LEVEL=trace
./dllm-server --debug
```

### Memory Profiling
```bash
./dllm-profile --memory \
               --tensor-stats

# Python profiling
python -m cProfile -o stats.out server.py
snakeviz stats.out
```

## SIMD Instruction Set Issues

### Checking Supported Instruction Sets

```bash
# Check CPU instruction set support
grep -o -E 'avx512|avx2|avx|sse4_2' /proc/cpuinfo | sort -u

# Expected output for modern CPUs:
# avx
# avx2
# sse4_2
```

### Build with Correct Instruction Set

```bash
# SSE4.2 only (baseline)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_SSE42=ON \
      -DUSE_AVX=OFF \
      -DUSE_AVX2=OFF \
      ..

# AVX + AVX2 (recommended)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DUSE_SSE42=ON \
      -DUSE_AVX=ON \
      -DUSE_AVX2=ON \
      ..
```

> **Note**: AVX-512 is intentionally not supported. This project targets SSE4.2 → AVX2.

## GPU Issues

> **Note**: GPU acceleration is declared as CMake options but no GPU backend source code exists yet.

### CUDA Not Detected

**Problem**: CUDA GPU not detected.

**Solutions**:
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA toolkit
nvcc --version

# Check CUDA libraries
ldconfig -p | grep cuda

# Reinstall NVIDIA driver if needed
sudo apt-get install --reinstall nvidia-driver-535
```

### ROCm Not Detected

**Problem**: ROCm GPU not detected.

**Solutions**:
```bash
# Check ROCm installation
rocm-smi

# Check ROCm libraries
ldconfig -p | grep rocm

# Verify ROCm version
rocminfo | grep "ROCm Version"
```

### SYCL Not Detected

**Problem**: SYCL GPU not detected.

**Solutions**:
```bash
# Check OneAPI installation
intel_gpu_top

# Check SYCL libraries
ldconfig -p | grep sycl

# Verify OneAPI version
clinfo | grep "Platform Name"
```

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs**: Look for error messages in `logs/` directory
2. **Search GitHub Issues**: Check if others have reported similar issues
3. **Create a New Issue**: Provide detailed information about your problem
4. **Join the Community**: Ask for help in our community channels

### Information to Include

When reporting issues, please include:

- **System Information**: OS, CPU, GPU, Python version
- **Build Configuration**: CMake flags used
- **Error Messages**: Full error output
- **Steps to Reproduce**: How to reproduce the issue
- **Expected vs Actual Behavior**: What you expected vs what happened
