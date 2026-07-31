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
# Verify the base_url is correct
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Ensure /v1 is included
    api_key="dummy"
)

# Test connection
print(client.models.list())  # Should return list of models
```

### Connection Timeout in Distributed Mode

**Problem**: Nodes cannot communicate.

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
  
# Use tensor parallelism to split model across nodes
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
1. Enable AVX512 instruction set
2. Increase thread count
3. Use larger batch size (if memory permits)
4. Reduce model precision

### High Network Latency

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
