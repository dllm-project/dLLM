# PV Cache API Reference in dLLM

## Overview

This document provides comprehensive API reference for the PV (Prefix Vector) cache functionality in dLLM. The API supports:

- **Python**: OpenAI-compatible API with PV cache extensions
- **C++**: Direct C++ interface for high-performance applications
- **HTTP REST**: Raw HTTP endpoints for integration

## Python API

### Basic Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Enable PV cache with default settings
response = client.chat.completions.create(
    model="llama-7b",
    messages=[
        {"role": "system", "content": "System prompt..."},
        {"role": "user", "content": "User message..."}
    ],
    extra_body={
        "pv_cache": {
            "enabled": True
        }
    }
)
```

### Full PV Cache Configuration

```python
response = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "pv_cache": {
            # Enable/disable PV cache
            "enabled": True,
            
            # Prefix detection
            "prefix_length": 4096,           # Maximum prefix length
            "min_prefix_length": 64,         # Minimum for meaningful hash
            
            # Quantization
            "quantization": "bf16",          # fp16, bf16, int8, int4
            "quantization_threshold": 0.95,  # Similarity threshold
            
            # Distribution (cluster mode)
            "distributed": True,
            "replication_factor": 2,
            
            # Performance tuning
            "prefetch_count": 3,
            "batch_lookup": True,
        }
    }
)
```

### Cache Management API

```python
# Get cache statistics
response = client.get("/v1/pv_cache/stats")
print(response.json())

# Clear cache
response = client.post("/v1/pv_cache/clear")

# Get prefix by hash
response = client.get("/v1/pv_cache/prefix/{hash}")

# Insert custom prefix
response = client.post(
    "/v1/pv_cache/insert",
    json={
        "prefix_hash": "abc123...",
        "k_values": [...],  # Compressed K values
        "v_values": [...]   # Compressed V values
    }
)
```

## C++ API

### Core Cache Interface

```cpp
// Include the header
#include "pv_cache/pv_cache.h"

// Create cache instance
dllm::PVCache cache({
    .max_prefix_length = 8192,
    .min_prefix_length = 64,
    .quantization = dllm::Quantization::BF16,
    .distributed = true
});

// Compute prefix hash
std::string hash = cache.computeHash(prefix_tokens);

// Lookup prefix
auto result = cache.lookup(hash);
if (result.has_value()) {
    auto [k_cache, v_cache] = result.value();
    // Use cached values
}

// Insert prefix
cache.insert(hash, k_values, v_values);
```

### Distributed Cache Interface

```cpp
#include "pv_cache/distributed_pv_cache.h"

// Create distributed cache
dllm::DistributedPVCache distributed_cache({
    .nodes = {"node1:8001", "node2:8001", "node3:8001"},
    .replication_factor = 2,
    .consistency_level = dllm::ConsistencyLevel::Quorum
});

// Batch lookup (optimized for multiple prefixes)
std::vector<std::string> hashes = {hash1, hash2, hash3};
auto results = distributed_cache.batchLookup(hashes);

// Batch insert with replication
distributed_cache.batchInsert(entries);
```

### Vector Storage Interface

```cpp
#include "pv_cache/vector_storage.h"

// Create compressed vector storage
dllm::VectorStorage storage({
    .quantization = dllm::Quantization::INT8,
    .max_size_gb = 32
});

// Store vectors
int vector_id = storage.store(k_values, v_values);

// Retrieve vectors
std::vector<float> k_retrieved, v_retrieved;
storage.retrieve(vector_id, k_retrieved, v_retrieved);
```

## HTTP REST API

### Cache Statistics

```http
GET /v1/pv_cache/stats HTTP/1.1
Host: localhost:8000

Response:
{
    "hit_count": 12345,
    "miss_count": 6789,
    "total_lookups": 19134,
    "hit_rate": 0.645,
    "memory_usage_bytes": 1073741824,
    "prefix_count": 5432,
    "distributed_nodes": 3
}
```

### Clear Cache

```http
POST /v1/pv_cache/clear HTTP/1.1
Host: localhost:8000

Response:
{
    "success": true,
    "cleared_prefixes": 5432
}
```

### Insert Prefix

```http
POST /v1/pv_cache/insert HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
    "prefix_hash": "abc123def456...",
    "k_values": [0.1, 0.2, ...],     // Compressed K values
    "v_values": [0.3, 0.4, ...],     // Compressed V values
    "quantization": "int8",          // Optional: override default
    "metadata": {                    // Optional metadata
        "token_count": 1024,
        "created_at": "2026-08-02T12:00:00Z"
    }
}

Response:
{
    "success": true,
    "hash": "abc123def456...",
    "vector_id": 789
}
```

### Lookup Prefix

```http
POST /v1/pv_cache/lookup HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
    "prefix_hashes": ["abc123...", "def456..."]
}

Response:
{
    "results": {
        "abc123...": {
            "found": true,
            "k_values": [...],
            "v_values": [...]
        },
        "def456...": {
            "found": false
        }
    }
}
```

## Command Line Interface

### Cache Statistics

```bash
# Get cache statistics
dllm pv-cache stats

# Output:
# Hit Rate: 64.5%
# Memory Usage: 1024 MB
# Prefix Count: 5432
```

### Clear Cache

```bash
# Clear all cached prefixes
dllm pv-cache clear

# Clear with confirmation
dllm pv-cache clear --confirm
```

### Export/Import

```bash
# Export cache to file
dllm pv-cache export --output /path/to/cache.bin

# Import cache from file
dllm pv-cache import --input /path/to/cache.bin
```

## Configuration Reference

### Python Configuration

```python
pv_cache_config = {
    # Enable PV cache
    "enabled": True,
    
    # Prefix detection parameters
    "prefix_length": 4096,           # Maximum prefix length in tokens
    "min_prefix_length": 64,         # Minimum for meaningful hash
    
    # Hash configuration
    "hash_algorithm": "sha3_256",    # sha3_256, blake3
    "approximate_match": True,       # Enable vector similarity matching
    "similarity_threshold": 0.92,    # Cosine similarity threshold
    
    # Quantization
    "quantization": "bf16",          # fp16, bf16, int8, int4
    "per_layer_quantization": {      # Per-layer quantization
        "attention_k": "int8",
        "attention_v": "bf16"
    },
    
    # Memory management
    "max_cache_size_gb": 32,
    "eviction_policy": "lru",        # lru, fifo, priority
    
    # Distribution (cluster mode)
    "distributed": True,
    "replication_factor": 2,
    "consistency_level": "quorum",   # one, quorum, all
    
    # Performance tuning
    "prefetch_count": 3,
    "batch_lookup": True,
    "async_eviction": True,
    
    # Monitoring
    "monitor_hit_rate": True,
    "monitor_memory": True,
}
```

### YAML Configuration

```yaml
pv_cache:
  enabled: true
  
  prefix_detection:
    max_length: 8192
    min_length: 64
    hash_algorithm: sha3_256
    approximate_match: true
    similarity_threshold: 0.92
  
  quantization:
    strategy: dynamic
    layers:
      attention_k: int8
      attention_v: bf16
  
  memory:
    max_size_gb: 32
    eviction_policy: lru
  
  distribution:
    enabled: true
    replication_factor: 2
    consistency_level: quorum
  
  performance:
    prefetch_count: 3
    batch_lookup: true
```

## Error Codes

### Cache Errors

| Code | Message | Description |
|------|---------|-------------|
| 1001 | CACHE_MISS | Prefix not found in cache |
| 1002 | CACHE_FULL | Cache has reached maximum size |
| 1003 | HASH_MISMATCH | Computed hash doesn't match stored hash |
| 1004 | QUANTIZATION_ERROR | Failed to quantize/dequantize values |

### Distributed Cache Errors

| Code | Message | Description |
|------|---------|-------------|
| 2001 | NODE_UNAVAILABLE | Node is not reachable |
| 2002 | REPLICATION_FAILED | Failed to replicate to all nodes |
| 2003 | CONSENSUS_TIMEOUT | Consensus protocol timed out |

## Performance Tuning

### Recommended Settings by Use Case

```python
# Chat application (high hit rate)
chat_config = {
    "prefix_length": 1024,
    "quantization": "int8",
    "prefetch_count": 5,
}

# Document processing (large prefixes)
document_config = {
    "prefix_length": 8192,
    "quantization": "bf16",
    "approximate_match": False,  # Exact match preferred
}

# Code generation (balanced)
code_config = {
    "prefix_length": 2048,
    "quantization": "int8",
    "prefetch_count": 3,
}
```

## Best Practices

### 1. Prefix Length Selection

- **Short prefixes** (64-256): High hit rate, less memory
- **Medium prefixes** (1024-4096): Balanced approach
- **Long prefixes** (8192+): Lower hit rate but more context reuse

### 2. Quantization Strategy

```yaml
# High accuracy (research)
quantization: bf16

# Production (balanced)
quantization: int8

# Cost-sensitive (maximum compression)
quantization: int4
```

### 3. Cache Sizing

```bash
# Estimate cache size
# Formula: prefix_count × avg_prefix_length × bytes_per_token

# Example: 10K prefixes × 2048 tokens × 2 bytes = 40MB
```

## Troubleshooting

### Common Issues

**Low Hit Rate**
- Increase `prefix_length`
- Enable `approximate_match`
- Check for tokenization variations

**High Memory Usage**
- Reduce `max_cache_size_gb`
- Use more aggressive quantization
- Enable eviction policies

**Distributed Cache Issues**
- Verify node connectivity
- Adjust `replication_factor`
- Check network bandwidth
