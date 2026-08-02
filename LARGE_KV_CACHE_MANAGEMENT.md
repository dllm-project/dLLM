# Large KV Cache Management in dLLM

## Overview

Managing KV caches for contexts exceeding 1,000,000 tokens presents unique challenges. This document covers:

- **Memory-efficient storage**: Compression and quantization strategies
- **Eviction policies**: Smart cache eviction for large contexts
- **Streaming processing**: Handle context that exceeds memory
- **Distributed caching**: Spread cache across cluster nodes

## Memory Challenges

### Storage Requirements

| Context Size | FP16 (GB) | BF16 (GB) | INT8 (GB) | INT4 (GB) |
|--------------|-----------|-----------|-----------|-----------|
| 10K tokens   | 0.4       | 0.4       | 0.2       | 0.1       |
| 100K tokens  | 4.0       | 4.0       | 2.0       | 1.0       |
| 1M tokens    | 40        | 40        | 20        | 10        |
| 2M tokens    | 80        | 80        | 40        | 20        |
| 5M tokens    | 200       | 200       | 100       | 50        |

### Memory Optimization Goals

```yaml
optimization_targets:
  memory_reduction: 60%      # Target reduction from baseline
  latency_overhead: <10%     # Acceptable latency increase
  hit_rate: >70%            # Cache hit rate target
```

## Compression Strategies

### Quantization Levels

```python
class KVQuantizer:
    """Quantize KV cache for memory efficiency"""
    
    def __init__(self, strategy='dynamic'):
        self.strategy = strategy
    
    def quantize(self, k_cache, v_cache):
        """Apply quantization to KV cache"""
        if self.strategy == 'fp16':
            return k_cache.astype(np.float16), v_cache.astype(np.float16)
        
        elif self.strategy == 'bf16':
            return self._to_bf16(k_cache), self._to_bf16(v_cache)
        
        elif self.strategy == 'int8':
            return self._quantize_int8(k_cache), self._quantize_int8(v_cache)
        
        elif self.strategy == 'int4':
            return self._quantize_int4(k_cache), self._quantize_int4(v_cache)
    
    def _to_bf16(self, tensor):
        """Convert to bfloat16"""
        # Preserve exponent, truncate mantissa
        return tensor.astype(np.float32).view(np.uint32) >> 16
    
    def _quantize_int8(self, tensor):
        """Quantize to int8"""
        max_val = np.abs(tensor).max()
        scale = max_val / 127.0
        quantized = (tensor / scale).astype(np.int8)
        return quantized, scale
    
    def _quantize_int4(self, tensor):
        """Quantize to int4 (nibble-level)"""
        # Pack two values per byte
        quantized = np.zeros(len(tensor) // 2, dtype=np.uint8)
        for i in range(0, len(tensor), 2):
            val1 = min(max(int(tensor[i] * 7.5 + 7.5), 0), 15)
            val2 = min(max(int(tensor[i+1] * 7.5 + 7.5), 0), 15)
            quantized[i // 2] = (val1 << 4) | val2
        return quantized
```

### Layer-wise Quantization

```yaml
quantization:
  strategy: per_layer
  
  # Different precision for different layers
  layers:
    attention_k:
      strategy: int8          # Keys can be more compressed
      threshold: 0.95         # Only compress if similarity > threshold
    
    attention_v:
      strategy: bf16          # Values need higher precision
      threshold: 0.99
    
    mlp:
      strategy: fp16          # MLP activations use FP16
```

## Eviction Policies

### LRU (Least Recently Used)

```python
class LRUEvictionPolicy:
    """LRU eviction for KV cache"""
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = OrderedDict()  # hash -> (k_data, v_data, size)
        self.access_order = {}      # hash -> last_access_time
    
    def access(self, prefix_hash):
        """Record access to prefix"""
        if prefix_hash in self.cache:
            self.access_order[prefix_hash] = time.time()
    
    def evict(self, needed_size):
        """Evict entries to free needed space"""
        freed = 0
        
        # Sort by last access (oldest first)
        sorted_hashes = sorted(
            self.access_order.items(),
            key=lambda x: x[1]
        )
        
        for prefix_hash, _ in sorted_hashes:
            if prefix_hash not in self.cache:
                continue
            
            _, _, size = self.cache[prefix_hash]
            
            del self.cache[prefix_hash]
            del self.access_order[prefix_hash]
            
            freed += size
            
            if freed >= needed_size:
                break
        
        return freed
```

### Priority-based Eviction

```python
class PriorityEvictionPolicy:
    """Priority-based eviction considering request importance"""
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = {}  # hash -> (k_data, v_data, size, priority)
        self.priority_weights = {
            'chat_history': 1.0,
            'document_context': 0.8,
            'code_context': 0.9,
            'temporary': 0.5
        }
    
    def get_priority(self, prefix_type):
        """Get priority weight for prefix type"""
        return self.priority_weights.get(prefix_type, 0.7)
    
    def evict(self, needed_size):
        """Evict lowest priority entries first"""
        # Sort by (priority, access_time) - lowest first
        sorted_items = sorted(
            self.cache.items(),
            key=lambda x: (x[1][3], x[1][4])  # priority, access_time
        )
        
        freed = 0
        for prefix_hash, (_, _, size, _) in sorted_items:
            del self.cache[prefix_hash]
            freed += size
            
            if freed >= needed_size:
                break
        
        return freed
```

### Sliding Window Eviction

```python
class SlidingWindowEviction:
    """Sliding window eviction for sequential access patterns"""
    
    def __init__(self, window_size):
        self.window_size = window_size
        self.prefixes = []  # Ordered list of prefix hashes
    
    def add_prefix(self, prefix_hash):
        """Add prefix to sliding window"""
        if prefix_hash in self.prefixes:
            self.prefixes.remove(prefix_hash)
        
        self.prefixes.append(prefix_hash)
        
        # Evict oldest if over window size
        while len(self.prefixes) > self.window_size:
            oldest = self.prefixes.pop(0)
            yield oldest  # Return for eviction
    
    def get_recent_prefixes(self, count):
        """Get most recent prefixes"""
        return self.prefixes[-count:]
```

## Streaming KV Cache

### Out-of-Core Processing

```python
class StreamingKVCache:
    """Handle KV cache that exceeds memory"""
    
    def __init__(self, max_memory_gb=16):
        self.max_memory = max_memory_gb * 1024**3  # Convert to bytes
        self.memory_used = 0
        self.disk_cache = DiskCache()
        
        # In-memory portion (most recent)
        self.active_cache = {}
    
    def store_prefix(self, prefix_hash, k_data, v_data):
        """Store prefix with streaming support"""
        size = k_data.nbytes + v_data.nbytes
        
        if self.memory_used + size <= self.max_memory:
            # Store in memory
            self.active_cache[prefix_hash] = (k_data, v_data)
            self.memory_used += size
        else:
            # Evict and move to disk
            self._evict_to_disk(prefix_hash, k_data, v_data)
    
    def _evict_to_disk(self, prefix_hash, k_data, v_data):
        """Move prefix to disk cache"""
        self.disk_cache.put(prefix_hash, (k_data, v_data))
        
        # Free memory if needed
        if self.memory_used > self.max_memory * 0.8:
            self._evict_lru()
    
    def retrieve_prefix(self, prefix_hash):
        """Retrieve prefix with streaming support"""
        if prefix_hash in self.active_cache:
            return self.active_cache[prefix_hash]
        
        elif prefix_hash in self.disk_cache:
            k_data, v_data = self.disk_cache.get(prefix_hash)
            
            # Move to memory if space available
            size = k_data.nbytes + v_data.nbytes
            if self.memory_used + size <= self.max_memory:
                self.active_cache[prefix_hash] = (k_data, v_data)
                self.memory_used += size
            
            return k_data, v_data
        
        else:
            raise CacheMiss()
```

### Chunked Processing

```python
class ChunkedKVCache:
    """Process KV cache in chunks"""
    
    def __init__(self, chunk_size=1024):
        self.chunk_size = chunk_size
        self.chunks = []  # List of (k_chunk, v_chunk)
    
    def process_request(self, tokens):
        """Process request with chunked KV cache"""
        results = []
        
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            
            # Process chunk
            k_cache, v_cache = self._compute_chunk(chunk_tokens)
            
            # Store chunk
            self.chunks.append((k_cache, v_cache))
            
            # Get attention weights for this chunk
            chunk_result = self._attention_with_chunks(k_cache, v_cache)
            results.append(chunk_result)
        
        return self._combine_results(results)
```

## Memory Monitoring

### Usage Tracking

```python
class KVCacheMonitor:
    """Monitor KV cache memory usage"""
    
    def __init__(self):
        self.total_allocated = 0
        self.active_count = 0
        self.peak_usage = 0
        
        # Per-prefix tracking
        self.prefix_sizes = {}  # hash -> size
    
    def record_allocation(self, prefix_hash, size):
        """Record new allocation"""
        self.total_allocated += size
        self.active_count += 1
        self.peak_usage = max(self.peak_usage, self.total_allocated)
        
        self.prefix_sizes[prefix_hash] = size
    
    def record_eviction(self, prefix_hash):
        """Record eviction"""
        if prefix_hash in self.prefix_sizes:
            size = self.prefix_sizes.pop(prefix_hash)
            self.total_allocated -= size
            self.active_count -= 1
    
    @property
    def usage_percent(self):
        """Get memory usage percentage"""
        # Assume max is 32GB for calculation
        max_memory = 32 * 1024**3
        return (self.total_allocated / max_memory) * 100
```

### Alerting Configuration

```yaml
monitoring:
  memory_thresholds:
    warning: 70      # Alert at 70% usage
    critical: 90     # Critical at 90% usage
    
  eviction_alerts:
    enabled: true
    min_eviction_rate: 10  # Per minute
    
  metrics:
    - name: kv_cache_memory_usage_bytes
      type: gauge
    
    - name: kv_cache_eviction_rate
      type: counter
    
    - name: kv_cache_hit_rate
      type: histogram
```

## Best Practices

### 1. Quantization Selection

| Use Case | Recommended Quantization |
|----------|-------------------------|
| Research/High accuracy | BF16 |
| Production (balanced) | INT8 |
| Cost-sensitive | INT4 |

### 2. Memory Budgeting

```bash
# Calculate required memory
# Formula: (context_size × 2 × bytes_per_token) / compression_ratio

# Example for 1M tokens with INT8:
# (1,000,000 × 2 × 1 byte) / 2 = 1GB effective
```

### 3. Eviction Policy Selection

| Workload | Recommended Policy |
|----------|-------------------|
| Chat (sequential) | Sliding Window |
| Document processing | LRU |
| Mixed workload | Priority-based |

## Performance Optimization

### Prefetching Strategy

```python
class KVCachePrefetcher:
    """Prefetch likely needed prefixes"""
    
    def __init__(self, cache):
        self.cache = cache
        self.predictor = PrefixPredictor()
    
    def prefetch(self, current_prefix):
        """Prefetch based on prediction"""
        # Predict next likely prefixes
        predictions = self.predictor.predict(current_prefix, top_k=5)
        
        for prefix in predictions:
            if not self.cache.has(prefix):
                self.cache.prefetch(prefix)
```

### Batch Operations

```python
class BatchedKVCache:
    """Batch operations for efficiency"""
    
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.pending_ops = []
    
    def add_operation(self, op_type, prefix_hash, data=None):
        """Add operation to batch"""
        self.pending_ops.append({
            'type': op_type,
            'hash': prefix_hash,
            'data': data
        })
        
        if len(self.pending_ops) >= self.batch_size:
            self._execute_batch()
    
    def _execute_batch(self):
        """Execute pending operations in batch"""
        # Group by type
        lookups = [op for op in self.pending_ops if op['type'] == 'lookup']
        inserts = [op for op in self.pending_ops if op['type'] == 'insert']
        
        # Execute batch lookup
        if lookups:
            results = self.cache.batch_lookup([op['hash'] for op in lookups])
        
        # Execute batch insert
        if inserts:
            self.cache.batch_insert(inserts)
        
        self.pending_ops.clear()
```

## Troubleshooting

### Common Issues

**Memory Exhaustion**
- Enable more aggressive quantization
- Reduce max context size
- Implement stricter eviction policies

**High Eviction Rate**
- Increase cache memory budget
- Use sliding window for sequential workloads
- Adjust priority weights

**Slow Performance**
- Enable prefetching
- Use batch operations
- Consider distributed caching

## Future Enhancements

### Planned Features

1. **Adaptive Quantization**: Dynamic precision based on attention patterns
2. **ML-based Eviction**: Predictive eviction using lightweight model
3. **GPU-Accelerated Compression**: CUDA/ROCm optimized quantization
4. **Tiered Storage**: SSD/HDD for cold cache entries
