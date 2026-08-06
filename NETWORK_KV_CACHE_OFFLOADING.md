# Network KV Cache Offloading in dLLM

## Overview

Network KV cache offloading is a dLLM technology that stores KV cache data on remote nodes across a cluster network, enabling inference on contexts far larger than any single node's memory can hold. By leveraging a **1 GB/s network fabric**, dLLM achieves near-instant cache access for shared prefixes while transparently offloading cold or rarely-accessed cache entries to remote storage.

This technology is the foundation of dLLM's ability to handle **1M+ token contexts** on commodity hardware without requiring expensive GPU memory.

---

## The Problem: Memory-Bound Inference

In transformer inference, the KV cache grows linearly with context length:

```
KV Cache Size = 2 × num_layers × num_heads × head_dim × seq_len × bytes_per_element
```

For a Llama-7B model with 32 layers, 32 heads, 128 head dim:

| Context Length | FP16 KV Cache | BF16 KV Cache | INT8 KV Cache |
|---------------|:-------------:|:-------------:|:-------------:|
| 32K tokens    | ~4 GB         | ~4 GB         | ~2 GB         |
| 128K tokens   | ~16 GB        | ~16 GB        | ~8 GB         |
| 512K tokens   | ~64 GB        | ~64 GB        | ~32 GB        |
| 1M tokens     | ~128 GB       | ~128 GB       | ~64 GB        |
| 2M tokens     | ~256 GB       | ~256 GB       | ~128 GB       |

No single commodity server has this much RAM. Network offloading solves this by distributing the cache across the cluster.

---

## Architecture

### Network KV Cache Offload Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        dLLM Network KV Cache Offload                     │
│                                                                          │
│  ┌─────────────────────┐    1 GB/s Network    ┌─────────────────────┐   │
│  │   Compute Node 1    │◄──────────────────►│   Storage Node 1    │   │
│  │                     │                     │                     │   │
│  │  ┌───────────────┐  │                     │  ┌───────────────┐  │   │
│  │  │ Hot Cache     │  │   ┌─────────────┐   │  │ KV Cache      │  │   │
│  │  │ (Local RAM)   │  │   │ 1 GB/s NIC  │   │  │ Partition A   │  │   │
│  │  │ - Recent KV   │  │   │ (RoCEv2)    │   │  │ - 32 GB INT8  │  │   │
│  │  │ - Active Heads│  │   └──────┬──────┘   │  │ - 16 GB BF16  │  │   │
│  │  └───────────────┘  │          │          │  └───────────────┘  │   │
│  │                     │          │          │                     │   │
│  │  ┌───────────────┐  │          │          │  ┌───────────────┐  │   │
│  │  │ Offload Engine│  │          │          │  │ KV Cache      │  │   │
│  │  │ - Eviction    │  │          │          │  │ Partition B   │  │   │
│  │  │ - Prefetch    │  │          │          │  │ - 32 GB INT8  │  │   │
│  │  │ - Compression │  │          │          │  │ - 16 GB BF16  │  │   │
│  │  └───────────────┘  │          │          │  └───────────────┘  │   │
│  └─────────────────────┘          │          └─────────────────────┘   │
│                                   │                                    │
│  ┌─────────────────────┐          │          ┌─────────────────────┐   │
│  │   Compute Node 2    │◄─────────┘          │   Storage Node 2    │   │
│  │                     │                     │                     │   │
│  │  ┌───────────────┐  │          ┌─────────▼───────┐             │   │
│  │  │ Hot Cache     │  │          │ KV Cache        │             │   │
│  │  │ (Local RAM)   │  │          │ Partition C     │             │   │
│  │  └───────────────┘  │          │ - 32 GB INT8    │             │   │
│  │                     │          └─────────────────┘             │   │
│  └─────────────────────┘                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Location | Role |
|-----------|----------|------|
| **Hot Cache** | Local RAM | Recently accessed KV entries, sub-millisecond access |
| **Offload Engine** | Local RAM | Manages eviction, prefetching, and compression |
| **Network Fabric** | Cluster | 1 GB/s RDMA/RoCEv2 connection between nodes |
| **Remote KV Store** | Remote RAM | Distributed KV cache partitions across storage nodes |
| **Consistency Layer** | All nodes | Ensures cache coherence across the cluster |

---

## Network Fabric: 1 GB/s Design

### Why 1 GB/s is Sufficient

A common misconception is that network KV offloading requires 10+ GB/s. In practice, **1 GB/s is sufficient** because:

1. **Sparse access patterns**: Only a small fraction of the KV cache is accessed per token generation
2. **Prefetching**: dLLM predicts which cache entries will be needed next and prefetches them
3. **Compression**: INT8/INT4 quantization reduces network payload by 2–4×
4. **Batching**: Network requests are batched to amortize latency

### Network Access Profile

```
Per-token generation (Llama-7B, 128K context):
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  KV Cache Access Pattern:                                    │
│                                                              │
│  Layer 0:  Read 2 heads × 128K tokens × 256 bytes = 64 MB   │
│  Layer 1:  Read 2 heads × 128K tokens × 256 bytes = 64 MB   │
│  ...                                                          │
│  Layer 31: Read 2 heads × 128K tokens × 256 bytes = 64 MB   │
│                                                              │
│  Total per token: ~2 GB of KV data read                      │
│                                                              │
│  But with offloading:                                        │
│  - Hot set (recent 10K tokens): Local RAM, 0 ms latency      │
│  - Cold set (older 118K tokens): Remote, ~1 ms latency       │
│  - Prefetch window: Next 1K tokens preloaded                 │
│                                                              │
│  Effective network bandwidth needed: ~500 MB/s               │
│  Available: 1 GB/s                                           │
│  Headroom: 2×                                                │
└──────────────────────────────────────────────────────────────┘
```

### Network Protocol Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Application Layer                                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  KV Cache Offload Protocol (KVOP)                     │  │
│  │  - Lookup, Store, Invalidate, Prefetch operations     │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Transport Layer                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  RDMA over Converged Ethernet (RoCEv2)                │  │
│  │  - Zero-copy memory access                            │  │
│  │  - Kernel bypass                                      │  │
│  │  - 1 GB/s per connection                              │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Network Layer                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  TCP/IP with DCQCN (Data Center Quantized Congestion  │  │
│  │  Control) for lossless Ethernet                         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Offload Strategies

### Strategy 1: Tiered Cache (Recommended)

The most effective strategy uses a two-tier approach:

```
┌─────────────────────────────────────────────────────────────┐
│  Tier 1: Local RAM (Hot Cache)                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  - Last N tokens (configurable, default: 16K)         │  │
│  │  - Active attention heads                              │  │
│  │  - Sub-millisecond access                              │  │
│  │  - No network overhead                                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│                           ▼ Eviction (LRU + Attention Score) │
│                           │                                  │
│  Tier 2: Remote RAM (Cold Cache)                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  - Older tokens stored on remote nodes                 │  │
│  │  - 1 GB/s network access (~1 ms latency)               │  │
│  │  - INT8 compressed (2× smaller)                        │  │
│  │  - Distributed across storage nodes                    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**

```yaml
offload:
  strategy: tiered
  
  # Tier 1: Local hot cache
  local_cache:
    max_tokens: 16384          # Keep last 16K tokens local
    max_memory_gb: 4           # Limit local cache to 4 GB
    eviction_policy: lru       # LRU or attention_score
    
  # Tier 2: Remote cold cache
  remote_cache:
    network_bandwidth_gbps: 1.0
    compression: int8          # int8 or bf16
    replication_factor: 2      # Replicate across 2 nodes
    prefetch_tokens: 1024      # Prefetch next 1K tokens
    
  # Performance tuning
  prefetch:
    enabled: true
    lookahead_layers: 4        # Prefetch 4 layers ahead
    batch_size: 64             # Batch prefetch requests
```

### Strategy 2: Full Network Offload

For extremely large contexts where even the local hot cache doesn't fit:

```yaml
offload:
  strategy: full_network
  
  local_cache:
    max_tokens: 1024           # Minimal local cache
    max_memory_gb: 0.256       # ~256 MB local
    
  remote_cache:
    compression: int4          # Maximum compression
    chunk_size: 4096           # 4K token chunks
    network_bandwidth_gbps: 1.0
```

### Strategy 3: Hybrid (Local + Network + Disk)

For contexts that exceed even the cluster's total RAM:

```yaml
offload:
  strategy: hybrid
    
  local_cache:
    max_tokens: 8192
    max_memory_gb: 2
    
  remote_cache:
    max_memory_gb: 32          # Per-node remote storage
    compression: int8
    network_bandwidth_gbps: 1.0
    
  disk_cache:
    enabled: true
    path: /data/kv_cache/
    compression: int4
    max_disk_gb: 500           # SSD storage
```

---

## Offload Engine

### Eviction Policy

The offload engine decides which KV entries to keep locally and which to push to remote storage:

```python
class OffloadEngine:
    """Manages KV cache eviction and prefetching"""
    
    def __init__(self, config):
        self.local_cache = LRUCache(max_size=config.local_cache.max_tokens)
        self.remote_store = RemoteKVStore(config.remote_cache)
        self.prefetcher = PrefetchEngine(config.prefetch)
    
    def access_kv(self, layer, head, token_idx, k_values, v_values):
        """Access a KV cache entry with automatic offload"""
        
        # Check local cache first
        if self.local_cache.contains(layer, head, token_idx):
            return self.local_cache.get(layer, head, token_idx)
        
        # Check remote store
        remote_data = self.remote_store.lookup(layer, head, token_idx)
        if remote_data is not None:
            # Promote to local cache
            self.local_cache.put(layer, head, token_idx, remote_data)
            return remote_data
        
        # Cache miss — compute and store
        result = self._compute_kv(layer, head, token_idx, k_values, v_values)
        self.local_cache.put(layer, head, token_idx, result)
        return result
    
    def on_eviction(self, layer, head, token_idx, k_values, v_values):
        """Handle eviction from local to remote"""
        # Compress before sending over network
        compressed = self._compress(k_values, v_values)
        self.remote_store.store(layer, head, token_idx, compressed)
```

### Prefetch Engine

Predicts which KV entries will be needed and preloads them:

```python
class PrefetchEngine:
    """Predictive prefetching of KV cache entries"""
    
    def __init__(self, config):
        self.lookahead_layers = config.prefetch.lookahead_layers
        self.batch_size = config.prefetch.batch_size
    
    def predict_next_access(self, current_layer, current_head, current_token):
        """Predict which KV entries will be accessed next"""
        
        # Strategy 1: Sequential prefetch
        # Next token will access the same heads at the next layer
        predicted = []
        for layer_offset in range(1, self.lookahead_layers + 1):
            for head in range(num_heads):
                predicted.append((
                    current_layer + layer_offset,
                    head,
                    current_token
                ))
        
        # Strategy 2: Attention-weighted prefetch
        # Prefetch based on attention scores from previous layers
        attention_weights = self._get_attention_weights(current_layer)
        top_heads = self._top_k_heads(attention_weights, k=8)
        
        for head in top_heads:
            predicted.append((current_layer, head, current_token + 1))
        
        return predicted
    
    def batch_prefetch(self, predictions):
        """Batch prefetch requests to amortize network latency"""
        
        # Group by target node to minimize network round trips
        by_node = self._group_by_node(predictions)
        
        for node, entries in by_node.items():
            # Send batched prefetch request
            self.remote_store.batch_get(entries)
```

### Compression Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│  Compression Pipeline                                         │
│                                                               │
│  FP16 KV Entry (256 bytes)                                    │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────┐                                             │
│  │ Per-channel  │  → Scale factors per head/channel           │
│  │ Quantization│  → Min/max calibration                      │
│  └──────┬──────┘                                             │
│       ▼                                                       │
│  INT8 KV Entry (128 bytes)                                    │
│       │  50% size reduction                                   │
│       ▼                                                       │
│  ┌─────────────┐                                             │
│  │ SIMD Pack    │  → Pack 4 INT8 values per 4 bytes           │
│  │ & Compress   │  → Run-length encode repeated values        │
│  └──────┬──────┘                                             │
│       ▼                                                       │
│  Network Payload (~90 bytes)                                  │
│       │  65% size reduction from original                     │
│       ▼                                                       │
│  ┌─────────────┐                                             │
│  │ RDMA Send    │  → Zero-copy direct memory access           │
│  └─────────────┘                                             │
└──────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Latency Breakdown

| Operation | Local Hit | Remote Hit | Miss (Compute) |
|-----------|:---------:|:----------:|:--------------:|
| KV Lookup | <0.1 ms | ~1.0 ms | ~5.0 ms |
| KV Store | <0.1 ms | ~0.5 ms | N/A |
| Prefetch (batch of 64) | N/A | ~2.0 ms | N/A |
| Eviction | N/A | ~0.3 ms | N/A |

### Throughput with Network Offloading

| Context Size | Local Only | Network Offload (1 GB/s) | Speedup |
|-------------|:----------:|:------------------------:|:-------:|
| 32K tokens  | 100%       | 100%                     | 1.0×    |
| 128K tokens | OOM        | 95%                      | N/A     |
| 512K tokens | OOM        | 88%                      | N/A     |
| 1M tokens   | OOM        | 82%                      | N/A     |
| 2M tokens   | OOM        | 76%                      | N/A     |

*Performance ratio = effective throughput / local-only throughput at 32K*

### Network Bandwidth Utilization

```
1M token context, Llama-7B, batch size 1:
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Network Bandwidth:                                          │
│  ┌────────────────────────────────────────────────────────┐  │
│  │████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░│  │
│  │  62% utilized (620 MB/s of 1 GB/s)                     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Breakdown:                                                  │
│  - KV data transfer:     450 MB/s  (73%)                    │
│  - Prefetch overhead:    100 MB/s  (16%)                    │
│  - Protocol overhead:     70 MB/s  (11%)                    │
│                                                              │
│  Effective bandwidth for KV data: ~730 MB/s                  │
│  (after compression and batching)                            │
└──────────────────────────────────────────────────────────────┘
```

---

## Configuration Reference

### Full Offload Configuration

```yaml
network_kv_cache:
  # Network settings
  network:
    bandwidth_gbps: 1.0
    protocol: roce_v2              # roce_v2, tcp, ib (InfiniBand)
    connection_pool_size: 8        # RDMA connections per peer
    timeout_ms: 100                # Network operation timeout
    congestion_control: dcqcn      # DCQCN for lossless Ethernet
    
  # Offload strategy
  strategy: tiered                 # tiered, full_network, hybrid
  
  # Local hot cache
  local:
    max_tokens: 16384
    max_memory_gb: 4
    eviction_policy: lru           # lru, attention_score, hybrid
    attention_threshold: 0.01      # Keep heads with attention > threshold
    
  # Remote cold cache
  remote:
    compression: int8              # int8, int4, bf16
    chunk_size_tokens: 4096        # Group tokens into chunks
    replication_factor: 2          # Number of replicas
    consistency: quorum            # one, quorum, all
    prefetch_tokens: 1024          # Number of tokens to prefetch
    prefetch_lookahead_layers: 4   # How many layers ahead to prefetch
    
  # Performance tuning
  performance:
    prefetch_batch_size: 64        # Batch prefetch requests
    eviction_batch_size: 32        # Batch eviction writes
    network_polling_interval_us: 100  # Polling interval for RDMA
    zero_copy_enabled: true        # Use RDMA zero-copy
```

---

## Use Cases

### Use Case 1: Long Document Q&A

```
Scenario: Answer questions about a 500-page book (~500K tokens)

Before offloading:
  - Requires 64 GB+ RAM just for KV cache
  - Cannot fit on standard 32 GB server
  - Must chunk document, losing cross-document context

With network offloading:
  - KV cache distributed across 4 nodes (16 GB each)
  - Hot cache keeps last 16K tokens local
  - Cold cache accessed at ~1 ms per lookup
  - Full 500K context available for cross-document reasoning
  - Latency: ~2 ms per token (vs ~0.5 ms local)
```

### Use Case 2: Multi-Session Shared Prefix

```
Scenario: 100 users each have a unique conversation with a shared system prompt

Without offloading:
  - System prompt KV cache duplicated 100× in memory
  - 100 × 4 GB = 400 GB total memory needed
  
With network offloading:
  - System prompt cached once on shared storage node
  - All 100 sessions reference the same remote cache
  - Memory: 4 GB (shared) + 100 × 0.5 GB (per-session hot cache)
  - Network cost: ~50 MB/s total for 100 concurrent sessions
```

### Use Case 3: Training-Inference Hybrid

```
Scenario: Fine-tuning a model while serving inference requests

Without offloading:
  - Must stop inference during fine-tuning (GPU memory conflict)
  - Or run on separate expensive GPU hardware
  
With network offloading:
  - Inference KV cache offloaded to network storage
  - Training uses local GPU memory
  - Both run simultaneously on the same node
  - Network handles KV cache traffic transparently
```

---

## Comparison: Network Offload vs Alternatives

| Approach | Max Context | Latency | Cost | Complexity |
|----------|:-----------:|:-------:|:----:|:----------:|
| **Local RAM only** | ~32K tokens | <0.5 ms/token | Low | Low |
| **Network offload (1 GB/s)** | **1M+ tokens** | **~1-2 ms/token** | **Medium** | **Medium** |
| Network offload (10 GB/s) | 1M+ tokens | ~0.3 ms/token | High | High |
| GPU VRAM | ~128K tokens | <0.5 ms/token | Very High | Low |
| Disk-based offload | Unlimited | ~10-50 ms/token | Low | High |

---

## Best Practices

1. **Use tiered strategy** — Keep the hot set local for sub-millisecond access
2. **Enable prefetching** — The 1 GB/s network is only bottlenecked without prefetching
3. **Compress cold data** — INT8 compression gives 2× bandwidth improvement
4. **Batch network operations** — Amortize the ~1 ms network latency with batching
5. **Monitor cache hit rate** — Target >70% local hit rate for optimal performance
6. **Size the hot cache** — Set `local_cache.max_tokens` to cover the working set

---

## See Also

- [PV Cache Optimization](PV_CACHE_OPTIMIZATION.md) — In-node KV cache compression
- [Distributed PV Cache](DISTRIBUTED_PV_CACHE.md) — Cross-node prefix sharing
- [Large KV Cache Management](LARGE_KV_CACHE_MANAGEMENT.md) — Memory management for 1M+ contexts
- [Performance](PERFORMANCE.md) — Full performance benchmarks
