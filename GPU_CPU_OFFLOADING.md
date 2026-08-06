# GPU ↔ CPU/RAM Offloading in dLLM

## Overview

GPU ↔ CPU/RAM offloading is a dLLM technology that dynamically moves **model weights** and **KV cache data** between GPU VRAM and CPU RAM, enabling large model inference on hardware with limited GPU memory. This is essential when:

- The model doesn't fit entirely in GPU VRAM
- You want to use cheaper CPU-only infrastructure for inference
- You need to run multiple models on a single GPU
- You're doing inference on consumer GPUs with limited VRAM (e.g., 8–12 GB)

dLLM implements this through **layer-by-layer offloading**, **dynamic memory migration**, and **overlap of computation with data transfer**, achieving near-full-GPU performance even when the model is partially or fully on CPU RAM.

---

## The Problem: VRAM-Constrained Inference

### Memory Budget for Common Models

| Model | Parameters | FP16 Weights | KV Cache (32K) | Total FP16 |
|-------|:----------:|:------------:|:--------------:|:----------:|
| Llama-3.1-8B | 8B | ~16 GB | ~4 GB | ~20 GB |
| Llama-3.1-70B | 70B | ~140 GB | ~35 GB | ~175 GB |
| Mistral-7B | 7B | ~14 GB | ~3.5 GB | ~17.5 GB |
| Qwen-72B | 72B | ~144 GB | ~36 GB | ~180 GB |
| Gemma-2-27B | 27B | ~54 GB | ~13.5 GB | ~67.5 GB |

| GPU | VRAM | Max Model (FP16) | Max Context (32K) |
|-----|:----:|:----------------:|:-----------------:|
| RTX 3090 | 24 GB | ~8B | 32K |
| RTX 4090 | 24 GB | ~8B | 32K |
| A100 40GB | 40 GB | ~15B | 32K |
| A100 80GB | 80 GB | ~30B | 32K |
| H100 80GB | 80 GB | ~30B | 32K |

**The gap**: A 70B model needs 140 GB of VRAM, but even the most common data center GPUs offer 80 GB. Offloading bridges this gap.

---

## Architecture

### Offload Design Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     dLLM GPU ↔ CPU Offload Engine                       │
│                                                                          │
│  ┌─────────────────────┐          ┌─────────────────────┐               │
│  │    GPU (VRAM)       │  PCIe    │    CPU (RAM)        │               │
│  │                     │  Bus     │                     │               │
│  │  ┌───────────────┐  │ 48 GB/s  │  ┌───────────────┐  │               │
│  │  │ Active Layer  │  │◄────────►│  │ Model Weights │  │               │
│  │  │ (Compute)     │  │          │  │ (Full Model)  │  │               │
│  │  └───────────────┘  │          │  └───────────────┘  │               │
│  │                     │          │                     │               │
│  │  ┌───────────────┐  │          │  ┌───────────────┐  │               │
│  │  │ KV Cache      │  │          │  │ Offloaded KV  │  │               │
│  │  │ (Hot Set)     │  │          │  │ Cache         │  │               │
│  │  │ - Recent      │  │          │  │ (Cold Set)    │  │               │
│  │  │ - Active Heads│  │          │  │               │  │               │
│  │  └───────────────┘  │          │  └───────────────┘  │               │
│  │                     │          │                     │               │
│  │  ┌───────────────┐  │          │  ┌───────────────┐  │               │
│  │  │ Compute Buffer│  │          │  │ Compute Buffer│  │               │
│  │  │ (Working Set) │  │          │  │ (Working Set) │  │               │
│  │  └───────────────┘  │          │  └───────────────┘  │               │
│  └─────────────────────┘          └─────────────────────┘               │
│         ▲                                    ▲                          │
│         │                                    │                          │
│         └────────── Offload Manager ─────────┘                          │
│                    (Layer scheduling,                                    │
│                     memory tracking,                                     │
│                     transfer overlap)                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Location | Role |
|-----------|----------|------|
| **Offload Manager** | CPU RAM | Schedules which layers to keep on GPU, orchestrates transfers |
| **Active Layer Buffer** | GPU VRAM | Currently computing layer's weights + activations |
| **KV Cache Hot Set** | GPU VRAM | Recently accessed KV entries for fast attention |
| **KV Cache Cold Set** | CPU RAM | Older KV entries, loaded on demand |
| **Model Weight Store** | CPU RAM | Full model weights when not all fit on GPU |
| **Transfer Engine** | PCIe Bus | Overlaps data movement with computation via DMA |

---

## Offload Strategies

### Strategy 1: Partial GPU Offload (Recommended)

Keep the most compute-intensive layers on GPU, offload the rest to CPU:

```
Layer Distribution for Llama-7B on RTX 3090 (24 GB VRAM):

GPU VRAM (24 GB):
┌─────────────────────────────────────────────────────────────┐
│  Layer 0    │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 1    │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  ...                                                          │
│  Layer 7    │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 8    │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 9    │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 10   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 11   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 12   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 13   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 14   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 15   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 16   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 17   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 18   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 19   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 20   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 21   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 22   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 23   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 24   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 25   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 26   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 27   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 28   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 29   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 30   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  Layer 31   │  Weights: 0.5 GB  │  KV Cache: 0.1 GB        │
│  ─────────────────────────────────────────────────────────  │
│  Total GPU: 24 GB (16 GB weights + 4 GB KV + 4 GB overhead) │
└─────────────────────────────────────────────────────────────┘

CPU RAM (64 GB):
┌─────────────────────────────────────────────────────────────┐
│  Model Weights (offloaded): 16 GB                           │
│  KV Cache Cold Set: 16 GB                                   │
│  Operating System + Overhead: 32 GB                         │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**

```yaml
offload:
  strategy: partial_gpu
  # How many layers to keep fully on GPU
  gpu_layers: 32          # All layers on GPU (if VRAM allows)
  # Or specify by memory budget:
  gpu_memory_budget_gb: 24
  # KV cache split ratio (0.0 = all CPU, 1.0 = all GPU)
  kv_cache_gpu_ratio: 0.3  # 30% of KV cache on GPU
  
  # Transfer optimization
  transfer:
    overlap_compute: true   # Overlap PCIe transfers with compute
    async_streams: 4        # Number of async PCIe streams
    pinned_memory: true     # Use pinned host memory for faster PCIe
```

### Strategy 2: Full CPU Offload (CPU-Only Inference)

Run the entire model on CPU when no GPU is available or desired:

```yaml
offload:
  strategy: full_cpu
  
  # Model weights in CPU RAM
  model_location: cpu
  
  # KV cache in CPU RAM
  kv_cache_location: cpu
  
  # SIMD optimization level
  simd:
    auto_detect: true       # Detect AVX2/AVX-512 automatically
    force_level: null       # Override: sse42, avx, avx2, avx512
    
  # Memory optimization
  memory:
    pinned_host_memory: true
    numa_aware: true        # Allocate memory on correct NUMA node
```

### Strategy 3: Dynamic Offload (Adaptive)

Automatically adjusts GPU/CPU split based on workload:

```yaml
offload:
  strategy: dynamic
  
  # Baseline configuration
  gpu_layers_min: 8          # Minimum layers on GPU
  gpu_layers_max: 32         # Maximum layers on GPU
  
  # Adaptive triggers
  adaptation:
    enabled: true
    # Increase GPU layers when batch size grows
    batch_threshold: 16
    # Decrease GPU layers when VRAM pressure is high
    vram_pressure_threshold: 0.85
    # Rebalance interval
    rebalance_interval_ms: 1000
    
  # Monitoring
  metrics:
    track_vram_usage: true
    track_transfer_latency: true
    track_compute_utilization: true
```

---

## Model Weight Offloading

### Layer-by-Layer Migration

dLLM uses a **layer-pipelined offload** approach where weights are moved layer by layer:

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer-by-Layer Weight Migration                                 │
│                                                                  │
│  Step 1: Load Layer 0 weights from CPU RAM to GPU VRAM          │
│  GPU: [Layer 0 weights]                                         │
│  CPU: [Layer 1..31 weights]                                     │
│                                                                  │
│  Step 2: Compute Layer 0, then migrate to Layer 1               │
│  GPU: [Layer 1 weights]                                         │
│  CPU: [Layer 0 weights, Layer 2..31 weights]                    │
│                                                                  │
│  Step 3: Compute Layer 1, then migrate to Layer 2               │
│  GPU: [Layer 2 weights]                                         │
│  CPU: [Layer 0..1 weights, Layer 3..31 weights]                 │
│                                                                  │
│  ...                                                            │
│                                                                  │
│  Step N: Compute Layer 31, migrate back to Layer 0              │
│  GPU: [Layer 0 weights] (ready for next sequence)               │
│  CPU: [Layer 1..31 weights]                                     │
└──────────────────────────────────────────────────────────────────┘
```

### Weight Quantization for Offload

When offloading to CPU, quantization reduces memory bandwidth requirements:

| Quantization | Weight Size (7B model) | PCIe Transfer Time (32 layers) | Quality Impact |
|-------------|:----------------------:|:------------------------------:|:--------------:|
| FP16 | 14 GB | ~290 ms | None |
| BF16 | 14 GB | ~290 ms | None |
| INT8 | 7 GB | ~145 ms | <0.5% perplexity |
| INT4 | 3.5 GB | ~73 ms | 1–2% perplexity |
| Q4_K | 3.8 GB | ~79 ms | <0.3% perplexity |

### Transfer Overlap Technique

dLLM overlaps PCIe data transfer with GPU computation using CUDA streams:

```python
class OverlappedTransfer:
    """Overlap PCIe transfer with GPU compute"""
    
    def __init__(self, num_streams=4):
        self.streams = [cuda.Stream() for _ in range(num_streams)]
        self.stream_index = 0
    
    def prefetch_next_layer(self, layer_idx, weights_cpu):
        """Start PCIe transfer for next layer while current layer computes"""
        stream = self.streams[self.stream_index]
        
        # Async PCIe copy: CPU RAM → GPU VRAM
        cuda.memcpy_htod_async(
            gpu_weights_buffer[layer_idx],
            weights_cpu[layer_idx],
            stream
        )
        
        self.stream_index = (self.stream_index + 1) % len(self.streams)
    
    def wait_for_transfer(self, layer_idx):
        """Wait for PCIe transfer to complete before computing"""
        stream = self.streams[layer_idx % len(self.streams)]
        stream.synchronize()  # Blocks until PCIe transfer done
    
    def compute_layer(self, layer_idx):
        """Compute a layer (GPU kernels run in parallel with next transfer)"""
        # GPU compute for layer
        self._run_layer_kernels(layer_idx)
        
        # While compute runs, prefetch next layer
        next_layer = (layer_idx + 1) % 32
        self.prefetch_next_layer(next_layer, self.weights_cpu)
```

```
Timeline: Overlapped Transfer vs Naive Transfer

Naive (no overlap):
┌────────┬────────┬────────┬────────┬────────┬────────┐
│ Transfer│ Compute│Transfer│ Compute│Transfer│ Compute│
│ L0      │ L0     │ L1     │ L1     │ L2     │ L2     │
└────────┴────────┴────────┴────────┴────────┴────────┘
Total: 6 units (T_transfer + T_compute per layer)

Overlapped:
┌────────┬────────┬────────┬────────┬────────┬────────┐
│Transfer│Compute │Transfer│Compute │Transfer│Compute │
│ L0     │ L0     │ L1     │ L1     │ L2     │ L2     │
│        │        │        │        │        │        │
│◄───►   │◄───►   │◄───►   │◄───►   │◄───►   │◄───►   │
│        │◄───►   │        │◄───►   │        │◄───►   │
│        │        │◄───►   │        │◄───►   │        │
└────────┴────────┴────────┴────────┴────────┴────────┘
Total: ~4 units (max(T_transfer, T_compute) per layer)

Speedup: ~1.5× for typical PCIe 4.0 systems
```

---

## KV Cache Offloading

### GPU KV Cache Management

The KV cache is the fastest-growing memory consumer during inference. dLLM manages it with a tiered approach:

```
┌──────────────────────────────────────────────────────────────┐
│  KV Cache Tiered Management                                  │
│                                                              │
│  Tier 1: GPU VRAM (Hot KV Cache)                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  - Last N tokens (configurable, default: 8K)           │  │
│  │  - Active attention heads only                         │  │
│  │  - FP16 precision (no compression)                     │  │
│  │  - Access latency: <0.01 ms                           │  │
│  └────────────────────────────────────────────────────────┘  │
│           │                                                  │
│           ▼ Eviction (when GPU KV cache is full)             │
│           │                                                  │
│  Tier 2: CPU RAM (Cold KV Cache)                             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  - Older tokens evicted from GPU                       │  │
│  │  - INT8 compressed (2× smaller)                        │  │
│  │  - Access latency: ~0.5 ms (PCIe)                      │  │
│  │  - Prefetched before attention needs them              │  │
│  └────────────────────────────────────────────────────────┘  │
│           │                                                  │
│           ▼ Promote (when tokens become active again)        │
│           │                                                  │
│  Back to Tier 1: GPU VRAM                                   │
└──────────────────────────────────────────────────────────────┘
```

### KV Cache Offload Algorithm

```python
class KVCacheOffloader:
    """Manages KV cache between GPU and CPU"""
    
    def __init__(self, config):
        self.gpu_cache = GPUKVCache(max_tokens=config.gpu_max_tokens)
        self.cpu_cache = CPUKVCache(max_tokens=config.cpu_max_tokens)
        self.prefetcher = KVCachePrefetcher()
    
    def attention_compute(self, layer, query, token_idx, seq_len):
        """Compute attention with KV cache offloading"""
        
        # 1. Ensure needed KV entries are on GPU
        self._ensure_kv_on_gpu(layer, token_idx, seq_len)
        
        # 2. Compute attention on GPU
        attn_output = self._compute_attention_gpu(layer, query)
        
        # 3. Prefetch next layer's KV entries
        self.prefetcher.prefetch_next_layer(layer + 1, token_idx, seq_len)
        
        return attn_output
    
    def _ensure_kv_on_gpu(self, layer, token_idx, seq_len):
        """Move needed KV entries from CPU to GPU"""
        
        # Determine which KV entries are needed
        needed_tokens = range(max(0, token_idx - 1024), token_idx + 1)
        
        for t in needed_tokens:
            if not self.gpu_cache.contains(layer, t):
                # Check if on CPU
                if self.cpu_cache.contains(layer, t):
                    # Transfer from CPU to GPU
                    k_data = self.cpu_cache.get(layer, t)
                    v_data = self.cpu_cache.get(layer, t, is_v=True)
                    
                    # Compressed INT8 → FP16 on GPU
                    k_fp16 = self._decompress_int8_to_fp16(k_data)
                    v_fp16 = self._decompress_int8_to_fp16(v_data)
                    
                    self.gpu_cache.put(layer, t, k_fp16, v_fp16)
    
    def on_eviction(self, layer, token_idx, k_data, v_data):
        """Evict KV entries from GPU to CPU"""
        # Compress to INT8 for storage
        k_int8 = self._compress_fp16_to_int8(k_data)
        v_int8 = self._compress_fp16_to_int8(v_data)
        
        self.cpu_cache.put(layer, token_idx, k_int8, v_int8)
```

### KV Cache Eviction Policies

| Policy | Description | Best For |
|--------|-------------|----------|
| **LRU** | Evict least recently used entries | General purpose |
| **LFU** | Evict least frequently used entries | Repeated prefixes |
| **Attention-Aware** | Keep entries with high attention scores | Long-range dependencies |
| **Hybrid** | LRU + attention score threshold | Best of both |

```yaml
kv_cache_offload:
  gpu_max_tokens: 8192           # Max tokens in GPU KV cache
  cpu_max_tokens: 524288         # Max tokens in CPU KV cache (512K)
  
  eviction:
    policy: hybrid               # lru, lfu, attention_aware, hybrid
    attention_threshold: 0.01    # Never evict tokens with attention > threshold
    
  compression:
    cpu_format: int8             # INT8 on CPU, FP16 on GPU
    per_channel_scale: true      # Per-channel quantization scales
    
  prefetch:
    enabled: true
    tokens_ahead: 2048           # Prefetch 2K tokens ahead
    layers_ahead: 4              # Prefetch 4 layers ahead
```

---

## Performance Characteristics

### Model Weight Offload Performance

| Configuration | Llama-7B Latency | Speed vs Full GPU |
|--------------|:----------------:|:-----------------:|
| Full GPU (24 GB) | 45 ms/token | 100% |
| Partial offload (16 layers GPU) | 52 ms/token | 87% |
| Partial offload (8 layers GPU) | 68 ms/token | 66% |
| Full CPU offload | 120 ms/token | 38% |
| Full CPU + AVX-512 | 95 ms/token | 47% |

### KV Cache Offload Performance

| Context Size | GPU-Only KV | Offloaded KV | Overhead |
|-------------|:-----------:|:------------:|:--------:|
| 32K tokens | 45 ms/token | 46 ms/token | 2% |
| 64K tokens | OOM | 55 ms/token | 22% |
| 128K tokens | OOM | 72 ms/token | 60% |
| 256K tokens | OOM | 98 ms/token | 118% |

### PCIe Bandwidth Utilization

```
Llama-7B, 32 layers, FP16 weights, PCIe 4.0 (64 GB/s):

Per-layer weight transfer:
  Weight size: 0.5 GB (per layer)
  Transfer time: 0.5 GB / 64 GB/s = 7.8 ms

Per-layer compute time (A100):
  FLOPs: ~15 GFLOPs
  Compute time: 15 GFLOPs / 19.5 TFLOPs = 0.77 ms

Overlap ratio:
  Transfer: 7.8 ms
  Compute: 0.77 ms
  → Transfer is 10× slower than compute
  → Need aggressive prefetching to hide transfer latency

Effective speedup with overlap:
  Without overlap: 7.8 + 0.77 = 8.57 ms per layer
  With overlap: max(7.8, 0.77) = 7.8 ms per layer
  Speedup: 1.1× (modest, because transfer dominates)
```

---

## Configuration Reference

### Complete Offload Configuration

```yaml
gpu_cpu_offload:
  # Offload strategy
  strategy: partial_gpu          # partial_gpu, full_cpu, dynamic
  
  # GPU configuration
  gpu:
    device_id: 0                 # GPU device index
    memory_fraction: 0.85        # Use 85% of available VRAM
    compute_capability: auto     # Auto-detect or specify (e.g., "8.0")
    
  # Model weight offload
  weights:
    location: auto               # auto, gpu, cpu
    quantization: bf16           # fp16, bf16, int8, int4, q4_k
    pinned_memory: true          # Use pinned host memory for PCIe
    num_transfer_streams: 4      # Async PCIe transfer streams
    
  # KV cache offload
  kv_cache:
    location: auto               # auto, gpu, cpu
    gpu_max_tokens: 8192         # Max tokens in GPU KV cache
    cpu_max_tokens: 524288       # Max tokens in CPU KV cache
    eviction_policy: hybrid      # lru, lfu, attention_aware, hybrid
    compression: int8            # Compression format for CPU storage
    prefetch_tokens: 2048        # Tokens to prefetch ahead
    prefetch_layers: 4           # Layers to prefetch ahead
    
  # Dynamic offload (strategy: dynamic)
  dynamic:
    enabled: true
    gpu_layers_min: 8
    gpu_layers_max: 32
    rebalance_interval_ms: 1000
    vram_pressure_threshold: 0.85
    
  # Performance tuning
  performance:
    overlap_compute_transfer: true
    numa_aware: true
    cpu_affinity: auto           # Pin threads to NUMA nodes
```

---

## Use Cases

### Use Case 1: Running 70B Models on Consumer GPUs

```
Scenario: Run Llama-3.1-70B (140 GB weights) on RTX 4090 (24 GB VRAM)

Configuration:
  - GPU: RTX 4090 (24 GB VRAM)
  - CPU: 128 GB RAM
  - Strategy: partial_gpu with INT8 weight quantization

Result:
  - Quantized weights: 70 GB (INT8)
  - GPU holds: 12 layers (12 GB weights + 4 GB KV + 4 GB overhead)
  - CPU holds: 20 layers (28 GB weights) + KV cache (30 GB)
  - Latency: ~180 ms/token (vs ~85 ms/token on A100 80GB)
  - Cost: $1,600 GPU vs $15,000+ for multi-GPU A100 setup
```

### Use Case 2: Multi-Model Serving

```
Scenario: Serve 3 models on a single A100 80GB

Without offload:
  - Each model needs ~20 GB VRAM (7B FP16)
  - 3 × 20 GB = 60 GB → fits, but no room for KV cache
  
With offload:
  - Model weights on CPU (60 GB total)
  - Only active model's layer on GPU at a time
  - KV cache for all 3 models on CPU
  - Switch overhead: ~50 ms per model switch
  - Throughput: 3× more models served per GPU
```

### Use Case 3: Long Context on Limited Hardware

```
Scenario: 256K context on RTX 3090 (24 GB VRAM)

Without offload:
  - KV cache for 256K tokens: ~64 GB → OOM
  
With KV cache offload:
  - Hot KV cache (last 8K tokens): 2 GB on GPU
  - Cold KV cache (older 248K tokens): 60 GB on CPU (INT8)
  - Total memory: 62 GB (fits in 64 GB RAM)
  - Latency: ~98 ms/token (vs ~45 ms/token for 32K on same GPU)
  - Throughput: ~10 tok/s sustained
```

---

## Comparison: Offload vs No-Offload

| Scenario | No Offload | GPU Offload | CPU Offload |
|----------|:----------:|:-----------:|:-----------:|
| 7B model, 32K context, 24 GB GPU | ✅ Fits | ✅ Full speed | ⚠️ 87% speed |
| 70B model, 32K context, 24 GB GPU | ❌ OOM | ⚠️ 66% speed | ⚠️ 38% speed |
| 7B model, 256K context, 24 GB GPU | ❌ OOM | ⚠️ 60% overhead | ⚠️ 118% overhead |
| 7B model, 32K context, 8 GB GPU | ❌ OOM | ⚠️ 50% speed | ⚠️ 30% speed |

---

## Best Practices

1. **Use partial offload** — Keep as many layers on GPU as VRAM allows
2. **Enable transfer overlap** — Overlap PCIe transfers with compute for 10–30% improvement
3. **Use pinned memory** — Pinned host memory is 2–3× faster for PCIe transfers
4. **Quantize offloaded weights** — INT8 quantization halves PCIe transfer time with minimal quality loss
5. **Set NUMA affinity** — Pin threads to the correct NUMA node to avoid cross-node memory access
6. **Monitor VRAM pressure** — Trigger offload rebalancing before OOM occurs
7. **Prefetch aggressively** — The bottleneck is always PCIe transfer, not compute

---

## See Also

- [GPU Hardware Support](GPU_HARDWARE_SUPPORT.md) — Supported GPU vendors and backends
- [Network KV Cache Offloading](NETWORK_KV_CACHE_OFFLOADING.md) — Offload KV cache to remote nodes
- [PV Cache Optimization](PV_CACHE_OPTIMIZATION.md) — In-node KV cache compression
- [Model Formats](MODEL_FORMATS.md) — Supported model weight formats
- [Performance](PERFORMANCE.md) — Full performance benchmarks
