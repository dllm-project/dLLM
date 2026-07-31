# Pipeline Parallelism in dLLM

## Overview

Pipeline parallelism distributes different model layers across multiple nodes, creating a computation pipeline where micro-batches flow through the stages sequentially.

In the two-tier architecture:
- **C++ Backend**: Manages layer distribution and stage coordination
- **Python Frontend**: Handles pipeline configuration via OpenAI-compatible API

## Architecture

```
Client Request → Python FastAPI → C++ Backend (Pipeline Coordinator)
                                      │
                                      ├─ Stage 1: [Layer 1-4] on Node 1
                                      ├─ Stage 2: [Layer 5-8] on Node 2
                                      ├─ Stage 3: [Layer 9-12] on Node 3
                                      └─ Stage 4: [Layer 13-16] on Node 4
```

## How It Works

### Basic Pipeline Structure
```
Time →
Node 1: [B1-L1] ──► [B2-L1] ──► [B3-L1] ──► [B4-L1]
                                       │
Node 2:             [B1-L2] ──► [B2-L2] ──► [B3-L2]
                                                  │
Node 3:                       [B1-L3] ──► [B2-L3] ──► [B3-L3]
                                                           │
Node 4:         [B1-L4] ──► [B2-L4] ──► [B3-L4] ──► [B4-L4]
```

### Micro-batching Strategy

Each batch is split into micro-batches that flow through the pipeline:

```
Batch 1: [μ1, μ2, μ3, μ4]

Stage Flow:
T+0:   Node 1 processes B1-μ1
T+Δt:  Node 2 processes B1-μ1 (Node 1 starts B1-μ2)
T+2Δt: Node 3 processes B1-μ1 (Node 2 starts B1-μ2, Node 1 starts B1-μ3)
```

## Configuration

```yaml
pipeline_parallelism:
  enabled: true
  num_stages: auto           # Automatically determine based on layers
  micro_batch_size: 8        # Number of samples per micro-batch
  interleaving: true         # Interleaved execution for better utilization
  
  # Pipeline schedule
  schedule: gpipe            # gpipe, pipe-fmff, 1f1b
  
  # Synchronization
  sync_interval: 10          # Sync every N forward passes
```

## Pipeline Schedule Types

### GPipe (Global Pipeline)
```
Forward Pass:
Node 1 → Node 2 → Node 3 → Node 4
All batches complete forward before backward.

Backward Pass:
Node 4 → Node 3 → Node 2 → Node 1
All batches complete backward.
```

**Pros**: Simple, no deadlocks
**Cons**: Higher memory usage

### 1F1B (One Forward One Backward)
```
Interleaved forward and backward passes through the pipeline.
```

**Pros**: Better memory efficiency
**Cons**: More complex scheduling

## Use Cases

### When to Use Pipeline Parallelism

| Scenario | Recommendation |
|----------|----------------|
| Deep models (many layers) | ✓ Pipeline parallel |
| Shallow models (<10 layers) | ✗ Not recommended |
| Limited bandwidth network | ✓ Reduces per-node comm |
| High-latency nodes | ✓ Staggered execution |

### Best Practices

1. **Balance layer compute**: Ensure similar time per stage
2. **Optimize micro-batch size**: Balance memory vs. pipeline fill
3. **Use interleaved scheduling**: Minimize bubble periods

## Performance Characteristics

| Micro-batches | Efficiency | Pipeline Fill |
|---------------|------------|---------------|
| 1             | 25%        | Poor          |
| 4             | 70%        | Good          |
| 8+            | 85%+       | Excellent     |

## Implementation Details

### Pipeline Stage Assignment

```cpp
struct PipelineStage {
    int stage_id;
    std::vector<Layer*> layers;
    int micro_batch_size;
    
    void forward(const Tensor& input) {
        for (auto& layer : layers) {
            input = layer->forward(input);
        }
        // Send to next stage or collect output
        if (is_last_stage()) {
            output_queue.push(input);
        } else {
            send_to_next_stage(input);
        }
    }
};
```

### Python Interface

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Use pipeline parallelism via config
response = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "pipeline_stages": 4,
        "micro_batch_size": 8
    }
)
```

### Gradient Accumulation

```cpp
void backward_pass() {
    for (int i = num_micro_batches - 1; i >= 0; i--) {
        // Receive gradient from next stage
        auto grad = receive_from_next_stage();
        
        // Backward through layers in reverse order
        for (auto& layer : reversed(layers)) {
            grad = layer->backward(grad);
        }
        
        // Accumulate gradients locally
        accumulate_gradients(grad);
    }
    
    // Apply gradients when all micro-batches processed
    optimizer.step();
}
```

## Tokenization Pipeline Integration

### End-to-End Request Flow with Rust Tokenizer

```
Client Request (text)
         ↓
Rust Tokenizer (AVX2/AVX512, zero-copy)
         ↓
Token IDs (u32 array) → Send to C++ Backend
                              ↓
                   Pipeline Stage 1: [Layers 1-4] on Node 1
                              ↓
                   Pipeline Stage 2: [Layers 5-8] on Node 2
                              ↓
                   Pipeline Stage 3: [Layers 9-12] on Node 3
                              ↓
                   Pipeline Stage 4: [Layers 13-16] on Node 4
                              ↓
                         Output tokens
```

### Tokenization as First Pipeline Stage

In high-throughput deployments, tokenization can be offloaded:

```
Node 0 (Tokenizer):    [Rust Tokenizer AVX512]
                            ↓
Nodes 1-4 (Inference): [Pipeline Stages 1-4]
```

This allows:
- Parallel text encoding while inference runs on previous batch
- Better GPU/CPU utilization balance
- Reduced end-to-end latency

### Batch Encoding Strategy

For pipeline parallelism with micro-batching:

| Step | Node 0 | Nodes 1-4 |
|------|--------|-----------|
| T+0  | Encode μ1 tokens | Wait for input |
| T+Δt | Encode μ2 tokens | Process μ1 through stage 1 |
| T+2Δt | Encode μ3 tokens | Process μ1 through stages 1-2 |
| T+3Δt | Encode μ4 tokens | Process μ1 through stages 1-3, μ2 through stage 1 |

### Performance Impact

With tokenizer integrated as pipeline stage:

| Configuration | Latency (P50) | Throughput |
|---------------|---------------|------------|
| Python tokenizer | 85ms | 12 req/s |
| Rust tokenizer (AVX2) | 65ms | 18 req/s |
| Rust tokenizer (AVX-512) | **48ms** | **25 req/s** |

### Best Practices

1. **Pre-encode batches**: Use Rust tokenizer to encode all batch texts before pipeline starts
2. **Contiguous memory**: Ensure token IDs are contiguous for efficient tensor creation
3. **Zero-copy FFI**: Pass raw pointers from Rust tokenizer to C++ engine
4. **AVX-512 clusters**: Enable SIMD in high-throughput distributed deployments

### Memory Efficiency Comparison

| Approach | Input Size | Peak Memory | Ratio |
|----------|-----------|-------------|-------|
| Python tokenizer + tensor | 10MB | 35MB | 3.5x |
| Rust tokenizer (SSE4.2) | 10MB | 12MB | 1.2x |
| Rust tokenizer (AVX-512) | 10MB | **11MB** | **1.1x** |

The Rust tokenizer's zero-copy design eliminates intermediate string allocations, reducing memory pressure during distributed inference.

## Troubleshooting

### Pipeline Bubble (Idle Time)
```yaml
# Fix: Increase micro-batch count for better fill
pipeline_parallelism:
  micro_batch_size: 16     # Larger micro-batches
  num_micro_batches: 32    # More concurrent micro-batches
```

### Memory Exhaustion
```yaml
# Fix: Reduce micro-batch size
pipeline_parallelism:
  micro_batch_size: 4      # Smaller micro-batches per node
  checkpoint_activations: true  # Recompute instead of store
