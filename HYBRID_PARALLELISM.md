# Hybrid Parallelism in dLLM

## Overview

Hybrid parallelism combines tensor parallelism and pipeline parallelism to achieve optimal distribution for large models. In the two-tier architecture:

- **C++ Backend**: Manages layer assignment and communication patterns
- **Python Frontend**: Handles hybrid configuration via OpenAI-compatible API

This strategy splits workloads both within stages (tensor) and across stages (pipeline).

## Architecture

```
                    ┌────────────── Pipeline Stage 1 ──────────────┐
                    │                                             │
          Tensor    │  Node 1     Node 2      Node 3      Node 4   │
        Parallel    │  [W1/4]     [W1/4]      [W1/4]      [W1/4]   │
        (within)    │                                             │
                    └────────────── Pipeline Stage 2 ──────────────┘
                    │                                             │
          Tensor    │  Node 5     Node 6      Node 7      Node 8   │
        Parallel    │  [W2/4]     [W2/4]      [W2/4]      [W2/4]   │
        (within)    │                                             │
                    └────────────── Pipeline Stage 3 ──────────────┘

Legend:
- Each stage has tensor-parallel nodes
- Different layers on different stages
```

## Python Interface

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Use hybrid parallelism via config
response = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "tensor_parallel_degree": 4,
        "pipeline_stages": 4
    }
)
```

## Tensor × Pipeline Decomposition

For a 16-node cluster:

| Strategy | Tensor Nodes | Pipeline Stages | Total |
|----------|--------------|-----------------|-------|
| Balanced | 4 | 4 | 16 |
| Tensor-heavy | 8 | 2 | 16 |
| Pipeline-heavy | 2 | 8 | 16 |

## Configuration

```yaml
hybrid_parallelism:
  enabled: true
  strategy: auto              # auto, manual, tensor_heavy, pipeline_heavy
  
  # Manual configuration
  tensor_parallel_degree: 4   # Nodes per pipeline stage
  pipeline_parallel_degree: 4 # Number of pipeline stages
  
  # Scheduling
  overlap_comm_compute: true
  adaptive_rebalancing: true
```

## Strategy Selection

### When to Use Each Strategy

| Model Size | Recommended Strategy |
|------------|---------------------|
| < 1B params | Tensor only (or local) |
| 1B - 7B params | Tensor parallel or hybrid (2×8) |
| 7B - 40B params | Hybrid (4×4 or 8×2) |
| > 40B params | Hybrid (8×4 or more) |

## Performance Characteristics

### Theoretical Speedup

For hybrid parallelism with T tensor nodes and P pipeline stages:

```
Speedup = (T × P) / (1 + (T-1)/T × CommFrac + (P-1)/P × PipeOverhead)
```

| Nodes | Tensor Only | Pipeline Only | Hybrid |
|-------|-------------|---------------|--------|
| 4     | 3.4x        | 3.0x          | 3.6x   |
| 8     | 6.8x        | 6.5x          | 7.2x   |
| 16    | 13.6x       | 14.0x         | 15.0x  |

## Use Cases

### Large Transformer Models

```
Llama-70B on 32-node cluster:
├── Tensor Parallel: 8 nodes per stage
├── Pipeline Parallel: 4 stages
└── Total: 32 nodes (8 × 4)
```

### Memory-Constrained Environments

```yaml
hybrid_parallelism:
  # Prioritize tensor parallel to reduce per-node memory
  strategy: tensor_heavy
  tensor_degree: 16
  pipeline_stages: 2
```

## Implementation Details

### Layer Assignment Algorithm

```cpp
struct HybridConfig {
    int tensor_degree;
    int pipeline_stages;
};

std::vector<std::vector<Layer*>> assign_layers_to_pipeline(
    const std::vector<Layer*>& all_layers,
    const HybridConfig& config) {
    
    int total_nodes = config.tensor_degree * config.pipeline_stages;
    int layers_per_stage = all_layers.size() / config.pipeline_stages;
    
    std::vector<std::vector<Layer*>> stages(config.pipeline_stages);
    
    for (int s = 0; s < config.pipeline_stages; s++) {
        int start_layer = s * layers_per_stage;
        int end_layer = start_layer + layers_per_stage;
        
        // Assign tensor-parallel nodes for this stage
        for (int t = 0; t < config.tensor_degree; t++) {
            stages[s].push_back(
                new TensorParallelLayer(
                    all_layers[start_layer:end_layer],
                    tensor_id = t,
                    total_tensors = config.tensor_degree
                )
            );
        }
    }
    
    return stages;
}
```

## Tokenization in Hybrid Parallelism

### End-to-End Flow with Rust Tokenizer

```
Client Request (text)
         ↓
Rust Tokenizer (AVX512, zero-copy) ← Node 0 (dedicated tokenizer)
         ↓
Token IDs (u32 array)
         ↓
     Broadcast to all nodes
    for KV cache creation
         ↓
Tensor Parallelism + Pipeline Parallelism
   [8 tensor nodes × 4 pipeline stages = 32 total]
```

### Distributed Tokenization Strategy

For hybrid parallelism with large clusters:

| Node Group | Role | Tokenizer Usage |
|------------|------|-----------------|
| Node 0 | Tokenizer node | Rust tokenizer processes all inputs |
| Nodes 1-8 | Tensor parallel group | Receive tokens, create KV caches |
| Pipeline stages | Layer processing | Generate output tokens |

### Performance Optimization

| Configuration | Latency (P50) | Throughput | Notes |
|---------------|---------------|------------|-------|
| Python tokenizer + hybrid | 3.2s | 4 req/s | Baseline |
| Rust tokenizer (SSE4.2) + hybrid | 2.8s | 6 req/s | 14% faster |
| Rust tokenizer (AVX-512) + hybrid | **2.2s** | **9 req/s** | **45% faster** |

### Best Practices

1. **Dedicated tokenizer node**: Use one node for all tokenization, broadcast to others
2. **Contiguous memory blocks**: Enables fast tensor transfer across nodes
3. **AVX-512 enables full hybrid parallelism**: No tokenizer bottleneck at scale
4. **Zero-copy FFI bridge**: Rust → C++ without serialization overhead

### Memory Efficiency

| Approach | Peak Memory (per node) | Notes |
|----------|----------------------|-------|
| Python + tensor split | 8GB | High allocation overhead |
| Rust + tensor split | 2.5GB | Zero-copy design |

### Tokenizer-Parallelism Integration Chart

```
32-node hybrid cluster:
├── Node 0: [Rust Tokenizer AVX-512] (tokenizer node)
│
├── Tensor Parallel Group A (8 nodes):
│   ├── Stage 1 (Node 1): Layers 1-4
│   ├── Stage 2 (Node 2): Layers 5-8  
│   ├── Stage 3 (Node 3): Layers 9-12
│   └── Stage 4 (Node 4): Layers 13-16
│
└── Tensor Parallel Group B (8 nodes):
    ├── Stage 5 (Node 5): Layers 17-20
    ├── Stage 6 (Node 6): Layers 21-24
    ├── Stage 7 (Node 7): Layers 25-28
    └── Stage 8 (Node 8): Layers 29-32

Token flow:
Text → Rust Tokenizer → u32 array → Broadcast → All nodes create KV caches

### Communication Pattern

```cpp
// Within pipeline stage: tensor parallel all-reduce
void within_stage_communication(std::vector<Tensor>& partial_results) {
    all_reduce(partial_results, op=sum);
}

// Between pipeline stages: send/receive
void between_stage_communication(Tensor& data, int next_stage_id) {
    if (is_last_node_in_stage()) {
        send_to_next_stage(data);
    }
}
```

## Troubleshooting

### Imbalanced Workload
```yaml
# Fix: Enable adaptive rebalancing
hybrid_parallelism:
  adaptive_rebalancing: true
  rebalance_interval: 100  # batches
```

### Communication Bottleneck
```yaml
# Fix: Prioritize tensor parallelism (less cross-stage comm)
hybrid_parallelism:
  strategy: tensor_heavy
  tensor_degree: 8
  pipeline_stages: 2
