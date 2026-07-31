# Distribution Clustering in dLLM

## Overview

dLLM supports distributed inference across multiple CPU nodes, enabling scale-out for large language models. The two-tier architecture:

- **Python Frontend**: Handles cluster coordination and load balancing via OpenAI-compatible API
- **C++ Backend**: Performs compute-intensive inference with tensor/pipeline parallelism

The clustering system handles node discovery, load balancing, and fault tolerance automatically.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │      Python FastAPI Server          │
                    │  - Load Balancer                    │
                    │  - Cluster Coordinator              │
                    │  - Health Monitor                   │
                    └──────────────┬──────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
    +------------+          +------------+           +------------+
    |  Worker 1  |          |  Worker 2  |    ...    |  Worker N  |
    | (C++ Backend)|        | (C++ Backend)|         | (C++ Backend)|
    | - Inference|◄─────────┤ - Inference├───────────┤ - Inference │
    | - Tensor   |          | - Tensor   |           | - Tensor   │
    +------------+          +------------+           +------------+
```

## Cluster Setup

### Minimal 3-Node Cluster (via config.yaml)

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

backend:
  cpp_library_path: "./build/libdllm.so"

distribution:
  mode: distributed
  nodes:
    - host: node1.example.com
      rank: 0
    - host: node2.example.com
      rank: 1
    - host: node3.example.com
      rank: 2

parallelism:
  tensor_degree: auto
  pipeline_stages: auto
```

### Starting the Cluster

```bash
# On head node (node1)
python server.py --config config.yaml

# Access from any client
curl http://node1:8000/v1/models
```

## Cluster Management

### Node Discovery

Python-based cluster management:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Get cluster status via health endpoint
response = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "status"}],
    extra_body={
        "cluster_status": True
    }
)
```

### Health Monitoring

```yaml
health_check:
  enabled: true
  interval: 5s
  timeout: 3s
  failure_threshold: 3
  
recovery:
  automatic: true
  failover_timeout: 30s
```

## Load Balancing Strategies

### Algorithm Options

| Strategy | Description | Best For |
|----------|-------------|----------|
| least_loaded | Route to node with fewest pending requests | Heterogeneous clusters |
| round_robin | Round-robin through available nodes | Homogeneous clusters |
| predictive | Use historical data to predict best node | Variable workload |

### Configuration

```yaml
server:
  workers: 16
  
load_balancer:
  strategy: least_loaded
  
  # Rate limiting
  max_requests_per_node: 100
  queue_size_per_node: 50
```

## Fault Tolerance

### Failure Detection

```bash
# Monitor cluster health via API
curl http://localhost:8000/health

# Response:
# {"status": "healthy", "nodes_online": 3, "total_nodes": 3}
```

### Recovery Procedures

```yaml
recovery:
  automatic: true
  
  # Node failure handling
  failover_timeout: 30s
  replica_factor: 2
  
  # Data recovery
  tensor_replication: true
  cache_sync_interval: 1s
```

## Distributed Execution

### Tensor Distribution

```yaml
distribution:
  mode: hybrid
  strategy: tensor_pipeline
  
  # Tensor splitting
  tensor_parallel_degree: auto
  
  # Pipeline stages  
  pipeline_stages: auto
  
  # Overlap communication and computation
  overlap_comm_compute: true
```

### Request Routing

```
Client → Python FastAPI → Load Balancer → Node Selection → C++ Inference
                                         │
                                         ├─ Check node health
                                         ├─ Check available memory
                                         ├─ Check instruction set support
                                         └─ Select optimal node
```

## Performance Scaling

### Linear vs. Sub-linear Scaling

| Nodes | Expected Speedup | Actual (with overhead) |
|-------|-----------------|------------------------|
| 2     | 2.0x            | 1.7x                   |
| 4     | 4.0x            | 3.4x                   |
| 8     | 8.0x            | 6.8x                   |

### Scaling Limitations

1. **Communication Overhead**: All-reduce becomes bottleneck
2. **Load Imbalance**: Variance in layer computation time
3. **Network Latency**: Round-trip time between nodes

## Cluster Examples

### Small Model (GPT-2 Small)

```bash
# Single node is sufficient
python server.py --model /models/gpt2-small
```

### Large Model (Llama-70B)

```yaml
# 8-node cluster configuration
server:
  host: "0.0.0.0"
  port: 8000

backend:
  cpp_library_path: "./build/libdllm.so"

distribution:
  mode: distributed
  nodes:
    - host: host1.example.com
    - host: host2.example.com
    # ... up to host8
  
parallelism:
  tensor_degree: 4
  pipeline_stages: 2
```

## Monitoring and Metrics

### Key Metrics (via API)

```bash
# View cluster metrics via API
curl http://localhost:8000/v1/models

# Check server health
curl http://localhost:8000/health
```

### Logs and Debugging

```bash
# View Python server logs
python server.py --log-level trace

# C++ backend debug
export DLLM_LOG_LEVEL=trace
