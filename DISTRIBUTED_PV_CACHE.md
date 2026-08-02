# Distributed PV Cache in dLLM

## Overview

The distributed PV cache extends prefix vector caching across multiple nodes in a dLLM cluster, enabling:

- **Cross-node prefix sharing**: Common prefixes cached once, used by all nodes
- **Scalable storage**: Distribute cache across cluster nodes
- **Fault tolerance**: Replicate critical prefixes for reliability
- **Load balancing**: Even distribution of cache load

## Architecture

### Distributed PV Cache Cluster

```
┌─────────────────────────────────────────────────────────────────────┐
│                    dLLM Distributed PV Cache                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐         │
│  │   Node 1     │     │   Node 2     │     │   Node N     │         │
│  │              │     │              │     │              │         │
│  │  PV Cache    │     │  PV Cache    │     │  PV Cache    │         │
│  │  ┌─────────┐ │     │  ┌─────────┐ │     │  ┌─────────┐ │         │
│  │  │ Hash    │ │     │  │ Hash    │ │     │  │ Hash    │ │         │
│  │  │ Table   │ │     │  │ Table   │ │     │  │ Table   │ │         │
│  │  └────┬────┘ │     │  └────┬────┘ │     │  └────┬────┘ │         │
│  │       │      │     │       │      │     │       │      │         │
│  │  ┌────▼────┐ │     │  ┌────▼────┐ │     │  ┌────▼────┐ │         │
│  │  │ Vector  │ │     │  │ Vector  │ │     │  │ Vector  │ │         │
│  │  │ Storage │ │     │  │ Storage │ │     │  │ Storage │ │         │
│  │  └────┬────┘ │     │  └────┬────┘ │     │  └────┬────┘ │         │
│  └───────┼──────┘     └───────┼──────┘     └───────┼──────┘         │
│          │                    │                    │                │
│          └────────────────────┴────────────────────┘                │
│                              │                                      │
│                      ┌───────▼───────┐                              │
│                      │  Consensus    │                              │
│                      │  Protocol     │                              │
│                      │  (Raft)       │                              │
│                      └───────────────┘                              │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

1. **Local PV Cache**: Per-node prefix storage with hash table + vector storage
2. **Distributed Hash Table (DHT)**: Consistent hashing for prefix distribution
3. **Replication Layer**: Cross-node replication for fault tolerance
4. **Consensus Protocol**: Raft-based coordination for cache coherence

## Distributed Hash Table

### Hash Ring Algorithm

```python
class DistributedHashTable:
    def __init__(self, nodes, replicas=150):
        self.ring = {}
        self.sorted_keys = []
        self.replicas = replicas
        
        for node in nodes:
            self.add_node(node)
    
    def add_node(self, node):
        """Add node to hash ring"""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            self.ring[key] = node
            self.sorted_keys.append(key)
        
        self.sorted_keys.sort()
    
    def remove_node(self, node):
        """Remove node from hash ring"""
        for i in range(self.replicas):
            key = self._hash(f"{node}:{i}")
            if key in self.ring:
                del self.ring[key]
                self.sorted_keys.remove(key)
    
    def get_node(self, key):
        """Get node responsible for key"""
        if not self.sorted_keys:
            return None
        
        hash_val = self._hash(key)
        
        for sorted_key in self.sorted_keys:
            if hash_val <= sorted_key:
                return self.ring[sorted_key]
        
        # Wrap around
        return self.ring[self.sorted_keys[0]]
    
    def _hash(self, key):
        """Compute hash value"""
        import hashlib
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
```

### Key Distribution

```python
# Example: Distribute prefixes across 3 nodes
nodes = ["node1", "node2", "node3"]
dht = DistributedHashTable(nodes)

prefixes = [
    "prefix_001",
    "prefix_002",
    "prefix_003",
    # ... more prefixes
]

for prefix in prefixes:
    node = dht.get_node(prefix)
    print(f"Prefix {prefix} -> Node {node}")
```

## Replication Strategy

### Replication Factor Configuration

```yaml
distributed_pv_cache:
  replication_factor: 2  # Number of nodes to replicate to
  
  # Replication modes
  replication_mode: async  # sync, async, lazy
  
  # Consistency levels
  consistency_level: quorum  # one, quorum, all
  
  # Failure handling
  failure_detection_timeout_ms: 5000
  recovery_timeout_ms: 30000
```

### Replication Protocol

```python
class ReplicationProtocol:
    def __init__(self, replication_factor):
        self.replication_factor = replication_factor
    
    def replicate(self, key, value, nodes):
        """Replicate to multiple nodes"""
        primary_node = nodes[0]
        
        # Write to primary
        primary_node.put(key, value)
        
        if self.replication_factor > 1:
            # Async replication to replicas
            for node in nodes[1:]:
                node.put_async(key, value)
        
        return True
    
    def read(self, key, consistency_level):
        """Read with consistency guarantee"""
        if consistency_level == "one":
            return self._read_from_nearest(key)
        elif consistency_level == "quorum":
            return self._read_with_quorum(key)
        elif consistency_level == "all":
            return self._read_from_all(key)
```

## Cache Coherence Protocol

### Coherence States

```python
class PVCacheCoherence:
    """Distributed cache coherence states"""
    
    # State machine for cache entries
    STATES = {
        'INVALID': 0,      # No valid data
        'SHARED': 1,       # Valid, may be on other nodes
        'EXCLUSIVE': 2,    # Valid, not on other nodes
        'MODIFIED': 3      # Modified, must replicate
    }
```

### Coherence Protocol Messages

```protobuf
// Cache coherence protocol messages
message PVCacheMessage {
    oneof message_type {
        LookupRequest lookup_request = 1;
        LookupResponse lookup_response = 2;
        InsertRequest insert_request = 3;
        InvalidateRequest invalidate_request = 4;
        ReplicationRequest replication_request = 5;
    }
    
    string prefix_hash = 6;
    int32 version = 7;
}
```

## Performance Optimization

### Batch Operations

```python
class DistributedPVCache:
    def __init__(self, nodes):
        self.nodes = nodes
        self.dht = DistributedHashTable(nodes)
    
    def batch_lookup(self, prefix_hashes):
        """Batch lookup across distributed cache"""
        # Group by node
        node_requests = defaultdict(list)
        
        for hash_val in prefix_hashes:
            node = self.dht.get_node(hash_val)
            node_requests[node].append(hash_val)
        
        # Parallel requests to each node
        results = {}
        for node, hashes in node_requests.items():
            node_results = node.batch_lookup(hashes)
            results.update(node_results)
        
        return results
    
    def batch_insert(self, entries):
        """Batch insert with replication"""
        # Group by node
        node_entries = defaultdict(list)
        
        for entry in entries:
            node = self.dht.get_node(entry.hash)
            node_entries[node].append(entry)
        
        # Parallel inserts with replication
        for node, entries_batch in node_entries.items():
            node.batch_insert_with_replication(entries_batch)
```

### Prefetching Strategy

```python
class Prefetcher:
    def __init__(self, cache, count=3):
        self.cache = cache
        self.count = count
    
    def prefetch(self, current_prefix):
        """Prefetch likely next prefixes"""
        # Get similar prefixes from cache
        similar = self.cache.get_similar(current_prefix, self.count)
        
        # Prefetch into local cache
        for prefix in similar:
            if not self.cache.local_has(prefix):
                self.cache.prefetch(prefix)
```

## Configuration

### Basic Distributed Setup

```yaml
distributed_pv_cache:
  enabled: true
  
  # Cluster configuration
  nodes:
    - host: node1.example.com
      port: 8001
    - host: node2.example.com
      port: 8001
    - host: node3.example.com
      port: 8001
  
  # Replication
  replication_factor: 2
  consistency_level: quorum
  
  # Performance tuning
  batch_size: 64
  prefetch_count: 5
```

### Advanced Configuration

```yaml
distributed_pv_cache:
  enabled: true
  
  # Hash ring configuration
  hash_ring:
    replicas_per_node: 150
    load_balancing: weighted  # uniform, weighted
    
  # Replication
  replication:
    mode: async
    batch_size: 32
    timeout_ms: 5000
    
  # Consensus
  consensus:
    protocol: raft
    election_timeout_ms: 1000
    heartbeat_interval_ms: 100
  
  # Monitoring
  monitoring:
    hit_rate_metrics: true
    replication_lag_metrics: true
    load_balance_metrics: true
```

## Fault Tolerance

### Node Failure Handling

```python
class FaultTolerantCache:
    def __init__(self, cache, replication_factor):
        self.cache = cache
        self.replication_factor = replication_factor
    
    def lookup(self, prefix_hash):
        """Lookup with fault tolerance"""
        # Try primary node
        try:
            return self.cache.lookup(prefix_hash)
        except NodeUnavailable:
            # Try replica nodes
            replicas = self.cache.get_replica_nodes(prefix_hash)
            
            for replica in replicas[1:]:
                try:
                    return self.cache.lookup_from(replica, prefix_hash)
                except:
                    continue
            
            raise CacheMiss()
```

### Automatic Recovery

```python
class CacheRecovery:
    def __init__(self, cache):
        self.cache = cache
    
    def on_node_failure(self, failed_node):
        """Handle node failure and recover data"""
        # Get all prefixes stored on failed node
        prefixes = self.cache.get_prefixes_on_node(failed_node)
        
        # Reassign to other nodes
        for prefix in prefixes:
            new_node = self.cache.dht.get_node(prefix)
            
            if new_node != failed_node:
                # Replicate to new node
                entry = self.cache.get_entry(prefix)
                self.cache.replicate_to(new_node, prefix, entry)
        
        # Update hash ring
        self.cache.dht.remove_node(failed_node)
```

## Monitoring and Metrics

### Key Metrics

```python
class PVCacheMetrics:
    def __init__(self):
        self.metrics = {
            'lookup_count': 0,
            'hit_count': 0,
            'miss_count': 0,
            'replication_count': 0,
            'memory_usage_bytes': 0,
            'network_latency_ms': [],
        }
    
    def record_lookup(self, hit):
        self.metrics['lookup_count'] += 1
        if hit:
            self.metrics['hit_count'] += 1
        else:
            self.metrics['miss_count'] += 1
    
    @property
    def hit_rate(self):
        total = self.metrics['lookup_count']
        if total == 0:
            return 0.0
        return self.metrics['hit_count'] / total
```

### Dashboard Metrics

```yaml
monitoring:
  metrics:
    - name: pv_cache_hit_rate
      type: gauge
      description: "PV cache hit rate per node"
    
    - name: pv_cache_memory_usage
      type: gauge
      description: "Memory usage by PV cache (bytes)"
    
    - name: pv_cache_replication_lag
      type: histogram
      description: "Replication latency (ms)"
    
    - name: pv_cache_lookup_latency
      type: histogram
      description: "Cache lookup latency (ms)"
```

## Best Practices

### 1. Node Sizing

```bash
# Calculate per-node cache capacity
# Formula: (total_cache_gb / num_nodes) × replication_factor

# Example for 32GB total, 4 nodes, replication=2:
# Per node: (32GB / 4) × 2 = 16GB usable per node
```

### 2. Replication Strategy

| Use Case | Replication Factor |
|----------|-------------------|
| Development | 1 (no replication) |
| Production (high availability) | 2-3 |
| Critical data | 3+ |

### 3. Consistency Level Selection

| Level | Latency | Consistency | Use Case |
|-------|---------|-------------|----------|
| one | Low | Weak | Read-heavy workloads |
| quorum | Medium | Strong | Balanced workloads |
| all | High | Strong | Critical data |

## Troubleshooting

### Common Issues

**Low Hit Rate**
- Check hash distribution uniformity
- Verify replication is working
- Increase cache size per node

**High Latency**
- Reduce replication factor
- Use async replication
- Enable batch operations

**Data Loss on Failure**
- Increase replication factor
- Use sync replication mode
- Implement proper failure detection

## Future Enhancements

### Planned Features

1. **Adaptive Replication**: Dynamic adjustment based on access patterns
2. **Geo-distributed Cache**: Cross-region cache sharing
3. **Tiered Storage**: SSD/HDD for less frequently accessed prefixes
4. **Machine Learning Prediction**: Predict prefix access patterns
