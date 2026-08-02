# PV Cache Examples

This directory contains examples for using the Prefix Vector (PV) cache optimization in dLLM.

## Quick Start Examples

### Python Example: Basic PV Cache Usage

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Enable PV cache with default settings
response = client.chat.completions.create(
    model="llama-7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    extra_body={
        "pv_cache": {
            "enabled": True,
            "prefix_length": 4096
        }
    }
)

print(f"Response: {response.choices[0].message.content}")
```

### Python Example: Long Context Chat

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Build a long conversation history (50K+ tokens)
messages = [
    {"role": "system", "content": "You are an expert assistant with access to extensive documentation."}
]

# Add many turns of conversation
for i in range(100):
    messages.append({
        "role": "user",
        "content": f"Question {i}: What is the capital of France? " + ("More context " * 50)
    })
    messages.append({
        "role": "assistant",
        "content": f"Answer {i}: The capital of France is Paris. " + ("Additional details " * 50)
    })

# Use PV cache to handle the long context efficiently
response = client.chat.completions.create(
    model="llama-7b",
    messages=messages,
    extra_body={
        "pv_cache": {
            "enabled": True,
            "prefix_length": 8192,
            "quantization": "int8"
        }
    }
)

print(f"Response: {response.choices[0].message.content}")
```

### Python Example: Document Processing

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Load a large document (100K+ tokens)
with open("large_document.txt", "r") as f:
    document_content = f.read()

# Process the document with PV cache
response = client.chat.completions.create(
    model="llama-7b",
    messages=[
        {"role": "system", "content": "You are a document analysis assistant."},
        {"role": "user", "content": f"Please summarize this document:\n\n{document_content}"}
    ],
    extra_body={
        "pv_cache": {
            "enabled": True,
            "prefix_length": 8192,
            "quantization": "bf16",
            "approximate_match": False
        }
    }
)

print(f"Summary: {response.choices[0].message.content}")
```

### Python Example: Code Generation with Large Context

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Load a large codebase context
with open("large_codebase_context.txt", "r") as f:
    code_context = f.read()

# Generate code with full project context
response = client.chat.completions.create(
    model="llama-7b",
    messages=[
        {"role": "system", "content": "You are an expert software developer."},
        {"role": "user", "content": f"""I have the following codebase context:

{code_context}

Please help me add a new feature that follows the existing patterns."""}
    ],
    extra_body={
        "pv_cache": {
            "enabled": True,
            "prefix_length": 4096,
            "approximate_match": True
        }
    }
)

print(f"Generated code:\n{response.choices[0].message.content}")
```

## C++ Examples

### Basic PV Cache Usage

```cpp
#include <iostream>
#include <vector>
#include "pv_cache/pv_cache.h"

int main() {
    // Create cache instance with default settings
    dllm::PVCache cache({
        .max_prefix_length = 8192,
        .quantization = dllm::Quantization::BF16
    });

    // Tokenize input (simplified)
    std::vector<int> tokens = {1, 2, 3, 4, 5};  // Example tokens

    // Compute prefix hash
    std::string hash = cache.computeHash(tokens);

    // Lookup prefix in cache
    auto result = cache.lookup(hash);
    
    if (result.has_value()) {
        auto [k_cache, v_cache] = result.value();
        std::cout << "Cache hit! Using cached values." << std::endl;
        
        // Use cached K and V values for attention computation
        // ...
    } else {
        std::cout << "Cache miss. Computing fresh values." << std::endl;
        
        // Compute fresh K and V values
        auto [k_values, v_values] = compute_attention_values(tokens);
        
        // Store in cache
        cache.insert(hash, k_values, v_values);
    }

    return 0;
}
```

### Distributed PV Cache Usage

```cpp
#include <iostream>
#include "pv_cache/distributed_pv_cache.h"

int main() {
    // Create distributed cache with multiple nodes
    dllm::DistributedPVCache distributed_cache({
        .nodes = {"node1:8001", "node2:8001", "node3:8001"},
        .replication_factor = 2,
        .consistency_level = dllm::ConsistencyLevel::Quorum
    });

    // Batch lookup for efficiency
    std::vector<std::string> hashes = {
        cache.computeHash(tokens1),
        cache.computeHash(tokens2),
        cache.computeHash(tokens3)
    };

    auto results = distributed_cache.batchLookup(hashes);

    for (const auto& [hash, result] : results) {
        if (result.found) {
            std::cout << "Found prefix in cache" << std::endl;
            // Use cached values
        } else {
            std::cout << "Cache miss for hash: " << hash << std::endl;
        }
    }

    return 0;
}
```

### Vector Storage Example

```cpp
#include <iostream>
#include <vector>
#include "pv_cache/vector_storage.h"

int main() {
    // Create compressed vector storage
    dllm::VectorStorage storage({
        .quantization = dllm::Quantization::INT8,
        .max_size_gb = 32
    });

    // Store K and V values
    std::vector<float> k_values(1024);  // Example K cache
    std::vector<float> v_values(1024);  // Example V cache

    // Initialize with some values
    for (int i = 0; i < 1024; i++) {
        k_values[i] = static_cast<float>(i) / 1024.0f;
        v_values[i] = std::sin(static_cast<float>(i) / 128.0f);
    }

    // Store in compressed format
    int vector_id = storage.store(k_values, v_values);
    std::cout << "Stored vector with ID: " << vector_id << std::endl;

    // Retrieve and decompress
    std::vector<float> k_retrieved, v_retrieved;
    storage.retrieve(vector_id, k_retrieved, v_retrieved);

    std::cout << "Retrieved " << k_retrieved.size() << " K values" << std::endl;

    return 0;
}
```

## Command Line Examples

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

### Export/Import Cache

```bash
# Export cache to file
dllm pv-cache export --output /path/to/cache.bin

# Import cache from file
dllm pv-cache import --input /path/to/cache.bin
```

## Advanced Examples

### Custom Eviction Policy

```python
from openai import OpenAI
import time

client = OpenAI(base_url="http://localhost:8000/v1")

# Monitor cache hit rate and adjust settings dynamically
def adaptive_cache_settings():
    # Get current statistics
    stats_response = client.get("/v1/pv_cache/stats")
    stats = stats_response.json()
    
    if stats['hit_rate'] < 0.5:
        # Low hit rate - increase prefix length
        return {"prefix_length": 8192, "quantization": "bf16"}
    elif stats['hit_rate'] > 0.8 and stats['memory_usage_bytes'] > 16 * 1024**3:
        # High hit rate but high memory - use more compression
        return {"prefix_length": 4096, "quantization": "int8"}
    else:
        # Balanced settings
        return {"prefix_length": 4096, "quantization": "int8"}

# Use adaptive settings in requests
settings = adaptive_cache_settings()
response = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={"pv_cache": {"enabled": True, **settings}}
)
```

### Batch Processing with PV Cache

```python
from openai import OpenAI
import concurrent.futures

client = OpenAI(base_url="http://localhost:8000/v1")

def process_request(prompt):
    """Process a single request with PV cache"""
    response = client.chat.completions.create(
        model="llama-7b",
        messages=[{"role": "user", "content": prompt}],
        extra_body={
            "pv_cache": {
                "enabled": True,
                "batch_lookup": True
            }
        }
    )
    return response.choices[0].message.content

# Process multiple requests in parallel
prompts = [
    "What is the capital of France?",
    "Explain quantum computing.",
    "Write a Python function to sort a list."
]

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_request, prompts))

for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt}")
    print(f"Result: {result}\n")
```

### Streaming with PV Cache

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Enable streaming with PV cache
response = client.chat.completions.create(
    model="llama-7b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
    extra_body={
        "pv_cache": {
            "enabled": True,
            "prefix_length": 2048
        }
    }
)

print("Streaming response:")
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Performance Optimization Examples

### Prefetching Strategy

```python
from openai import OpenAI
import time

client = OpenAI(base_url="http://localhost:8000/v1")

class PVCachePrefetcher:
    def __init__(self, client):
        self.client = client
        self.predicted_prefixes = []
    
    def predict_next_prefix(self, current_context):
        """Predict likely next prefix based on context"""
        # Simple prediction: use similar prefixes from cache
        return self._get_similar_prefixes(current_context)
    
    def prefetch(self, prefixes):
        """Prefetch prefixes into local cache"""
        for prefix in prefixes:
            if not self._has_in_cache(prefix):
                self.client.post("/v1/pv_cache/prefetch", json={"hash": prefix})
    
    def process_request(self, context):
        """Process request with prefetching"""
        # Predict and prefetch likely needed prefixes
        predictions = self.predict_next_prefix(context)
        self.prefetch(predictions[:3])  # Prefetch top 3
        
        # Process the actual request
        response = self.client.chat.completions.create(
            model="llama-7b",
            messages=[{"role": "user", "content": context}],
            extra_body={"pv_cache": {"enabled": True}}
        )
        
        return response

# Usage
prefetcher = PVCachePrefetcher(client)
response = prefetcher.process_request("Your prompt here")
```

### Memory-Efficient Large Context Processing

```python
from openai import OpenAI
import math

client = OpenAI(base_url="http://localhost:8000/v1")

def process_large_context(context, max_chunk_size=4096):
    """Process large context in chunks with PV cache"""
    
    # Split context into chunks
    tokens = tokenize(context)
    num_chunks = math.ceil(len(tokens) / max_chunk_size)
    
    all_results = []
    
    for i in range(num_chunks):
        start_idx = i * max_chunk_size
        end_idx = min((i + 1) * max_chunk_size, len(tokens))
        
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_context = detokenize(chunk_tokens)
        
        # Process chunk with PV cache
        response = client.chat.completions.create(
            model="llama-7b",
            messages=[{"role": "user", "content": chunk_context}],
            extra_body={
                "pv_cache": {
                    "enabled": True,
                    "prefix_length": max_chunk_size,
                    "quantization": "int8"  # More compression for large contexts
                }
            }
        )
        
        all_results.append(response.choices[0].message.content)
    
    # Combine results (simplified - in practice, use a more sophisticated merging strategy)
    return "\n\n".join(all_results)

# Usage
large_context = load_large_document()
result = process_large_context(large_context)
```

## Best Practices Examples

### 1. Prefix Length Selection

```python
def select_prefix_length(use_case):
    """Select optimal prefix length based on use case"""
    lengths = {
        "chat_short": 256,
        "chat_long": 4096,
        "document": 8192,
        "code_generation": 2048,
        "research": 4096
    }
    return {"prefix_length": lengths.get(use_case, 1024)}
```

### 2. Quantization Strategy

```python
def select_quantization(accuracy_required):
    """Select quantization based on accuracy requirements"""
    if accuracy_required == "high":
        return {"quantization": "bf16"}
    elif accuracy_required == "balanced":
        return {"quantization": "int8"}
    else:  # cost_sensitive
        return {"quantization": "int4"}
```

### 3. Cache Sizing

```python
def estimate_cache_size(context_length, count, quantization="int8"):
    """Estimate cache size needed"""
    bytes_per_token = {
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    # Formula: count × context_length × 2 (K and V) × bytes_per_token / compression_ratio
    compression = {"fp16": 1, "bf16": 1, "int8": 2, "int4": 4}
    
    size_bytes = (count * context_length * 2 * 
                  bytes_per_token[quantization] / 
                  compression[quantization])
    
    return size_bytes / (1024**3)  # Convert to GB
```

## Troubleshooting Examples

### Low Hit Rate Diagnosis

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Check cache statistics
stats = client.get("/v1/pv_cache/stats").json()
print(f"Hit rate: {stats['hit_rate']:.2%}")

if stats['hit_rate'] < 0.5:
    print("Low hit rate detected. Consider:")
    print("1. Increasing prefix_length")
    print("2. Enabling approximate_match")
    print("3. Checking for tokenization variations")
```

### Memory Pressure Handling

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")

# Monitor memory usage
stats = client.get("/v1/pv_cache/stats").json()
memory_gb = stats['memory_usage_bytes'] / (1024**3)

if memory_gb > 24:
    print("High memory usage. Consider:")
    print("1. Reducing max_cache_size_gb")
    print("2. Using more aggressive quantization (int8 or int4)")
    print("3. Enabling LRU eviction policy")
```
