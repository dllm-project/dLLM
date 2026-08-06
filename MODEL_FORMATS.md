# Model Format Support

dLLM supports multiple model weight formats for loading pre-trained language models. This document provides comprehensive details on supported formats, their specifications, and how to use them.

## Supported Formats

| Format | Extension | Description | Status |
|--------|-----------|-------------|--------|
| **GGUF** | `.gguf` | GGML Unified Format - quantized models with metadata | ✓ Supported |
| **Safetensors** | `.safetensors` | Safe tensor serialization format by Hugging Face | ✓ Supported |
| **Sharded Safetensors** | `*.index.json` | Multi-file safetensors with index | ✓ Supported |
| **PyTorch** | `.pt`, `.pth` | PyTorch native format | ✓ Supported |

## GGUF Format

GGUF (GGML Unified Format) is the primary format for dLLM, supporting quantized models with rich metadata.

### GGUF Specifications

| Property | Value |
|----------|-------|
| **Header** | Magic bytes `GGUF` (4 bytes) |
| **Version** | 3 (current) |
| **Endianness** | Little-endian |
| **Metadata** | Key-value pairs with type tags |
| **Tensors** | Binary blob with shape and data type |

### GGUF Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `general.architecture` | String | Model architecture (e.g., `llama`, `mistral`) |
| `general.name` | String | Model name |
| `general.file_type` | U32 | Quantization type identifier |
| `general.quantization_version` | U32 | Quantization version |
| `llama.context_length` | U32 | Maximum context length |
| `llama.embedding_length` | U32 | Hidden size |
| `llama.block_count` | U32 | Number of transformer layers |
| `llama.feed_forward_length` | U32 | FFN hidden size |
| `llama.attention.head_count` | U32 | Number of attention heads |
| `llama.attention.head_count_kv` | U32 | Number of KV heads |
| `llama.rope.freq_base` | F32 | RoPE base frequency |
| `llama.rope.dimension_count` | U32 | RoPE dimension |
| `tokenizer.ggml.tokens` | String array | Token vocabulary |
| `tokenizer.ggml.scores` | F32 array | Token scores |
| `tokenizer.ggml.token_type` | U32 array | Token types |

### GGUF Quantization Types

| Type | Bits/Weight | Description | Quality |
|------|-------------|-------------|---------|
| **Q2_K** | 2.56 | 2-bit quantization with K-quants | Low |
| **Q3_K** | 3.35 | 3-bit quantization with K-quants | Low-Medium |
| **Q4_0** | 4.5 | 4-bit quantization, original | Medium |
| **Q4_1** | 5.34 | 4-bit quantization with scaling | Medium |
| **Q4_K** | 4.5 | 4-bit K-quants (mixed precision) | Medium-High |
| **Q5_0** | 5.5 | 5-bit quantization | High |
| **Q5_1** | 6.34 | 5-bit quantization with scaling | High |
| **Q5_K** | 5.5 | 5-bit K-quants (mixed precision) | High |
| **Q6_K** | 6.58 | 6-bit quantization | Very High |
| **Q8_0** | 8.5 | 8-bit quantization | Near-lossless |
| **Q8_K** | 8.5 | 8-bit K-quants | Near-lossless |
| **F16** | 16 | Half precision float | Lossless |
| **F32** | 32 | Full precision float | Lossless |

### GGUF File Structure

```
┌─────────────────────────┐
│     Header (4 bytes)     │  Magic: "GGUF"
├─────────────────────────┤
│   Version (4 bytes)      │  Version number
├─────────────────────────┤
│  Tensor count (8 bytes)  │  Number of tensors
├─────────────────────────┤
│ Metadata count (8 bytes) │  Number of metadata entries
├─────────────────────────┤
│   Metadata KV pairs      │  Key-value metadata
│   (variable size)        │
├─────────────────────────┤
│   Tensor data            │  Binary tensor data
│   (variable size)        │
└─────────────────────────┘
```

### Loading GGUF Models

```python
from backend_connector import BackendConnector, ModelFormat

connector = BackendConnector()

# Auto-detect format from .gguf extension
connector.load_model("models/llama-3.1-8b.gguf")

# Explicit format specification
connector.load_model("models/llama-3.1-8b-q4_k.gguf",
                     model_format=ModelFormat.GGUF)
```

```cpp
#include "engine/model_loader.h"

// Parse format from path
auto format = parse_model_format("models/llama-3.1-8b.gguf");
// Returns: ModelFormat::GGUF

// Create appropriate loader
auto loader = create_model_loader(format);

// Load model and get metadata
auto metadata = loader->load("models/llama-3.1-8b.gguf");
```

## Safetensors Format

Safetensors is a safe tensor serialization format developed by Hugging Face, designed to avoid arbitrary code execution during model loading.

### Safetensors Specifications

| Property | Value |
|----------|-------|
| **Header** | JSON metadata (variable length) |
| **Tensor data** | Binary blob following header |
| **Endianness** | Little-endian |
| **Safety** | No code execution, only data |

### Safetensors File Structure

```
┌─────────────────────────────────────┐
│  JSON Header (N bytes)              │
│  {                                  │
│    "__metadata__": {                │
│      "format": "pt",                │
│      "model_type": "llama"          │
│    },                               │
│    "tensor_name": {                 │
│      "dtype": "F32",                │
│      "shape": [1024, 4096],         │
│      "data_offsets": [0, 16777216]  │
│    },                               │
│    ...                              │
│  }                                  │
├─────────────────────────────────────┤
│  Binary Tensor Data (M bytes)       │
│  [tensor_0_data][tensor_1_data]...  │
└─────────────────────────────────────┘
```

### Safetensors Metadata

The JSON header contains:
- **`__metadata__`**: Global model metadata
- **Per-tensor entries**: dtype, shape, data_offsets

### Safetensors Data Types

| Type | Bytes/Tensor | Description |
|------|--------------|-------------|
| **F32** | 4 × elements | 32-bit float |
| **F16** | 2 × elements | 16-bit float |
| **BF16** | 2 × elements | Bfloat16 |
| **I64** | 8 × elements | 64-bit integer |
| **I32** | 4 × elements | 32-bit integer |
| **I16** | 2 × elements | 16-bit integer |
| **I8** | 1 × elements | 8-bit integer |
| **U8** | 1 × elements | 8-bit unsigned |
| **BOOL** | 1 × elements | Boolean |

### Loading Safetensors Models

```python
from backend_connector import BackendConnector, ModelFormat

connector = BackendConnector()

# Single-file safetensors
connector.load_model("models/mistral-7b.safetensors")

# Sharded safetensors (with index)
connector.load_model("models/mistral-7b/model.safetensors.index.json")
```

### Sharded Safetensors

For large models that don't fit in a single file, safetensors supports sharding:

**Index file** (`model.safetensors.index.json`):
```json
{
  "metadata": {
    "total_size": 6442450944
  },
  "weight_map": {
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
    ...
  }
}
```

**Data files**: `model-00001-of-00004.safetensors`, etc.

## PyTorch Format

PyTorch native format for loading models directly from PyTorch checkpoints.

### PyTorch Specifications

| Property | Value |
|----------|-------|
| **Extension** | `.pt`, `.pth`, `.bin` |
| **Serialization** | Python pickle |
| **Content** | State dict or full model |

### Loading PyTorch Models

```python
from backend_connector import BackendConnector, ModelFormat

connector = BackendConnector()
connector.load_model("models/pytorch_model.bin",
                     model_format=ModelFormat.PYTORCH)
```

## Model Architecture Detection

dLLM automatically detects model architecture from format metadata.

### Architecture Detection Table

| Architecture | GGUF `general.architecture` | Safetensors `model_type` | Supported Models |
|-------------|----------------------------|--------------------------|-----------------|
| **Llama** | `llama` | `llama` | Llama 2, Llama 3, Llama 3.1 |
| **Mistral** | `mistral` | `mistral` | Mistral 7B, Mistral Small |
| **Gemma** | `gemma` | `gemma` | Gemma 2B, Gemma 7B, Gemma 2 |
| **Phi** | `phi2`, `phi3` | `phi` | Phi-2, Phi-3 |
| **Qwen** | `qwen2` | `qwen2` | Qwen 7B, Qwen 14B |
| **CodeLlama** | `llama` | `llama` | CodeLlama 7B, 13B, 34B |
| **ChatGLM** | `chatglm` | `chatglm` | ChatGLM 6B |
| **DeepSeek** | `deepseek` | `deepseek` | DeepSeek Coder, DeepSeek V2 |

### C++ Architecture Detection

```cpp
#include "engine/model_loader.h"

// Parse architecture from string
auto arch = parse_architecture("llama");
// Returns: ModelArchitecture::LLAMA

// Convert to string
std::string arch_str = architecture_to_string(ModelArchitecture::LLAMA);
// Returns: "llama"
```

### Python Architecture Detection

```python
from backend_connector import ModelArchitecture

# Parse from string
arch = ModelArchitecture["LLAMA"]

# Convert to string
arch_str = arch.value  # "llama"
```

## Format Comparison

| Feature | GGUF | Safetensors | PyTorch |
|---------|------|-------------|---------|
| **Quantization** | ✓ Native support | ✗ Requires conversion | ✗ Requires conversion |
| **Safety** | ✓ No code execution | ✓ No code execution | ✗ Uses pickle |
| **Lazy Loading** | ✓ Memory-mapped | ✓ Offset-based | ✗ Full load |
| **Metadata** | ✓ Rich KV pairs | ✓ JSON header | ✗ Limited |
| **Sharding** | ✗ Single file | ✓ Index-based | ✓ Manual |
| **Size Efficiency** | ✓ Best (quantized) | △ Standard | △ Standard |
| **Ecosystem** | ✓ llama.cpp | ✓ Hugging Face | ✓ PyTorch |
| **dLLM Support** | ✓ Primary | ✓ Supported | ✓ Supported |

## Best Practices

### Choosing a Format

| Use Case | Recommended Format |
|----------|-------------------|
| **Production inference** | GGUF (quantized for efficiency) |
| **Development/testing** | Safetensors (easier to inspect) |
| **Custom models** | PyTorch (direct from training) |
| **Large models (>30B)** | GGUF Q4_K or Sharded Safetensors |
| **Maximum quality** | GGUF F16 or Safetensors F32 |

### GGUF Quantization Guidelines

| Model Size | Recommended Quantization | Memory (8B model) | Quality Loss |
|-----------|-------------------------|--------------------|--------------|
| 7B | Q4_K | ~4.5 GB | <1% |
| 13B | Q4_K | ~8 GB | <1% |
| 70B | Q4_K | ~40 GB | <1% |
| 7B (max quality) | Q8_0 | ~8 GB | <0.1% |
| 70B (max quality) | Q6_K | ~50 GB | <0.5% |

### Loading Performance Tips

1. **GGUF**: Use memory-mapped loading for models > RAM size
2. **Safetensors**: Lazy loading loads tensors on demand
3. **Sharded**: Load index file first, then stream tensors
4. **PyTorch**: Full model load required before inference

## API Reference

### C++ API

```cpp
// Format parsing
ModelFormat parse_model_format(const std::string& model_path);
std::string format_to_string(ModelFormat format);

// Architecture parsing
ModelArchitecture parse_architecture(const std::string& arch_name);
std::string architecture_to_string(ModelArchitecture arch);

// Model loader
class ModelLoader {
public:
    virtual ~ModelLoader() = default;
    virtual ModelMetadata load(const std::string& path) = 0;
    virtual std::vector<Tensor> get_weights() = 0;
};

// Factory
std::unique_ptr<ModelLoader> create_model_loader(ModelFormat format);
```

### Python API

```python
# Format enum
class ModelFormat(Enum):
    UNKNOWN = "unknown"
    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    SHARDED = "sharded"
    PYTORCH = "pytorch"

# Architecture enum
class ModelArchitecture(Enum):
    UNKNOWN = "unknown"
    LLAMA = "llama"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    PHI = "phi"
    QWEN = "qwen"
    CODELLAMA = "codellama"
    CHATGLM = "chatglm"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"

# Backend connector
connector.load_model(path, model_format=None)
connector.get_model_metadata() -> Dict[str, Any]
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [FEATURES.md](FEATURES.md) - Feature documentation
- [INSTALL.md](INSTALL.md) - Installation guide
- [GETTING_STARTED.md](GETTING_STARTED.md) - Quick start guide
