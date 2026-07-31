# Contributing to dLLM

Thank you for your interest in contributing to dLLM! This document outlines the process for contributing to the project.

dLLM uses a three-tier architecture:
- **Python Frontend**: FastAPI server with OpenAI-compatible API
- **Rust Tokenizer**: High-performance tokenizer with SIMD optimizations (85K-92K tok/s, AVX2/AVX512)
- **C++ Backend**: High-performance inference engine with SIMD optimizations (AVX2/AVX512)

## Code of Conduct

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what is best for the community

## How to Contribute

### 1. Report Bugs

Use GitHub Issues to report bugs:
- Describe the issue clearly
- Include steps to reproduce
- Specify your environment (OS, compiler, CPU features)

### 2. Suggest Features

Feature requests should include:
- The problem you're solving
- Proposed solution
- Any alternatives considered

### 3. Submit Pull Requests

**Prerequisites:**
- Follow coding style (see below)
- Add tests for new functionality
- Update documentation as needed

**PR Process:**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and benchmarks
5. Submit PR with clear description

## Project Structure

```
dLLM/
├── src/cpp/                    # C++ backend (inference engine)
│   ├── engine/                 # Inference orchestration
│   │   ├── inference_core.cpp/hpp
│   │   └── request_handler.cpp/hpp
│   ├── tensor/                 # Tensor operations with SIMD
│   │   ├── tensor.cpp/hpp
│   │   └── ops/
│   ├── comm/                   # Communication layer
│   │   ├── network.cpp/hpp
│   │   └── rpc.cpp/hpp
│   └── python_bridge/          # pybind11 bindings
├── src/python/                 # Python frontend (FastAPI)
│   ├── server.py               # FastAPI application
│   ├── api_routes/             # OpenAI-compatible endpoints
│   │   ├── chat_completions.py
│   │   └── embeddings.py
│   └── backend_connector.py    # C++ bridge interface
├── docs/                       # Documentation
└── tests/                      # Test suite
```

## Coding Standards

### C++ Style (Backend)

```cpp
// Header guards
#pragma once

// Include order
#include <system_headers.h>
#include "local_headers.h"

// Class naming: PascalCase
class MyClass {
public:
    // Methods: camelCase
    void myMethod();
    
    // Constants: kConstantName
    const int kDefaultBufferSize = 4096;
    
private:
    // Member variables: snake_case_
    int member_variable_;
};
```

### Python Style (Frontend)

Follow PEP 8 guidelines:
- 4 spaces per indentation level
- Max line length 100 characters
- Docstrings for all public functions

### API Design

OpenAI-compatible endpoints must follow the official OpenAI API specification.

## Testing

### Unit Tests

```cpp
// src/cpp/tests/tensor_test.cpp
TEST(TensorTest, MatmulShape) {
    Tensor a({2, 3}, FP32);
    Tensor b({3, 4}, FP32);
    Tensor result = matmul(a, b);
    
    EXPECT_EQ(result.shape(), Shape({2, 4}));
}
```

### Python Tests

```python
# tests/test_api.py
def test_chat_completion():
    client = OpenAI(base_url="http://localhost:8000/v1")
    
    response = client.chat.completions.create(
        model="llama-7b",
        messages=[{"role": "user", "content": "Hello"}]
    )
    
    assert len(response.choices) > 0

def test_embeddings():
    client = OpenAI(base_url="http://localhost:8000/v1")
    
    response = client.embeddings.create(
        model="bert-base",
        input=["test"]
    )
    
    assert len(response.data) == 1
```

### Integration Tests

```python
# tests/test_distributed.py
def test_distributed_inference():
    # Test distributed inference across nodes
    pass
```

## Documentation

All new features must include:
- API documentation in headers/Python docstrings
- Usage examples
- Performance characteristics

## Adding Rust Tokenizer Features

To contribute to the Rust tokenizer:

### Prerequisites
- **Rust 1.70+** installed via rustup
- Basic understanding of SIMD vectorization (AVX2/AVX512)
- Familiarity with FFI (Foreign Function Interface)

### Development Steps
1. Create a feature branch: `git checkout -b feature/rust-tokenizer`
2. Implement your changes in the `tokenizer/` directory
3. Update documentation in `RUST_TOKENIZER.md` as needed
4. Add tests covering new functionality

### SIMD Optimization Guidelines
- Use intrinsics from `std::simd` or `std::arch` for platform-specific optimizations
- Ensure fallback paths for CPUs without AVX2/AVX512
- Profile before and after changes to verify performance improvements

### FFI Requirements
- All exported functions must have `extern "C"` linkage
- Document parameter ownership (who frees memory)
- Handle errors gracefully with clear error codes

### Testing
```bash
cd tokenizer

# Run tests
cargo test --all-features

# Run benchmarks (requires nightly)
cargo +nightly bench

# Check SIMD optimization output
cargo build --release --features avx512
```

## Adding New OpenAI Endpoints

To add a new OpenAI-compatible endpoint:

1. Create a route file in `src/python/api_routes/`
2. Implement the request/response models using Pydantic
3. Add tests for the new endpoint
4. Update API_REFERENCE.md with documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
