# Tensor module build configuration

add_library(dllm_tensor STATIC
    tensor.cpp
    ops/arithmetics.cpp
    ops/matrix.cpp
    ops/activation.cpp
    memory_pool.cpp
)

target_include_directories(dllm_tensor PUBLIC ${CMAKE_SOURCE_DIR}/src/cpp)
target_link_libraries(dllm_tensor PUBLIC dllm_core)

# SIMD-specific source files (compiled conditionally)
set(SIMD_SOURCES "")

# SSE4.2 fallback path
if(USE_SSE42)
    target_compile_definitions(dllm_tensor PRIVATE USE_SSE42=1)
endif()

# AVX path
if(USE_AVX)
    target_compile_definitions(dllm_tensor PRIVATE USE_AVX=1)
endif()

# AVX2 path (default for modern CPUs)
if(USE_AVX2)
    target_compile_definitions(dllm_tensor PRIVATE USE_AVX2=1)
endif()