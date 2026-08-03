# Tensor module build configuration

add_library(dllm_tensor STATIC
    tensor.cpp
)

target_include_directories(dllm_tensor PUBLIC ${CMAKE_SOURCE_DIR}/src/cpp)

# Include SIMD operation subdirectories (must be before linking)
add_subdirectory(ops/sse42)
add_subdirectory(ops/avx)
add_subdirectory(ops/avx2)

# Link SIMD library and operation libraries based on enabled instruction sets
target_link_libraries(dllm_tensor PUBLIC dllm_simd)

if(USE_SSE42)
    target_link_libraries(dllm_tensor PRIVATE dllm_ops_sse42)
endif()

if(USE_AVX)
    target_link_libraries(dllm_tensor PRIVATE dllm_ops_avx)
endif()

if(USE_AVX2)
    target_compile_definitions(dllm_tensor PRIVATE USE_AVX2=1)
    target_link_libraries(dllm_tensor PRIVATE dllm_ops_avx2)
endif()