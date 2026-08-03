# SSE4.2 implementation of arithmetic operations

add_library(dllm_ops_sse42 STATIC
    arithmetics.cpp
)

target_include_directories(dllm_ops_sse42 PUBLIC ${CMAKE_SOURCE_DIR}/src/cpp)
target_link_libraries(dllm_ops_sse42 PUBLIC dllm_tensor dllm_simd)

# SSE4.2 specific compilation flags
target_compile_definitions(dllm_ops_sse42 PRIVATE USE_SSE42=1)
target_compile_options(dllm_ops_sse42 PRIVATE -msse4.2)