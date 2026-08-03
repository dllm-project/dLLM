# AVX implementation of arithmetic operations

add_library(dllm_ops_avx STATIC
    arithmetics.cpp
)

target_include_directories(dllm_ops_avx PUBLIC ${CMAKE_SOURCE_DIR}/src/cpp)
target_link_libraries(dllm_ops_avx PUBLIC dllm_tensor dllm_simd)

# AVX specific compilation flags
target_compile_definitions(dllm_ops_avx PRIVATE USE_AVX=1)
target_compile_options(dllm_ops_avx PRIVATE -mavx)