# AVX2 implementation of arithmetic operations (with FMA)

add_library(dllm_ops_avx2 STATIC
    arithmetics.cpp
    matrix.cpp
    activation.cpp
)

target_include_directories(dllm_ops_avx2 PUBLIC ${CMAKE_SOURCE_DIR}/src/cpp)
target_link_libraries(dllm_ops_avx2 PUBLIC dllm_tensor dllm_simd)

# AVX2 specific compilation flags (with FMA)
target_compile_definitions(dllm_ops_avx2 PRIVATE USE_AVX2=1)
target_compile_options(dllm_ops_avx2 PRIVATE -mavx2 -mfma)