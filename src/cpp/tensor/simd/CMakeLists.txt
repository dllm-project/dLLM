# SIMD module build configuration

add_library(dllm_simd STATIC
    simd_config.cpp
)

target_include_directories(dllm_simd PUBLIC 
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_SOURCE_DIR}/src/cpp/tensor/simd
)
target_link_libraries(dllm_simd PUBLIC dllm_tensor)

# Install SIMD headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/cpp/tensor/simd
        DESTINATION include/dllm
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Instruction set specific compilation flags
if(USE_SSE42)
    target_compile_definitions(dllm_simd PRIVATE USE_SSE42=1)
endif()

if(USE_AVX)
    target_compile_definitions(dllm_simd PRIVATE USE_AVX=1)
    target_compile_options(dllm_simd PRIVATE -mavx)
endif()

if(USE_AVX2)
    target_compile_definitions(dllm_simd PRIVATE USE_AVX2=1)
    target_compile_options(dllm_simd PRIVATE -mavx2 -mfma)
endif()

