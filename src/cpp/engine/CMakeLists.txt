# Engine module build configuration

add_library(dllm_engine STATIC
    inference_core.cpp
    request_handler.cpp
)

target_include_directories(dllm_engine PUBLIC ${CMAKE_SOURCE_DIR}/src/cpp)
target_link_libraries(dllm_engine PUBLIC dllm_tensor)

# SIMD-specific compilation
if(USE_SSE42)
    target_compile_definitions(dllm_engine PRIVATE USE_SSE42=1)
endif()

if(USE_AVX)
    target_compile_definitions(dllm_engine PRIVATE USE_AVX=1)
endif()

if(USE_AVX2)
    target_compile_definitions(dllm_engine PRIVATE USE_AVX2=1)
endif()