# Communication module build configuration

add_library(dllm_comm STATIC
    network.cpp
    rpc.cpp
)

target_include_directories(dllm_comm PUBLIC ${CMAKE_SOURCE_DIR}/src/cpp)
target_link_libraries(dllm_comm PUBLIC dllm_engine)

# SIMD-specific compilation
if(USE_SSE42)
    target_compile_definitions(dllm_comm PRIVATE USE_SSE42=1)
endif()

if(USE_AVX2)
    target_compile_definitions(dllm_comm PRIVATE USE_AVX2=1)
endif()