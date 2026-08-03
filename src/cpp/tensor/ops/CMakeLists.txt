# Ops module build configuration - SIMD implementations
# This file is included by src/cpp/tensor/CMakeLists.txt

# SIMD operation source files
set(SIMD_OPS_SOURCES "")

# SSE4.2 implementation (baseline)
if(USE_SSE42)
    set(SIMD_OPS_SOURCES "${SIMD_OPS_SOURCES}" ops/sse42/arithmetics.cpp)
endif()

# AVX implementation
if(USE_AVX)
    set(SIMD_OPS_SOURCES "${SIMD_OPS_SOURCES}" ops/avx/arithmetics.cpp)
endif()

# AVX2 implementation (with FMA support)
if(USE_AVX2)
    set(SIMD_OPS_SOURCES "${SIMD_OPS_SOURCES}" ops/avx2/arithmetics.cpp)
    set(SIMD_OPS_SOURCES "${SIMD_OPS_SOURCES}" ops/avx2/matrix.cpp)
    set(SIMD_OPS_SOURCES "${SIMD_OPS_SOURCES}" ops/avx2/activation.cpp)
endif()

# Create SIMD library if any implementations exist
if(NOT USE_SSE42 AND NOT USE_AVX AND NOT USE_AVX2)
    message(WARNING "No SIMD instruction sets enabled, building with basic C++ fallback")
endif()