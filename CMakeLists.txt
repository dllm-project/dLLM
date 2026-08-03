# dLLM - Distributed CPU AI Inference Engine
# CMake build configuration

cmake_minimum_required(VERSION 3.20)
project(dLLM VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type (default to Release if not specified)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# ============================================
# Instruction Set Configuration
# ============================================

option(USE_SSE42 "Enable SSE4.2 instruction set support" ON)
option(USE_AVX "Enable AVX instruction set support" OFF)
option(USE_AVX2 "Enable AVX2 instruction set support (default)" ON)
option(USE_CUDA "Enable NVIDIA CUDA GPU acceleration" OFF)

# Check supported instruction sets
include(CheckCXXCompilerFlag)

# SSE4.2 - baseline compatibility (works on all modern CPUs)
if(USE_SSE42)
    check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
    if(COMPILER_SUPPORTS_SSE42)
        message(STATUS "SSE4.2: SUPPORTED")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
    else()
        message(WARNING "SSE4.2 not supported by compiler")
    endif()
endif()

# AVX - 256-bit operations
if(USE_AVX)
    check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
    if(COMPILER_SUPPORTS_AVX)
        message(STATUS "AVX: SUPPORTED")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    else()
        message(WARNING "AVX not supported by compiler")
    endif()
endif()

# AVX2 - 256-bit with FMA
if(USE_AVX2)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    if(COMPILER_SUPPORTS_AVX2)
        message(STATUS "AVX2: SUPPORTED")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
        # Enable FMA support with AVX2
        check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
        if(COMPILER_SUPPORTS_FMA)
            message(STATUS "FMA: SUPPORTED")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
        endif()
    else()
        message(WARNING "AVX2 not supported by compiler (building without AVX2)")
    endif()
endif()

# ============================================
# CUDA Configuration
# ============================================

if(USE_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_NVCC_FLAGS}")
    message(STATUS "CUDA: ENABLED (compute capability >= 6.0 for Pascal)")
endif()

# ============================================
# Compiler-specific flags
# ============================================

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
    
    # Debug symbols for debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DDEBUG")
    endif()
endif()

# ============================================
# Output directories
# ============================================

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ============================================
# Include directories
# ============================================

include_directories(
    ${CMAKE_SOURCE_DIR}/src/cpp
    ${CMAKE_BINARY_DIR}
)

# ============================================
# Subdirectories
# ============================================

# Add subdirectories in dependency order (simd before tensor, tensor before engine, etc.)
add_subdirectory(src/cpp/simd)
add_subdirectory(src/cpp/tensor)
add_subdirectory(src/cpp/engine)
add_subdirectory(src/cpp/comm)

if(USE_CUDA)
    add_subdirectory(src/cpp/gpu)
endif()

# ============================================
# Python bindings (pybind11)
# ============================================

find_package(pybind11 REQUIRED)
add_subdirectory(src/python)

# ============================================
# Installation
# ============================================

install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/cpp
        DESTINATION include/dllm
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")

install(TARGETS dllm_core
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)