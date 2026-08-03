#pragma once

#include <cstdint>

namespace dllm {
namespace simd {

/**
 * @brief Detect the highest supported SIMD instruction set
 */
enum class InstructionSet {
    NONE = 0,        // No SIMD support (fallback to C++)
    SSE42 = 1,       // SSE4.2 (baseline)
    AVX = 2,         // AVX 256-bit
    AVX2 = 3,        // AVX2 with FMA
    AVX512 = 4,      // AVX-512 (32 registers, 512-bit vectors)
};

/**
 * @brief Get the highest supported instruction set at runtime
 */
InstructionSet detect_instruction_set();

/**
 * @brief Check if a specific instruction set is available
 */
bool has_instruction_set(InstructionSet set);

/**
 * @brief Compile-time instruction set selection based on build flags
 */
#if defined(__AVX512F__)
    #define DLLM_SIMD_COMPILE_TIME_LEVEL InstructionSet::AVX512
#elif defined(__AVX2__)
    #define DLLM_SIMD_COMPILE_TIME_LEVEL InstructionSet::AVX2
#elif defined(__AVX__)
    #define DLLM_SIMD_COMPILE_TIME_LEVEL InstructionSet::AVX
#elif defined(__SSE4_2__)
    #define DLLM_SIMD_COMPILE_TIME_LEVEL InstructionSet::SSE42
#else
    #define DLLM_SIMD_COMPILE_TIME_LEVEL InstructionSet::NONE
#endif

/**
 * @brief Runtime instruction set level (determined at program startup)
 */
extern InstructionSet g_runtime_instruction_set;

/**
 * @brief Get the current working instruction set
 */
inline InstructionSet get_current_instruction_set() {
    return g_runtime_instruction_set != InstructionSet::NONE 
        ? g_runtime_instruction_set 
        : DLLM_SIMD_COMPILE_TIME_LEVEL;
}

} // namespace simd
} // namespace dllm