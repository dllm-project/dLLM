#include "simd_config.h"
#include <string>

#if defined(__linux__) || defined(__APPLE__)
    #include <cpuid.h>
#endif

namespace dllm {
namespace simd {

// Runtime instruction set level (initialized at startup)
InstructionSet g_runtime_instruction_set = InstructionSet::NONE;

/**
 * @brief Check if CPUID is supported
 */
static bool has_cpuid() {
#if defined(__GNUC__) || defined(__clang__)
    unsigned int a, b, c, d;
    // Try to execute CPUID instruction
    __asm__ volatile (
        "push %%rbx\n\t"
        "cpuid\n\t"
        "pop %%rbx"
        : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
        : "a"(0)
        : "cc"
    );
    return true;
#else
    // For other compilers, assume CPUID is available on x86_64
    #if defined(__x86_64__) || defined(_M_X64) || defined(_X86_) || defined(__i386__)
        return true;
    #else
        return false;  // No SIMD support on non-x86 architectures
    #endif
#endif
}

/**
 * @brief Get CPU vendor string
 */
static std::string get_cpu_vendor() {
#if defined(__linux__) || defined(__APPLE__)
    if (!has_cpuid()) return "unknown";
    
    unsigned int a, b, c, d;
    char vendor[13];
    
    __asm__ volatile (
        "push %%rbx\n\t"
        "cpuid\n\t"
        "mov %%ebx, %1\n\t"
        "pop %%rbx"
        : "=a"(a), "=r"(b)
        : "a"(0)
        : "c", "d", "cc"
    );
    
    // Reorder vendor string (Intel: "GenuineIntel", AMD: "AuthenticAMD")
    *(uint32_t*)(vendor + 0) = b;
    *(uint32_t*)(vendor + 4) = d;
    *(uint32_t*)(vendor + 8) = c;
    vendor[12] = '\0';
    
    return std::string(vendor);
#else
    return "unknown";
#endif
}

/**
 * @brief Check for specific CPU feature using CPUID
 */
static bool check_feature(unsigned int leaf, unsigned int reg, unsigned int bit) {
#if defined(__linux__) || defined(__APPLE__)
    if (!has_cpuid()) return false;
    
    unsigned int a, b, c, d;
    
    __asm__ volatile (
        "push %%rbx\n\t"
        "cpuid\n\t"
        "pop %%rbx"
        : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
        : "a"(leaf)
        : "cc"
    );
    
    unsigned int reg_val = 0;
    switch (reg) {
        case 0: reg_val = a; break;
        case 1: reg_val = b; break;
        case 2: reg_val = c; break;
        case 3: reg_val = d; break;
    }
    
    return (reg_val & bit) != 0;
#else
    return false;
#endif
}

/**
 * @brief Detect instruction set using CPUID
 */
InstructionSet detect_instruction_set() {
#if !defined(__x86_64__) && !defined(_M_X64) && !defined(_X86_) && !defined(__i386__)
    // Non-x86 architecture - no SIMD support
    return InstructionSet::NONE;
#endif

    if (!has_cpuid()) {
        return InstructionSet::NONE;
    }

    // Get CPUID leaf 1 for feature flags
    unsigned int a, b, c, d;
    
    __asm__ volatile (
        "push %%rbx\n\t"
        "cpuid\n\t"
        "pop %%rbx"
        : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
        : "a"(1)
        : "cc"
    );

    // Check for SSE4.2 (bit 20 of ECX)
    bool sse42 = (c & (1u << 20)) != 0;
    
    // Check for AVX (bit 28 of ECX)
    bool avx = (c & (1u << 28)) != 0;
    
    // Check for FMA (bit 12 of ECX) - part of AVX2
    bool fma = (c & (1u << 12)) != 0;
    
    // Get CPUID leaf 7 for AVX2/AVX-512
    unsigned int a7, b7, c7, d7;
    __asm__ volatile (
        "push %%rbx\n\t"
        "cpuid\n\t"
        "pop %%rbx"
        : "=a"(a7), "=b"(b7), "=c"(c7), "=d"(d7)
        : "a"(7)
        : "cc"
    );

    // Check for AVX2 (bit 5 of EBX)
    bool avx2 = (b7 & (1u << 5)) != 0;
    
    // Check for AVX-512 (various bits in EBX, ECX, EDX)
    bool avx512 = false;
    if (a7 >= 7) {
        // Bit 16 of EBX: AVX-512 F
        // Bit 17 of EBX: AVX-512 CD
        // Bit 28 of EBX: AVX-512 VL
        // Bit 21 of EBX: AVX-512 BW
        // Bit 26 of EBX: AVX-512 DQ
        avx512 = (b7 & ((1u << 16) | (1u << 17) | (1u << 28) | (1u << 21) | (1u << 26))) 
                 == ((1u << 16) | (1u << 17) | (1u << 28) | (1u << 21) | (1u << 26));
    }

    // Return highest supported instruction set
    if (avx512) return InstructionSet::AVX512;
    if (avx2 && fma) return InstructionSet::AVX2;
    if (avx) return InstructionSet::AVX;
    if (sse42) return InstructionSet::SSE42;
    
    return InstructionSet::NONE;
}

bool has_instruction_set(InstructionSet set) {
    // Initialize runtime instruction set on first call
    if (g_runtime_instruction_set == InstructionSet::NONE) {
        g_runtime_instruction_set = detect_instruction_set();
    }
    
    switch (set) {
        case InstructionSet::AVX512:
            return g_runtime_instruction_set >= InstructionSet::AVX512;
        case InstructionSet::AVX2:
            return g_runtime_instruction_set >= InstructionSet::AVX2;
        case InstructionSet::AVX:
            return g_runtime_instruction_set >= InstructionSet::AVX;
        case InstructionSet::SSE42:
            return g_runtime_instruction_set >= InstructionSet::SSE42;
        default:
            return false;
    }
}

} // namespace simd
} // namespace dllm