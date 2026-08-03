#include <immintrin.h>
#include <cmath>
#include <algorithm>

namespace dllm {
namespace simd {

/**
 * @brief AVX2 optimized ReLU activation
 * output = max(0, input)
 */
void relu_avx2(const float* input, float* output, size_t length) {
    __m256 vzero = _mm256_set1_ps(0.0f);
    size_t i = 0;
    
    for (; i + 7 < length; i += 8) {
        __m256 vin = _mm256_loadu_ps(&input[i]);
        // Use max to implement ReLU: output = max(0, input)
        __m256 vout = _mm256_max_ps(vin, vzero);
        _mm256_storeu_ps(&output[i], vout);
    }
    
    // Handle remaining elements
    for (; i < length; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

/**
 * @brief AVX2 optimized Leaky ReLU activation
 * output = (input > 0) ? input : alpha * input
 */
void leaky_relu_avx2(const float* input, float* output, size_t length, float alpha) {
    __m256 vzero = _mm256_set1_ps(0.0f);
    __m256 valpha = _mm256_set1_ps(alpha);
    size_t i = 0;
    
    for (; i + 7 < length; i += 8) {
        __m256 vin = _mm256_loadu_ps(&input[i]);
        
        // For AVX2 without FMA, use scalar comparison approach
        float temp_in[8], temp_out[8];
        _mm256_storeu_ps(temp_in, vin);
        
        for (int j = 0; j < 8; ++j) {
            temp_out[j] = (temp_in[j] > 0) ? temp_in[j] : alpha * temp_in[j];
        }
        
        _mm256_storeu_ps(&output[i], _mm256_loadu_ps(temp_out));
    }
    
    for (; i < length; ++i) {
        output[i] = (input[i] > 0) ? input[i] : alpha * input[i];
    }
}

/**
 * @brief AVX2 optimized GELU activation (approximate)
 * GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 */
void gelu_avx2(const float* input, float* output, size_t length) {
    // Constants for GELU approximation
    __m256 c0 = _mm256_set1_ps(0.5f);
    __m256 c1 = _mm256_set1_ps(0.7978845608028654f);  // sqrt(2/π)
    __m256 c2 = _mm256_set1_ps(0.044715f);
    
    size_t i = 0;
    
    for (; i + 7 < length; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        
        // x^3
        __m256 x3 = _mm256_mul_ps(x, _mm256_mul_ps(x, x));
        
        // sqrt(2/π) * (x + 0.044715 * x^3)
        __m256 inner = _mm256_mul_ps(c1, _mm256_add_ps(x, _mm256_mul_ps(c2, x3)));
        
        // tanh(inner)
        // Using approximation: tanh(x) ≈ x / (1 + 0.5|x|)
        __m256 abs_inner = _mm256_and_ps(inner, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));
        __m256 tanh_val = _mm256_div_ps(inner, _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(_mm256_set1_ps(0.5f), abs_inner)));
        
        // 0.5 * x * (1 + tanh(...))
        __m256 result = _mm256_mul_ps(c0, _mm256_mul_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), tanh_val)));
        
        _mm256_storeu_ps(&output[i], result);
    }
    
    // Handle remaining elements with scalar
    for (; i < length; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x3);
        float tanh_val = std::tanh(inner);
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

/**
 * @brief AVX2 optimized SiLU (Swish) activation
 * SiLU(x) = x * sigmoid(x)
 */
void silu_avx2(const float* input, float* output, size_t length) {
    size_t i = 0;
    
    for (; i + 7 < length; i += 8) {
        float temp_in[8], temp_out[8];
        _mm256_storeu_ps(temp_in, _mm256_loadu_ps(&input[i]));
        
        for (int j = 0; j < 8; ++j) {
            float x = temp_in[j];
            temp_out[j] = x / (1.0f + std::exp(-x));
        }
        
        _mm256_storeu_ps(&output[i], _mm256_loadu_ps(temp_out));
    }
    
    for (; i < length; ++i) {
        output[i] = input[i] / (1.0f + std::exp(-input[i]));
    }
}

/**
 * @brief AVX2 optimized sigmoid activation
 */
void sigmoid_avx2(const float* input, float* output, size_t length) {
    __m256 vone = _mm256_set1_ps(1.0f);
    
    size_t i = 0;
    
    for (; i + 7 < length; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        
        // Sigmoid: 1 / (1 + exp(-x))
        float temp[8];
        _mm256_storeu_ps(temp, x);
        
        for (int j = 0; j < 8; ++j) {
            output[i + j] = 1.0f / (1.0f + std::exp(-temp[j]));
        }
    }
    
    for (; i < length; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

/**
 * @brief AVX2 optimized tanh activation
 */
void tanh_avx2(const float* input, float* output, size_t length) {
    __m256 vone = _mm256_set1_ps(1.0f);
    
    size_t i = 0;
    
    for (; i + 7 < length; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        
        // tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        float temp[8];
        _mm256_storeu_ps(temp, x);
        
        for (int j = 0; j < 8; ++j) {
            output[i + j] = std::tanh(temp[j]);
        }
    }
    
    for (; i < length; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

} // namespace simd
} // namespace dllm