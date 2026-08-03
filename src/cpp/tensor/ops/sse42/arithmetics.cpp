#include <immintrin.h>
#include <cstring>
#include <cmath>

namespace dllm {
namespace simd {

/**
 * @brief SSE4.2 optimized tensor addition
 * @param a First input vector
 * @param b Second input vector  
 * @param output Output vector (can be same as a or b for in-place op)
 * @param length Number of elements
 */
void vector_add_sse42(const float* a, const float* b, float* output, size_t length) {
    size_t i = 0;
    
    // Process 4 elements at a time using SSE
    for (; i + 3 < length; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vsum = _mm_add_ps(va, vb);
        _mm_storeu_ps(&output[i], vsum);
    }
    
    // Handle remaining elements (less than 4)
    for (; i < length; ++i) {
        output[i] = a[i] + b[i];
    }
}

/**
 * @brief SSE4.2 optimized tensor subtraction
 */
void vector_sub_sse42(const float* a, const float* b, float* output, size_t length) {
    size_t i = 0;
    
    for (; i + 3 < length; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vdiff = _mm_sub_ps(va, vb);
        _mm_storeu_ps(&output[i], vdiff);
    }
    
    for (; i < length; ++i) {
        output[i] = a[i] - b[i];
    }
}

/**
 * @brief SSE4.2 optimized scalar-vector multiplication
 */
void vector_scale_sse42(const float* input, float scale, float* output, size_t length) {
    __m128 vscale = _mm_set1_ps(scale);
    size_t i = 0;
    
    for (; i + 3 < length; i += 4) {
        __m128 vinput = _mm_loadu_ps(&input[i]);
        __m128 voutput = _mm_mul_ps(vinput, vscale);
        _mm_storeu_ps(&output[i], voutput);
    }
    
    for (; i < length; ++i) {
        output[i] = input[i] * scale;
    }
}

/**
 * @brief SSE4.2 optimized element-wise multiplication
 */
void vector_mul_sse42(const float* a, const float* b, float* output, size_t length) {
    size_t i = 0;
    
    for (; i + 3 < length; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vresult = _mm_mul_ps(va, vb);
        _mm_storeu_ps(&output[i], vresult);
    }
    
    for (; i < length; ++i) {
        output[i] = a[i] * b[i];
    }
}

/**
 * @brief SSE4.2 optimized dot product
 */
float vector_dot_sse42(const float* a, const float* b, size_t length) {
    size_t i = 0;
    __m128 vsum = _mm_setzero_ps();
    
    // Process 4 elements at a time
    for (; i + 3 < length; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vprod = _mm_mul_ps(va, vb);
        vsum = _mm_add_ps(vsum, vprod);
    }
    
    // Horizontal sum using SSE
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    
    float result;
    _mm_store_ss(&result, vsum);
    
    // Handle remaining elements
    for (; i < length; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

/**
 * @brief SSE4.2 optimized sum of vector elements
 */
float vector_sum_sse42(const float* vec, size_t length) {
    size_t i = 0;
    __m128 vsum = _mm_setzero_ps();
    
    for (; i + 3 < length; i += 4) {
        __m128 v = _mm_loadu_ps(&vec[i]);
        vsum = _mm_add_ps(vsum, v);
    }
    
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    
    float result;
    _mm_store_ss(&result, vsum);
    
    for (; i < length; ++i) {
        result += vec[i];
    }
    
    return result;
}

/**
 * @brief SSE4.2 optimized vector copy
 */
void vector_copy_sse42(const float* src, float* dst, size_t length) {
    size_t i = 0;
    
    for (; i + 3 < length; i += 4) {
        __m128 vsrc = _mm_loadu_ps(&src[i]);
        _mm_storeu_ps(&dst[i], vsrc);
    }
    
    // Handle remaining elements
    for (; i < length; ++i) {
        dst[i] = src[i];
    }
}

/**
 * @brief SSE4.2 optimized vector memset (set all elements to value)
 */
void vector_set_sse42(float* vec, float value, size_t length) {
    __m128 vvalue = _mm_set1_ps(value);
    size_t i = 0;
    
    for (; i + 3 < length; i += 4) {
        _mm_storeu_ps(&vec[i], vvalue);
    }
    
    for (; i < length; ++i) {
        vec[i] = value;
    }
}

/**
 * @brief SSE4.2 optimized softmax calculation (scalar fallback)
 */
void softmax_sse42(const float* input, float* output, size_t length) {
    // Find max value for numerical stability
    float max_val = input[0];
    for (size_t i = 1; i < length; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    // Calculate exponentials and sum
    float sum = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    vector_scale_sse42(output, inv_sum, output, length);
}

} // namespace simd
} // namespace dllm