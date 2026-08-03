#pragma once

#include <cstddef>

namespace dllm {
namespace simd {

/**
 * @brief Vector addition: output = a + b
 */
void vector_add_sse42(const float* a, const float* b, float* output, size_t length);
void vector_add_avx(const float* a, const float* b, float* output, size_t length);
void vector_add_avx2(const float* a, const float* b, float* output, size_t length);

/**
 * @brief Vector subtraction: output = a - b
 */
void vector_sub_sse42(const float* a, const float* b, float* output, size_t length);
void vector_sub_avx(const float* a, const float* b, float* output, size_t length);
void vector_sub_avx2(const float* a, const float* b, float* output, size_t length);

/**
 * @brief Vector scaling: output = input * scale
 */
void vector_scale_sse42(const float* input, float scale, float* output, size_t length);
void vector_scale_avx(const float* input, float scale, float* output, size_t length);
void vector_scale_avx2(const float* input, float scale, float* output, size_t length);

/**
 * @brief Element-wise multiplication: output = a * b
 */
void vector_mul_sse42(const float* a, const float* b, float* output, size_t length);
void vector_mul_avx(const float* a, const float* b, float* output, size_t length);
void vector_mul_avx2(const float* a, const float* b, float* output, size_t length);

/**
 * @brief Dot product: sum(a[i] * b[i])
 */
float vector_dot_sse42(const float* a, const float* b, size_t length);
float vector_dot_avx(const float* a, const float* b, size_t length);
float vector_dot_avx2(const float* a, const float* b, size_t length);

/**
 * @brief Vector sum: sum(vec[i])
 */
float vector_sum_sse42(const float* vec, size_t length);
float vector_sum_avx(const float* vec, size_t length);
float vector_sum_avx2(const float* vec, size_t length);

/**
 * @brief Vector copy
 */
void vector_copy_sse42(const float* src, float* dst, size_t length);
void vector_copy_avx(const float* src, float* dst, size_t length);
void vector_copy_avx2(const float* src, float* dst, size_t length);

/**
 * @brief Vector set (memset for floats)
 */
void vector_set_sse42(float* vec, float value, size_t length);
void vector_set_avx(float* vec, float value, size_t length);
void vector_set_avx2(float* vec, float value, size_t length);

/**
 * @brief Softmax normalization
 */
void softmax_sse42(const float* input, float* output, size_t length);
void softmax_avx(const float* input, float* output, size_t length);
void softmax_avx2(const float* input, float* output, size_t length);

/**
 * @brief GEMM: C = alpha * A * B + beta * C
 */
void gemm_sse42(int M, int N, int K,
                float alpha,
                const float* A, int lda,
                const float* B, int ldb,
                float beta,
                float* C, int ldc);
void gemm_avx(const float* A, const float* B, float* C, int M, int N, int K);
void gemm_avx2(int M, int N, int K,
               float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta,
               float* C, int ldc);

/**
 * @brief Matrix-vector multiplication
 */
void matvec_sse42(const float* A, const float* x, float* y, int M, int N);
void matvec_avx(const float* A, const float* x, float* y, int M, int N);
void matvec_avx2(int M, int N,
                 const float* A, int lda,
                 const float* x,
                 float* y);

/**
 * @brief ReLU activation
 */
void relu_sse42(const float* input, float* output, size_t length);
void relu_avx(const float* input, float* output, size_t length);
void relu_avx2(const float* input, float* output, size_t length);

/**
 * @brief Leaky ReLU activation
 */
void leaky_relu_sse42(const float* input, float* output, size_t length, float alpha);
void leaky_relu_avx(const float* input, float* output, size_t length, float alpha);
void leaky_relu_avx2(const float* input, float* output, size_t length, float alpha);

/**
 * @brief GELU activation
 */
void gelu_sse42(const float* input, float* output, size_t length);
void gelu_avx(const float* input, float* output, size_t length);
void gelu_avx2(const float* input, float* output, size_t length);

/**
 * @brief SiLU (Swish) activation
 */
void silu_sse42(const float* input, float* output, size_t length);
void silu_avx(const float* input, float* output, size_t length);
void silu_avx2(const float* input, float* output, size_t length);

/**
 * @brief Sigmoid activation
 */
void sigmoid_sse42(const float* input, float* output, size_t length);
void sigmoid_avx(const float* input, float* output, size_t length);
void sigmoid_avx2(const float* input, float* output, size_t length);

/**
 * @brief Tanh activation
 */
void tanh_sse42(const float* input, float* output, size_t length);
void tanh_avx(const float* input, float* output, size_t length);
void tanh_avx2(const float* input, float* output, size_t length);

} // namespace simd
} // namespace dllm