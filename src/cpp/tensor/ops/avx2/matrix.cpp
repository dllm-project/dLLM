#include <immintrin.h>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace dllm {
namespace simd {

/**
 * @brief AVX2 optimized GEMM (General Matrix Multiply)
 * Computes: C = alpha * A * B + beta * C
 * 
 * Parameters:
 *   M - number of rows in A and C
 *   N - number of columns in B and C
 *   K - number of columns in A and rows in B
 */
void gemm_avx2(int M, int N, int K,
               float alpha,
               const float* A, int lda,
               const float* B, int ldb,
               float beta,
               float* C, int ldc) {
    // Simple tiling strategy for cache efficiency
    const int BLOCK_SIZE = 64;
    
    // Initialize C with beta * C if needed
    if (beta != 0.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    } else {
        // Set C to zero if beta is 0
        for (int i = 0; i < M; ++i) {
            std::memset(&C[i * ldc], 0, N * sizeof(float));
        }
    }
    
    // Block-wise GEMM
    for (int kb = 0; kb < K; kb += BLOCK_SIZE) {
        int k_end = std::min(kb + BLOCK_SIZE, K);
        
        for (int ib = 0; ib < M; ib += BLOCK_SIZE) {
            int i_end = std::min(ib + BLOCK_SIZE, M);
            
            for (int jb = 0; jb < N; jb += BLOCK_SIZE) {
                int j_end = std::min(jb + BLOCK_SIZE, N);
                
                // Process block
                for (int k = kb; k < k_end; ++k) {
                    for (int i = ib; i < i_end; ++i) {
                        float a_val = A[i * lda + k];
                        
                        // AVX2 vectorized loop for j dimension
                        int j = jb;
                        __m256 va = _mm256_set1_ps(a_val);
                        
                        for (; j + 7 < j_end; j += 8) {
                            __m256 vb = _mm256_loadu_ps(&B[k * ldb + j]);
                            __m256 vc = _mm256_loadu_ps(&C[i * ldc + j]);
                            
                            // Multiply and add
                            __m256 vprod = _mm256_mul_ps(va, vb);
                            vc = _mm256_add_ps(vc, vprod);
                            
                            _mm256_storeu_ps(&C[i * ldc + j], vc);
                        }
                        
                        // Handle remaining elements
                        for (; j < j_end; ++j) {
                            C[i * ldc + j] += a_val * B[k * ldb + j];
                        }
                    }
                }
            }
        }
    }
    
    // Scale by alpha if needed
    if (alpha != 1.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] *= alpha;
            }
        }
    }
}

/**
 * @brief AVX2 optimized matrix-vector multiplication
 */
void matvec_avx2(int M, int N,
                 const float* A, int lda,
                 const float* x,
                 float* y) {
    for (int i = 0; i < M; ++i) {
        // Compute dot product of row i with vector x
        float sum = 0.0f;
        
        int j = 0;
        __m256 vsum = _mm256_setzero_ps();
        
        for (; j + 7 < N; j += 8) {
            __m256 va = _mm256_loadu_ps(&A[i * lda + j]);
            __m256 vx = _mm256_loadu_ps(&x[j]);
            
            vsum = _mm256_fmadd_ps(va, vx, vsum);
        }
        
        // Horizontal sum
        float temp[8];
        _mm256_storeu_ps(temp, vsum);
        for (int k = 0; k < 8; ++k) {
            sum += temp[k];
        }
        
        // Handle remaining elements
        for (; j < N; ++j) {
            sum += A[i * lda + j] * x[j];
        }
        
        y[i] = sum;
    }
}

} // namespace simd
} // namespace dllm