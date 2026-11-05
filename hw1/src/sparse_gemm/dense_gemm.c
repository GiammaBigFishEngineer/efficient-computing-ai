#include "sparse_gemm.h"

#include <arm_neon.h>

void gemm_dense_scalar(const float *a, const float *w, float *c,
                       size_t m, size_t k, size_t n)
{
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t q = 0; q < k; ++q)
                sum += a[i * k + q] * w[q * n + j];
            c[i * n + j] = sum;
        }
}

void gemm_dense_tiled_neon_openmp_TN8_T4(const float *a, const float *w,
                                         float *c, size_t m, size_t k,
                                         size_t n)
{
#pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; j += 8) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            for (size_t q = 0; q < k; ++q) {
                float32x4_t av = vdupq_n_f32(a[i * k + q]);
                acc0 = vfmaq_f32(acc0, av, vld1q_f32(w + q * n + j));
                acc1 = vfmaq_f32(acc1, av, vld1q_f32(w + q * n + j + 4));
            }
            vst1q_f32(c + i * n + j, acc0);
            vst1q_f32(c + i * n + j + 4, acc1);
        }
    }
}
