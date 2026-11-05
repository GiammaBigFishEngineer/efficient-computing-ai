/*
 * Experiment: alternative inner loops for gemm_sparse_openmp_tiled_neon.
 * Compiled INSTEAD of src/sparse_gemm/sparse_openmp_tiled_neon.c with
 * -DVARIANT=<n> by experiments/try_combined_variants.sh, to find which NEON
 * formulation is fastest with the course toolchain (GCC, -O0) on the Pi.
 *
 * All variants: same 4-row tile, same OpenMP loop over row tiles, same block
 * traversal; they differ only in how the 4x4 update of one block is written.
 *
 *   1  vld1q A + vfmaq_laneq_f32            (current submitted version)
 *   2  vdupq_n_f32(A) + vfmaq_f32, r loop   (template hint)
 *   3  vfmaq_n_f32(acc, w, A), r loop       (FMA by scalar, no explicit dup)
 *   4  vfmaq_n_f32, fully unrolled, precomputed pointers
 *   5  vmlaq_n_f32 (mul+add by scalar), fully unrolled
 */
#include "sparse_gemm.h"

#include <arm_neon.h>

#ifndef VARIANT
#define VARIANT 1
#endif

#define TM 4

static void row_neon(const float *a_row, const sparse_weights *w, float *c_row)
{
    for (size_t g = 0; g < w->n / 4; ++g) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
            const float *a_blk = a_row + 4 * (size_t)w->block_row_idx[p];
            const float *w_blk = w->block_values + 16 * p;
            for (size_t r = 0; r < 4; ++r)
                acc = vfmaq_f32(acc, vdupq_n_f32(a_blk[r]), vld1q_f32(w_blk + 4 * r));
        }
        vst1q_f32(c_row + 4 * g, acc);
    }
}

int gemm_sparse_openmp_tiled_neon(const float *a, const sparse_weights *w,
                                  float *c, size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;

    const size_t k = w->k;
    const size_t n = w->n;
    const size_t groups = n / 4;
    const size_t tiles = m / TM;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t t = 0; t < tiles; ++t) {
        const size_t i0 = t * TM;
        const float *a0 = a + (i0 + 0) * k;
        const float *a1 = a + (i0 + 1) * k;
        const float *a2 = a + (i0 + 2) * k;
        const float *a3 = a + (i0 + 3) * k;
        float *c0 = c + (i0 + 0) * n;
        float *c1 = c + (i0 + 1) * n;
        float *c2 = c + (i0 + 2) * n;
        float *c3 = c + (i0 + 3) * n;

        for (size_t g = 0; g < groups; ++g) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
                const size_t k0 = 4 * (size_t)w->block_row_idx[p];
                const float *w_blk = w->block_values + 16 * p;

#if VARIANT == 1
                const float32x4_t x0 = vld1q_f32(a0 + k0);
                const float32x4_t x1 = vld1q_f32(a1 + k0);
                const float32x4_t x2 = vld1q_f32(a2 + k0);
                const float32x4_t x3 = vld1q_f32(a3 + k0);
                float32x4_t wv;
                wv = vld1q_f32(w_blk + 0);
                acc0 = vfmaq_laneq_f32(acc0, wv, x0, 0); acc1 = vfmaq_laneq_f32(acc1, wv, x1, 0);
                acc2 = vfmaq_laneq_f32(acc2, wv, x2, 0); acc3 = vfmaq_laneq_f32(acc3, wv, x3, 0);
                wv = vld1q_f32(w_blk + 4);
                acc0 = vfmaq_laneq_f32(acc0, wv, x0, 1); acc1 = vfmaq_laneq_f32(acc1, wv, x1, 1);
                acc2 = vfmaq_laneq_f32(acc2, wv, x2, 1); acc3 = vfmaq_laneq_f32(acc3, wv, x3, 1);
                wv = vld1q_f32(w_blk + 8);
                acc0 = vfmaq_laneq_f32(acc0, wv, x0, 2); acc1 = vfmaq_laneq_f32(acc1, wv, x1, 2);
                acc2 = vfmaq_laneq_f32(acc2, wv, x2, 2); acc3 = vfmaq_laneq_f32(acc3, wv, x3, 2);
                wv = vld1q_f32(w_blk + 12);
                acc0 = vfmaq_laneq_f32(acc0, wv, x0, 3); acc1 = vfmaq_laneq_f32(acc1, wv, x1, 3);
                acc2 = vfmaq_laneq_f32(acc2, wv, x2, 3); acc3 = vfmaq_laneq_f32(acc3, wv, x3, 3);

#elif VARIANT == 2
                for (size_t r = 0; r < 4; ++r) {
                    const float32x4_t wv = vld1q_f32(w_blk + 4 * r);
                    acc0 = vfmaq_f32(acc0, vdupq_n_f32(a0[k0 + r]), wv);
                    acc1 = vfmaq_f32(acc1, vdupq_n_f32(a1[k0 + r]), wv);
                    acc2 = vfmaq_f32(acc2, vdupq_n_f32(a2[k0 + r]), wv);
                    acc3 = vfmaq_f32(acc3, vdupq_n_f32(a3[k0 + r]), wv);
                }

#elif VARIANT == 3
                for (size_t r = 0; r < 4; ++r) {
                    const float32x4_t wv = vld1q_f32(w_blk + 4 * r);
                    acc0 = vfmaq_n_f32(acc0, wv, a0[k0 + r]);
                    acc1 = vfmaq_n_f32(acc1, wv, a1[k0 + r]);
                    acc2 = vfmaq_n_f32(acc2, wv, a2[k0 + r]);
                    acc3 = vfmaq_n_f32(acc3, wv, a3[k0 + r]);
                }

#elif VARIANT == 4 || VARIANT == 5
#if VARIANT == 4
#define UPD(acc, wv, s) acc = vfmaq_n_f32(acc, wv, s)
#else
#define UPD(acc, wv, s) acc = vmlaq_n_f32(acc, wv, s)
#endif
                const float *x0 = a0 + k0, *x1 = a1 + k0;
                const float *x2 = a2 + k0, *x3 = a3 + k0;
                float32x4_t wv;
                wv = vld1q_f32(w_blk + 0);
                UPD(acc0, wv, x0[0]); UPD(acc1, wv, x1[0]);
                UPD(acc2, wv, x2[0]); UPD(acc3, wv, x3[0]);
                wv = vld1q_f32(w_blk + 4);
                UPD(acc0, wv, x0[1]); UPD(acc1, wv, x1[1]);
                UPD(acc2, wv, x2[1]); UPD(acc3, wv, x3[1]);
                wv = vld1q_f32(w_blk + 8);
                UPD(acc0, wv, x0[2]); UPD(acc1, wv, x1[2]);
                UPD(acc2, wv, x2[2]); UPD(acc3, wv, x3[2]);
                wv = vld1q_f32(w_blk + 12);
                UPD(acc0, wv, x0[3]); UPD(acc1, wv, x1[3]);
                UPD(acc2, wv, x2[3]); UPD(acc3, wv, x3[3]);
#undef UPD
#else
#error "unknown VARIANT"
#endif
            }

            vst1q_f32(c0 + 4 * g, acc0);
            vst1q_f32(c1 + 4 * g, acc1);
            vst1q_f32(c2 + 4 * g, acc2);
            vst1q_f32(c3 + 4 * g, acc3);
        }
    }

    for (size_t i = tiles * TM; i < m; ++i)
        row_neon(a + i * k, w, c + i * n);

    return SPARSE_OK;
}
