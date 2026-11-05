#include "sparse_gemm.h"

#include <arm_neon.h>

#define TM 4 /* row tile: fixed at four input rows */

/*
 * Single-row NEON update (same as gemm_sparse_neon). Only used for a trailing
 * partial tile when m is not a multiple of TM (never for M = 128).
 */
static void sparse_row_neon(const float *a_row, const sparse_weights *w,
                            float *c_row)
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

/*
 * 4x4 block-sparse GEMM: 4-row tile + 4-column NEON + OpenMP over row tiles.
 *
 * Each OpenMP iteration owns one independent tile of four output rows.
 * For every output-column group it keeps one float32x4_t accumulator per row
 * (acc0..acc3 = the 4x4 output tile). For each stored block the four input
 * values A[i, 4q .. 4q+3] of every tile row are loaded once as a vector
 * (x0..x3). Each 4-value weight row r of the block is loaded once with
 * vld1q_f32 and reused by four FMAs, one per tile row; vfmaq_laneq_f32
 * broadcasts lane r of x_t inside the FMA (same math as
 * vfmaq_f32(acc, vdupq_n_f32(A[i, 4q + r]), wv), but without a separate
 * scalar load + dup per update). The block traversal and ordering are the
 * same as in the other kernels.
 */
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
            }

            vst1q_f32(c0 + 4 * g, acc0);
            vst1q_f32(c1 + 4 * g, acc1);
            vst1q_f32(c2 + 4 * g, acc2);
            vst1q_f32(c3 + 4 * g, acc3);
        }
    }

    for (size_t i = tiles * TM; i < m; ++i)
        sparse_row_neon(a + i * k, w, c + i * n);

    return SPARSE_OK;
}
