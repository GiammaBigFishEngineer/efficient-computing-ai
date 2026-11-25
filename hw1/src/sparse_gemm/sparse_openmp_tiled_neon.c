#include "sparse_gemm.h"

#include <arm_neon.h>

#define TM 4 /* row tile: fixed at four input rows */

/*
 * Single-row NEON update (same as gemm_sparse_neon). Only used for a trailing
 * partial tile when m is not a multiple of TM (never for M = 128).
 */
static void sparse_row_neon(const float *a_row, const sparse_weights *w,
                            float *c_row) {
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
 * (acc0..acc3 = the 4x4 output tile). Each 4-value weight row r of a stored
 * block is loaded once with vld1q_f32 and reused by four FMAs, one per tile
 * row. vfmaq_n_f32(acc, wv, x) computes acc + wv * x with the scalar x
 * broadcast to the four lanes inside the fused multiply-add: the same math as
 * vfmaq_f32(acc, vdupq_n_f32(x), wv), without a separate dup.
 *
 * This formulation was chosen by measurement on the Raspberry Pi 4 with the
 * course toolchain (GCC, -O0), see experiments/try_combined_variants.sh:
 * 1-thread latency 26.2 ms vs 29.6 ms (vdupq + vfmaq) and 50.3 ms
 * (vfmaq_laneq_f32), with linear scaling to 6.6 ms on 4 threads.
 * The block traversal and ordering are the same as in the other kernels.
 */
int gemm_sparse_openmp_tiled_neon(const float *a, const sparse_weights *w,
                                  float *c, size_t m) {
    if (!a || !w || !c)
        return SPARSE_ERROR;

    const size_t k = w->k;
    const size_t n = w->n;
    const size_t groups = n / 4;
    const size_t tiles = m / TM;

    // distribute row tiles among threads
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    // iterate on each input tile (group of 4 WHOLE rows)
    for (size_t t = 0; t < tiles; ++t) {
        // i0: index of the first input row of tile t
        // point to the first element of each whole input row of the tile
        const size_t i0 = t * TM;
        const float *a0 = a + (i0 + 0) * k;
        const float *a1 = a + (i0 + 1) * k;
        const float *a2 = a + (i0 + 2) * k;
        const float *a3 = a + (i0 + 3) * k;

        // prepare the respective result pointers
        float *c0 = c + (i0 + 0) * n;
        float *c1 = c + (i0 + 1) * n;
        float *c2 = c + (i0 + 2) * n;
        float *c3 = c + (i0 + 3) * n;

        // iterate on each weight group (4 whole columns)
        // initialize one NEON accumulator per selected output row
        for (size_t g = 0; g < groups; ++g) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            // access the selected group start and end idx (excluded)
            // iterate from group start idx to group end idx through p
            // p: weight subgroup idx
            for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
                // acccess subgroup row idx and obtain its first row idx
                // in the full weight matrix (k0) --> also the first matching input
                // column get pointer to the first stored value of the weight subgroup
                const size_t k0 = 4 * (size_t)w->block_row_idx[p];
                const float *w_blk = w->block_values + 16 * p;

                // load one weight row values and re-use them for the 4 selected input
                // rows their matching values are in the same input column
                for (size_t r = 0; r < 4; ++r) {
                    // load the 4 weights of subgroup row into one NEON vector
                    const float32x4_t wv = vld1q_f32(w_blk + 4 * r);

                    // access the matching input value, multiply the single input value
                    // for all 4 weight values in the same row
                    // no need for vdupq_n_f32 --> vfmaq_n_f32 specifically accepts a
                    // scalar and uses it for all four lanes
                    //acc0 = vfmaq_n_f32(acc0, wv, a0[k0 + r]);
                    acc0 = vfmaq_f32(acc0, vdupq_n_f32(a0[k0 + r]), wv);
                    //acc1 = vfmaq_n_f32(acc1, wv, a1[k0 + r]);
                    acc1 = vfmaq_f32(acc1, vdupq_n_f32(a1[k0 + r]), wv);
                    //acc2 = vfmaq_n_f32(acc2, wv, a2[k0 + r]);
                    acc2 = vfmaq_f32(acc2, vdupq_n_f32(a2[k0 + r]), wv);
                    //acc3 = vfmaq_n_f32(acc3, wv, a3[k0 + r]);
                    acc3 = vfmaq_f32(acc3, vdupq_n_f32(a3[k0 + r]), wv);
                }
            }

            // save the results before moving to the next group
            vst1q_f32(c0 + 4 * g, acc0);
            vst1q_f32(c1 + 4 * g, acc1);
            vst1q_f32(c2 + 4 * g, acc2);
            vst1q_f32(c3 + 4 * g, acc3);
        }
    }

    // process the remaining values we cannot handle
    // through tiling individually with NEON
    for (size_t i = tiles * TM; i < m; ++i)
        sparse_row_neon(a + i * k, w, c + i * n);

    return SPARSE_OK;
}
