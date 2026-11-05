#include "sparse_gemm.h"

#define TM 4 /* row tile: exactly four input rows */

/*
 * Plain-C helper for a single output row (same loop as gemm_sparse_scalar).
 * Only used for a trailing partial tile when m is not a multiple of TM
 * (never the case for the fixed M = 128 test case).
 */
static void sparse_row_scalar(const float *a_row, const sparse_weights *w,
                              float *c_row)
{
    for (size_t g = 0; g < w->n / 4; ++g) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
            size_t k0 = 4 * (size_t)w->block_row_idx[p];
            for (size_t r = 0; r < 4; ++r) {
                float av = a_row[k0 + r];
                for (size_t col = 0; col < 4; ++col)
                    acc[col] += av * w->block_values[16 * p + 4 * r + col];
            }
        }
        for (size_t col = 0; col < 4; ++col) c_row[4 * g + col] = acc[col];
    }
}

/*
 * 4x4 block-sparse GEMM with a 4-row tile and no SIMD.
 *
 * Rows i0 .. i0+3 are computed together. For every output-column group the
 * 4x4 output tile C[i0..i0+3, 4g..4g+3] lives in local accumulators. Each
 * weight row of a stored block (four values) is loaded once into locals and
 * reused for all four input rows, so the weight stream is read M/4 times
 * instead of M times (4x fewer weight/index loads than gemm_sparse_scalar).
 */
int gemm_sparse_tiled(const float *a, const sparse_weights *w, float *c,
                        size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;

    const size_t k = w->k;
    const size_t n = w->n;
    const size_t groups = n / 4;
    const size_t m_full = m - m % TM;

    for (size_t i0 = 0; i0 < m_full; i0 += TM) {
        const float *a0 = a + (i0 + 0) * k;
        const float *a1 = a + (i0 + 1) * k;
        const float *a2 = a + (i0 + 2) * k;
        const float *a3 = a + (i0 + 3) * k;

        for (size_t g = 0; g < groups; ++g) {
            float acc[TM][4] = {{0.0f}};

            for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
                const size_t k0 = 4 * (size_t)w->block_row_idx[p];
                const float *w_blk = w->block_values + 16 * p;

                for (size_t r = 0; r < 4; ++r) {
                    /* Load the four weights of block row r once ... */
                    const float w0 = w_blk[4 * r + 0];
                    const float w1 = w_blk[4 * r + 1];
                    const float w2 = w_blk[4 * r + 2];
                    const float w3 = w_blk[4 * r + 3];
                    /* ... and the matching input of each of the four rows. */
                    const float x0 = a0[k0 + r];
                    const float x1 = a1[k0 + r];
                    const float x2 = a2[k0 + r];
                    const float x3 = a3[k0 + r];

                    acc[0][0] += x0 * w0; acc[0][1] += x0 * w1;
                    acc[0][2] += x0 * w2; acc[0][3] += x0 * w3;
                    acc[1][0] += x1 * w0; acc[1][1] += x1 * w1;
                    acc[1][2] += x1 * w2; acc[1][3] += x1 * w3;
                    acc[2][0] += x2 * w0; acc[2][1] += x2 * w1;
                    acc[2][2] += x2 * w2; acc[2][3] += x2 * w3;
                    acc[3][0] += x3 * w0; acc[3][1] += x3 * w1;
                    acc[3][2] += x3 * w2; acc[3][3] += x3 * w3;
                }
            }

            for (size_t t = 0; t < TM; ++t)
                for (size_t col = 0; col < 4; ++col)
                    c[(i0 + t) * n + 4 * g + col] = acc[t][col];
        }
    }

    for (size_t i = m_full; i < m; ++i)
        sparse_row_scalar(a + i * k, w, c + i * n);

    return SPARSE_OK;
}
