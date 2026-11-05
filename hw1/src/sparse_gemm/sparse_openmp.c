#include "sparse_gemm.h"

/*
 * 4x4 block-sparse GEMM, scalar inner loops, OpenMP over output rows.
 *
 * Every iteration i owns row i of C exclusively and only reads A and the
 * shared (read-only) sparse weights, so the rows are independent and no
 * synchronisation is required. No tiling and no NEON in this version.
 * The thread count is set by the caller (omp_set_num_threads).
 */
int gemm_sparse_openmp(const float *a, const sparse_weights *w, float *c, size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;

    const size_t k = w->k;
    const size_t n = w->n;
    const size_t groups = n / 4;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < m; ++i) {
        const float *a_row = a + i * k;
        float *c_row = c + i * n;

        for (size_t g = 0; g < groups; ++g) {
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
    return SPARSE_OK;
}
