#include "sparse_gemm.h"

int gemm_sparse_scalar(const float *a, const sparse_weights *w, float *c,
                       size_t m)
{
    if (!a || !w || !c)
        return SPARSE_ERROR;
    for (size_t i = 0; i < m; ++i)
        for (size_t g = 0; g < w->n / 4; ++g) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
                size_t k0 = 4 * (size_t)w->block_row_idx[p];
                for (size_t r = 0; r < 4; ++r) {
                    float av = a[i * w->k + k0 + r];
                    for (size_t col = 0; col < 4; ++col)
                        acc[col] += av * w->block_values[16 * p + 4 * r + col];
                }
            }
            for (size_t col = 0; col < 4; ++col) c[i * w->n + 4 * g + col] = acc[col];
        }
    return SPARSE_OK;
}
