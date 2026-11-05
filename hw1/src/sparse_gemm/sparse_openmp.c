#include "sparse_gemm.h"
#include <stdio.h>

int gemm_sparse_openmp(const float *a, const sparse_weights *w, float *c, size_t m)
{
    (void)a; (void)w; (void)c; (void)m;
    /* TODO: parallelize only the independent output-row loop. Each iteration
       must own a distinct row of C and compute that row's 4-column groups. */
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < m; ++i) {
        /* TODO: compute output row i with the scalar sparse loop. */
        (void)i;
    }
    fputs("TODO: implement gemm_sparse_openmp.\n", stderr);
    return SPARSE_UNIMPLEMENTED;
}
