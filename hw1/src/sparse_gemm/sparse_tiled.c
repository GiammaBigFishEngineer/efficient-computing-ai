#include "sparse_gemm.h"
#include <stdio.h>

int gemm_sparse_tiled(const float *a, const sparse_weights *w, float *c,
                        size_t m)
{
    (void)a; (void)w; (void)c; (void)m;
    /* TODO: use a row tile of exactly four input rows. Keep four 4-column
       output accumulators in local storage. For every retained block, load
       its weight values once and reuse them for all four rows. */
    fputs("TODO: implement gemm_sparse_tiled and reuse weights across rows\n", stderr);
    return SPARSE_UNIMPLEMENTED;
}
