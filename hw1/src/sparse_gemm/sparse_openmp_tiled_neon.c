#include "sparse_gemm.h"
#include <stdio.h>

int gemm_sparse_openmp_tiled_neon(const float *a, const sparse_weights *w,
                                  float *c, size_t m)
{
    (void)a; (void)w; (void)c; (void)m;
    /* TODO: combine the row-tile reuse in gemm_sparse_tiled with the
       four-column NEON update in gemm_sparse_neon. For each independent row
       tile, keep one float32x4_t accumulator per row and output-column group.
       Load each retained 4-value weight row once, then update all rows in the
       four-row tile with vfmaq_f32. Parallelize only the outer row-tile loop
       with a plain '#pragma omp parallel for'. The row tile is fixed at four;
       the benchmark tests thread counts 1/2/4. Use the same packed block
       traversal and ordering as the other kernels. */
    fputs("TODO: implement tiled NEON sparse GEMM with OpenMP\n", stderr);
    return SPARSE_UNIMPLEMENTED;
}
