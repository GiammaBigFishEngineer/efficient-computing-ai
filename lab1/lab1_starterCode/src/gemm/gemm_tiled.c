#include "kernels.h"

void gemm_tiled(const float *a, const float *b, float *c,
                size_t m, size_t n, size_t k, size_t tile) {
    /* TODO 6 — Tiled GEMM
       Arguments/layout: same A[M,K] * B[K,N] -> C[M,N] as gemm_scalar;
       tile is the positive tile width. Divide the i, j, and reduction q
       ranges into blocks. Clamp the final block in each dimension to M, N,
       or K (for example with min(block_start+tile, dimension)). This keeps
       a smaller working set active and can improve data reuse.
       Pseudocode:
         for ii in blocks of tile over M:
           for jj in blocks of tile over N:
             for qq in blocks of tile over K:
               for i in ii .. min(ii+tile,M):
                 for j in jj .. min(jj+tile,N):
                   for q in qq .. min(qq+tile,K):
                     accumulate a[i*K+q]*b[q*N+j] into c[i*N+j]
       Initialize C to zero before these accumulation loops. Compare tiles
       4, 8, 16, and 32. */
    /* The placeholder does not read these inputs/options yet. The casts
       suppress unused-parameter warnings; remove them when implementing. */
    (void)a;
    (void)b;
    (void)k;
    (void)tile;
    for (size_t i = 0; i < m*n; ++i) {
        c[i] = 0.0f;
    }
}
