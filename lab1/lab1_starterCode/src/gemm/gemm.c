#include "kernels.h"

void gemm_scalar(const float *a, const float *b, float *c,
                 size_t m, size_t n, size_t k) {
    /* TODO 5 — Scalar GEMM
       Arguments: a is row-major A[M,K], b is row-major B[K,N], and c is the
       writable row-major result C[M,N]. Compute C[i,j] as the K-term dot
       product of row i of A and column j of B. Initialize each sum to zero.
       Pseudocode:
         for i = 0 .. M-1:
           for j = 0 .. N-1:
             sum = 0
             for q = 0 .. K-1: sum += a[i*K+q] * b[q*N+j]
             c[i*N+j] = sum */
    /* The placeholder does not read these inputs/dimensions yet. The casts
       suppress unused-parameter warnings; remove them when implementing. */
    (void)a;
    (void)b;
    (void)k;
    for (size_t i = 0; i < m*n; ++i) {
        c[i] = 0.0f;
    }
}
