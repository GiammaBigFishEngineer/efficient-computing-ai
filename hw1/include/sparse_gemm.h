#ifndef SPARSE_GEMM_H
#define SPARSE_GEMM_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t k;
    size_t n;
    size_t nnzb;
    uint32_t *block_col_ptr; /* n/4 + 1 32-bit offsets */
    uint32_t *block_row_idx; /* nnzb 32-bit indices, sorted within each group */
    float *block_values; /* row-major 4x4 values: 16*p + 4*r + c */
} sparse_weights;

enum { SPARSE_OK = 0, SPARSE_UNIMPLEMENTED = 1, SPARSE_ERROR = -1 };

void sparse_weights_destroy(sparse_weights *s);

void gemm_dense_scalar(const float *a, const float *w, float *c,
                       size_t m, size_t k, size_t n);
void gemm_dense_tiled_neon_openmp_TN8_T4(const float *a, const float *w,
                                         float *c, size_t m, size_t k,
                                         size_t n);
int gemm_sparse_scalar(const float *a, const sparse_weights *w, float *c,
                       size_t m);
int gemm_sparse_neon(const float *a, const sparse_weights *w, float *c,
                     size_t m);
int gemm_sparse_tiled(const float *a, const sparse_weights *w, float *c,
                        size_t m);
int gemm_sparse_openmp(const float *a, const sparse_weights *w, float *c,
                       size_t m);
int gemm_sparse_openmp_tiled_neon(const float *a, const sparse_weights *w,
                                  float *c, size_t m);

#endif
