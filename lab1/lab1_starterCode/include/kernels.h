#ifndef KERNELS_H
#define KERNELS_H
#include <stddef.h>

void vector_add_scalar(const float *a, const float *b, float *y, size_t n);
void vector_add_neon(const float *a, const float *b, float *y, size_t n);
float reduce_sum_scalar(const float *x, size_t n);
float reduce_sum_neon(const float *x, size_t n);
void gemv_scalar(const float *a, const float *x, float *y, size_t m, size_t k);
void gemv_neon(const float *a, const float *x, float *y, size_t m, size_t k);
void gemm_scalar(const float *a, const float *b, float *c, size_t m, size_t n, size_t k);
void gemm_neon(const float *a, const float *b, float *out, size_t m, size_t n, size_t k);
void gemm_tiled(const float *a, const float *b, float *c, size_t m, size_t n, size_t k, size_t tile);
/* Batched row-major GEMM: A[batch,m,k] * B[k,n] -> C[batch,m,n]. */
void gemm_batched_scalar(const float *a, const float *b, float *c, size_t batch, size_t m, size_t n, size_t k);
void gemm_batched_tiled(const float *a, const float *b, float *c, size_t batch, size_t m, size_t n, size_t k, size_t tile);
void gemm_batched_omp_batch(const float *a, const float *b, float *c, size_t batch, size_t m, size_t n, size_t k, size_t tile);
void gemm_batched_omp_tiles(const float *a, const float *b, float *c, size_t batch, size_t m, size_t n, size_t k, size_t tile);

#endif
