#include "kernels.h"

void gemm_scalar(const float *a, const float *b, float *c, size_t m, size_t n,
                 size_t k) {
  for (size_t i = 0; i < m; ++i) {

    for (size_t j = 0; j < n; ++j) {
      float sum = 0.0f;

      for (size_t el = 0; el < k; ++el) {
        sum += a[i * k + el] * b[el * n + j];
      }

      c[i * n + j] = sum;
    }
  }
}
