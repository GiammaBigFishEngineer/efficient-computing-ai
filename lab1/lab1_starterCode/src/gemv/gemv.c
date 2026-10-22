#include "kernels.h"

void gemv_scalar(const float *a, const float *x, float *y, size_t m, size_t k) {
  for (size_t i = 0; i < m; ++i) {

    float sum = 0.0f;

    for (size_t el = 0; el < k; ++el) {
      sum += a[i * k + el] * x[el];
    }

    y[i] = sum;
  }
}
