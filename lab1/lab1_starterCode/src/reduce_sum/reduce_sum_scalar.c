#include "kernels.h"

float reduce_sum_scalar(const float *x, size_t n) {
  float tot_sum = 0.0f;
  size_t i = 0;

  while (i < n) {
    tot_sum += x[i];
    ++i;
  }

  return tot_sum;
}
