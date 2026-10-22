#include "kernels.h"
#include <arm_neon.h>

float reduce_sum_neon(const float *x, size_t n) {
  float32x4_t acc = vdupq_n_f32(0.0f);
  size_t i = 0;

  while (i + 4 <= n) {
    float32x4_t values = vld1q_f32(x + i); // load 4 values
    acc = vaddq_f32(acc, values);          // and sum them VERTICALLY

    i += 4;
  }

  float sum = vaddvq_f32(acc);

  // if the number of elements in the array is not a multiple of 4 there are
  // values left --> handle them with normal scalar sum
  while (i < n) {
    sum += x[i];
    ++i;
  }

  return sum;
}
