#include "kernels.h"
#include <arm_neon.h>

void gemv_neon(const float *a, const float *x, float *y, size_t m, size_t k) {
  // consider one row of A
  for (size_t i = 0; i < m; ++i) {

    size_t el = 0;
    float32x4_t sum = vdupq_n_f32(0.0f);

    // multiply the selected row of A for all values in X
    // perform the multiplication 4 elements at a time and sum them together
    // VERTICALLY to obtain partial sums keep summing the partial sums until
    // there aren't 4 values anymore, then sum HORIZONTALLY to obtain the total
    // sum of the selcted row if there are still values left that are not enough
    // for the register --> scalar GEMV
    while (el + 4 <= k) {

      float32x4_t weights = vld1q_f32(&a[i * k + el]);
      // if you use the pointers indexing it makes a lot of
      // sense, starting from the first value pointers skip
      // i rows and select the el element

      float32x4_t inputs = vld1q_f32(&x[el]);

      float32x4_t mult = vmulq_f32(weights, inputs);
      sum = vaddq_f32(sum, mult);

      el += 4;
    }

    // when exiting the while we are still in the same row i
    // there could be additional values left out because not divisible by 4
    // do normal scalar GEMV
    // while for the 4 partial sums kept until now, merge them into a single
    // scalar
    float sum_scalar = vaddvq_f32(sum);

    while (el < k) {
      sum_scalar += a[i * k + el] * x[el];
      ++el;
    }

    // write the output before changing row
    y[i] = sum_scalar;
  }
}
