#include "kernels.h"
#include <math.h>
#include <stdio.h>

int main(void) {
  float a[11], b[11], scalar[11], neon[11];
  for (int i = 0; i < 11; ++i) {
    a[i] = (float)i * 0.25f;
    b[i] = (float)(10 - i) * 0.5f;
  }

  vector_add_scalar(a, b, scalar, 11);
  vector_add_neon(a, b, neon, 11);

  for (int i = 0; i < 11; ++i) {
    if (fabsf(scalar[i] - neon[i]) > 1e-4f * (1.0f + fabsf(scalar[i]))) {
      fprintf(stderr, "FAIL: vector add and tail at index %d\n", i);
      return 1;
    }
  }
  puts("Vector-add correctness test passed.");
  return 0;
}
