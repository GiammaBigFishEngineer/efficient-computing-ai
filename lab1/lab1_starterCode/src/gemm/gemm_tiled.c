#include "kernels.h"

void gemm_tiled(const float *a, const float *b, float *c, size_t m, size_t n,
                size_t k, size_t tile) {
  // Tile sizes, available to all the loops below
  // const size_t t_r_a = 2;
  // const size_t t_c_b = 1;
  // const size_t t_s = 3;

  const size_t t_r_a = tile;
  const size_t t_c_b = tile;
  const size_t t_s = tile;

  // Select a group of rows
  for (size_t i0 = 0; i0 < m; i0 += t_r_a) {

    // Select a group of columns
    for (size_t j0 = 0; j0 < n; j0 += t_c_b) {

      // Find this output tile's boundaries
      // height boundary in A
      size_t i_end = i0 + t_r_a;
      if (i_end > m) {
        i_end = m;
      }

      // width boundary in B
      size_t j_end = j0 + t_c_b;
      if (j_end > n) {
        j_end = n;
      }

      // Initialize this output tile ONCE
      for (size_t i = i0; i < i_end; ++i) {
        for (size_t j = j0; j < j_end; ++j) {
          c[i * n + j] = 0.0f;
        }
      }

      // Move through chunks of the shared dimension to define
      // the horizontal tile part of A and vertical tile part of B
      for (size_t el0 = 0; el0 < k; el0 += t_s) {

        // width boundary in A
        // height boundary in B
        size_t el_end = el0 + t_s;
        if (el_end > k) {
          el_end = k;
        }

        // Calculate this chunk's contribution to the output tile
        for (size_t i = i0; i < i_end; ++i) {
          for (size_t j = j0; j < j_end; ++j) {
            float sum = 0.0f;

            for (size_t el = el0; el < el_end; ++el) {
              sum += a[i * k + el] * b[el * n + j];
            }

            // Preserve previous chunks' contributions
            c[i * n + j] += sum;
          }
        }
      }
    }
  }
}
