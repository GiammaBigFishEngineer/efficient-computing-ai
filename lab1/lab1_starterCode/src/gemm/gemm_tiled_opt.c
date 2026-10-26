#include "kernels.h"
#include <arm_neon.h>

// tile + NEON

void gemm_tiled(const float *a, const float *b, float *c, size_t m, size_t n,
                size_t k, size_t tile) {

  // Select an output tile.
  for (size_t i0 = 0; i0 < m; i0 += tile) {
    for (size_t j0 = 0; j0 < n; j0 += tile) {

      // Clamp the boundaries for smaller edge tiles.
      size_t i_end = i0 + tile;
      if (i_end > m) {
        i_end = m;
      }

      size_t j_end = j0 + tile;
      if (j_end > n) {
        j_end = n;
      }

      // Initialize this output tile once.
      for (size_t i = i0; i < i_end; ++i) {
        for (size_t j = j0; j < j_end; ++j) {
          c[i * n + j] = 0.0f;
        }
      }

      // Visit chunks of the shared dimension.
      for (size_t el0 = 0; el0 < k; el0 += tile) {

        size_t el_end = el0 + tile;
        if (el_end > k) {
          el_end = k;
        }

        for (size_t i = i0; i < i_end; ++i) {
          size_t j = j0;

          // Update four adjacent output columns together.
          while (j + 4 <= j_end) {

            // Preserve contributions from previous chunks.
            float32x4_t acc = vld1q_f32(&c[i * n + j]);

            for (size_t el = el0; el < el_end; ++el) {

              // Repeat one A value across four lanes.
              float32x4_t av = vdupq_n_f32(a[i * k + el]);

              // Load four adjacent values from row el of B.
              float32x4_t bv = vld1q_f32(&b[el * n + j]);

              // Four output updates: acc += av * bv.
              acc = vfmaq_f32(acc, av, bv);
            }

            // Store the four updated outputs.
            vst1q_f32(&c[i * n + j], acc);

            j += 4;
          }

          // Process any remaining columns individually.
          while (j < j_end) {
            float sum = 0.0f;

            for (size_t el = el0; el < el_end; ++el) {
              sum += a[i * k + el] * b[el * n + j];
            }

            c[i * n + j] += sum;
            ++j;
          }
        }
      }
    }
  }
}