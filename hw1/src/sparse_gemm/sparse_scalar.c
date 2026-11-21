#include "sparse_gemm.h"

int gemm_sparse_scalar(const float *a, const sparse_weights *w, float *c,
                       size_t m) {
  // check if pointers are null
  if (!a || !w || !c)
    return SPARSE_ERROR;

  // select input row
  for (size_t i = 0; i < m; ++i)
    // select group and initialise the first 4 output sums to zero

    for (size_t g = 0; g < w->n / 4; ++g) {
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

      // for the selected block obtain its start index and end index (excluded)
      // iterate from start to stop --> p: the subgroup index
      // use p to obtain the subgroup index row and compute its starting row k0
      for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
        size_t k0 = 4 * (size_t)w->block_row_idx[p];

        // iterate on the rows of the selected subgroup AND obtain the
        // respective inputs
        for (size_t r = 0; r < 4; ++r) {
          // obtain the respective input value: skip input rows (i*k), move to
          // input column k0, then select column k0+r from the weight subgroup
          float av = a[i * w->k + k0 + r];

          // the input value is a single value
          // access single values for the weight subgroup row too
          // multiply the values and keep the cumulative sums
          for (size_t col = 0; col < 4; ++col)
            acc[col] += av * w->block_values[16 * p + 4 * r + col];
        }
      }

      // before moving to the next group write the results in c
      for (size_t col = 0; col < 4; ++col)
        c[i * w->n + 4 * g + col] = acc[col];
    }
  return SPARSE_OK;
}
