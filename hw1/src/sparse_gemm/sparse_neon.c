#include "sparse_gemm.h"

#include <arm_neon.h>

/*
 * 4x4 block-sparse GEMM, one output row at a time, NEON over the 4 columns
 * of each output-column group.
 *
 * For every (row i, group g) pair one float32x4_t accumulator holds
 * C[i, 4g .. 4g+3]. Each stored block p contributes four rank-1 updates:
 * for local row r, the scalar A[i, 4q + r] is broadcast to all four lanes
 * (vdupq_n_f32) and multiplied by the four adjacent block values
 * block_values[16p + 4r .. 16p + 4r + 3] (vld1q_f32), fused with vfmaq_f32.
 * The block traversal order (p, then r) is the same as gemm_sparse_scalar.
 */

int gemm_sparse_neon(const float *a, const sparse_weights *w, float *c,
                     size_t m) {
  // check pointers are not null
  if (!a || !w || !c)
    return SPARSE_ERROR;

  // access to the "attributes" k and n of the sparse_weights structure
  const size_t k = w->k;
  const size_t n = w->n;
  const size_t groups = n / 4;

  // iterate on input rows
  for (size_t i = 0; i < m; ++i) {
    // points to the position of the current row i --> *a_row is where to read
    // each input row produces one output row --> *c_row is where to write
    const float *a_row = a + i * k;
    float *c_row = c + i * n;

    // iterate on groups and initialise the 4 NEON accumulator lanes to zero
    // one running output sum per lane
    for (size_t g = 0; g < groups; ++g) {
      float32x4_t acc = vdupq_n_f32(0.0f);

      // for each group access the start and end (excluded) index
      // iterate from start to end --> p: is the subgroup idx
      for (size_t p = w->block_col_ptr[g]; p < w->block_col_ptr[g + 1]; ++p) {
        // acccess subgroup row idx and obtain its first row idx
        // locate the matching input values
        // w_blk points to the first of 16 weights
        const float *a_blk = a_row + 4 * (size_t)w->block_row_idx[p];
        const float *w_blk = w->block_values + 16 * p;

        // duplicate the first input value
        // load the first weight row: its first value and the next 3
        // multiply matching lanes and add to the 4 separate sums in acc
        acc = vfmaq_f32(acc, vdupq_n_f32(a_blk[0]), vld1q_f32(w_blk + 0));
        acc = vfmaq_f32(acc, vdupq_n_f32(a_blk[1]), vld1q_f32(w_blk + 4));
        acc = vfmaq_f32(acc, vdupq_n_f32(a_blk[2]), vld1q_f32(w_blk + 8));
        acc = vfmaq_f32(acc, vdupq_n_f32(a_blk[3]), vld1q_f32(w_blk + 12));
      }

      // store the 4 completed output values before moving to the next group
      vst1q_f32(c_row + 4 * g, acc);
    }
  }
  return SPARSE_OK;
}
