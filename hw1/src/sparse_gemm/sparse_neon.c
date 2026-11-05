#include "sparse_gemm.h"
#include <stdio.h>

int gemm_sparse_neon(const float *a, const sparse_weights *w, float *c, size_t m)
{
    (void)a; (void)w; (void)c; (void)m;
    /* TODO: for each output row and 4-column group, keep a float32x4_t
       accumulator. For each stored block/reduction row, broadcast A with
       vdupq_n_f32, load four adjacent block values with vld1q_f32, and use
       vfmaq_f32. Store once with vst1q_f32. See gemm_sparse_scalar for the
       compressed-column traversal and preserve the same block ordering. */
    fputs("TODO: implement gemm_sparse_neon with four adjacent FP32 lanes\n", stderr);
    return SPARSE_UNIMPLEMENTED;
}
