#include "kernels.h"
#include <math.h>
#include <stdio.h>

static int near(const float *a, const float *b, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        if (fabsf(a[i] - b[i]) > 1e-4f * (1.0f + fabsf(a[i]))) return 0;
    }
    return 1;
}

int main(void)
{
    float a[15], b[20], scalar[12], tiled[12];
    for (int i = 0; i < 15; ++i) a[i] = (float)((i * 3) % 11) * 0.1f;
    for (int i = 0; i < 20; ++i) b[i] = (float)((i * 7) % 13) * 0.07f;
    gemm_scalar(a, b, scalar, 3, 4, 5);
    gemm_tiled(a, b, tiled, 3, 4, 5, 2);
    if (!near(scalar, tiled, 12)) {
        fprintf(stderr, "FAIL: tiled GEMM edge tiles\n");
        return 1;
    }

    float ba[16], bb[12], bc[12], bt[12], bp[12], bpt[12];
    for (int i = 0; i < 16; ++i) ba[i] = (float)(i - 5) * 0.1f;
    for (int i = 0; i < 12; ++i) bb[i] = (float)(i % 5 - 2) * 0.2f;
    gemm_batched_scalar(ba, bb, bc, 2, 2, 3, 4);
    gemm_batched_tiled(ba, bb, bt, 2, 2, 3, 4, 2);
    gemm_batched_omp_batch(ba, bb, bp, 2, 2, 3, 4, 2);
    gemm_batched_omp_tiles(ba, bb, bpt, 2, 2, 3, 4, 2);
    if (!near(bc, bt, 12) || !near(bc, bp, 12) || !near(bc, bpt, 12)) {
        fprintf(stderr, "FAIL: batched GEMM schedules\n");
        return 1;
    }
    puts("GEMM correctness tests passed.");
    return 0;
}
