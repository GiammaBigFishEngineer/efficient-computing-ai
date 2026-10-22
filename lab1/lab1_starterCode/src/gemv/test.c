#include "kernels.h"
#include <math.h>
#include <stdio.h>

int main(void)
{
    float matrix[15], x[5], scalar[3], neon[3];
    for (int i = 0; i < 15; ++i) matrix[i] = (float)(i % 7 - 3) * 0.2f;
    for (int i = 0; i < 5; ++i) x[i] = (float)i * 0.3f;
    gemv_scalar(matrix, x, scalar, 3, 5);
    gemv_neon(matrix, x, neon, 3, 5);
    for (int i = 0; i < 3; ++i) {
        if (fabsf(scalar[i] - neon[i]) > 1e-4f * (1.0f + fabsf(scalar[i]))) {
            fprintf(stderr, "FAIL: GEMV with tail at row %d\n", i);
            return 1;
        }
    }
    puts("GEMV correctness test passed.");
    return 0;
}
