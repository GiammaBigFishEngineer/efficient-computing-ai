#include "kernels.h"
#include <math.h>
#include <stdio.h>

static int close_enough(float a, float b)
{
    return fabsf(a - b) <= 1e-4f * (1.0f + fabsf(a));
}

static int check_length(const float *x, size_t n)
{
    const float reference = reduce_sum_scalar(x, n);
    if (!close_enough(reference, reduce_sum_neon(x, n))) {
        fprintf(stderr, "FAIL: SIMD reduction for n=%zu\n", n);
        return 0;
    }
    return 1;
}

int main(void)
{
    float values[1031];
    for (size_t i = 0; i < 1031; ++i) {
        values[i] = (float)((int)(i % 19) - 9) * 0.125f;
    }
    if (!check_length(values, 0) || !check_length(values, 1) ||
        !check_length(values, 7) || !check_length(values, 16) ||
        !check_length(values, 1031)) {
        return 1;
    }
    puts("Reduce-sum correctness tests passed.");
    return 0;
}
