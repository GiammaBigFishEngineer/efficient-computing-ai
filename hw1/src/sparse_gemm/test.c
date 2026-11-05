#include "case_data.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef int (*sparse_fn)(const float *, const sparse_weights *, float *, size_t);
static int errors;
static int skipped;

static int near(const float *x, const float *y, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(x[i]) || fabsf(x[i] - y[i]) >
            2.0e-4f * (1.0f + fabsf(y[i]))) return 0;
    return 1;
}

static void check_sparse(const char *name, sparse_fn fn, const float *a,
                         const sparse_weights *w, float *out,
                         const float *reference)
{
    int rc = fn(a, w, out, HW1_M);
    if (rc == SPARSE_UNIMPLEMENTED) {
        ++skipped;
        printf("SKIP %s: TODO not implemented yet\n", name);
    } else if (rc != SPARSE_OK || !near(out, reference, HW1_M * HW1_N)) {
        fprintf(stderr, "FAIL %s (status=%d)\n", name, rc);
        ++errors;
    } else {
        printf("PASS %s\n", name);
    }
}

int main(void)
{
    float *a = malloc(HW1_M * HW1_K * sizeof(*a));
    float *dense_w = malloc(HW1_K * HW1_N * sizeof(*dense_w));
    float *reference = malloc(HW1_M * HW1_N * sizeof(*reference));
    float *out = malloc(HW1_M * HW1_N * sizeof(*out));
    sparse_weights w = {0};
    if (!a || !dense_w || !reference || !out ||
        sparse_case_load_fixed(a, dense_w, &w) != SPARSE_OK) {
        fputs("FAIL: fixed test-case setup/allocation\n", stderr);
        return 2;
    }
    if (w.nnzb != HW1_NNZB) {
        fputs("FAIL: fixed packed weights have wrong density or layout\n", stderr);
        return 1;
    }

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(4);
#endif
    gemm_dense_scalar(a, dense_w, reference, HW1_M, HW1_K, HW1_N);
    gemm_dense_tiled_neon_openmp_TN8_T4(a, dense_w, out,
                                        HW1_M, HW1_K, HW1_N);
    if (!near(out, reference, HW1_M * HW1_N)) {
        fputs("FAIL dense_tiled_neon_openmp_TN8_T4\n", stderr);
        ++errors;
    } else {
        puts("PASS dense_tiled_neon_openmp_TN8_T4");
    }

    printf("Case M=%d K=%d N=%d density=0.25 nnzb=%zu\n",
           HW1_M, HW1_K, HW1_N, w.nnzb);
    check_sparse("sparse_scalar", gemm_sparse_scalar, a, &w, out, reference);
    check_sparse("sparse_neon", gemm_sparse_neon, a, &w, out, reference);
    check_sparse("sparse_tiled_TM4", gemm_sparse_tiled, a, &w, out, reference);

    const int thread_counts[] = {1, 2, 4};
    for (size_t i = 0; i < sizeof(thread_counts) / sizeof(thread_counts[0]); ++i) {
        int threads = thread_counts[i];
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        char label[64];
        snprintf(label, sizeof(label), "sparse_openmp_T%d", threads);
        check_sparse(label, gemm_sparse_openmp, a, &w, out, reference);
        snprintf(label, sizeof(label), "sparse_openmp_tiled_neon_TM4_T%d", threads);
        check_sparse(label, gemm_sparse_openmp_tiled_neon,
                     a, &w, out, reference);
    }

    if (errors) return 1;
    if (skipped)
        printf("Completed implementations pass; %d TODO kernel checks skipped.\n", skipped);
    else
        puts("All fixed-case kernel tests passed.");
    sparse_weights_destroy(&w);
    free(a); free(dense_w); free(reference); free(out);
    return 0;
}
