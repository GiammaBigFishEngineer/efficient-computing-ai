#define _POSIX_C_SOURCE 200809L
#include "case_data.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

enum { TARGET_SAMPLE_US = 20000, MAX_REPETITIONS = 100000,
       SAMPLE_COUNT = 5, WARMUP_COUNT = 2 };

typedef int (*sparse_kernel)(const float *, const sparse_weights *, float *, size_t);
typedef void (*dense_kernel)(const float *, const float *, float *, size_t, size_t, size_t);

typedef struct {
    const char *name;
    int threads;
    int tile_m;
    dense_kernel dense;
    sparse_kernel sparse;
} kernel_case;

static const float *ga, *gw;
static const sparse_weights *gs;
static float *gc;
static volatile float observable;

static double now_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int compare_floats(const float *x, const float *y, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        if (!isfinite(x[i]) || fabsf(x[i] - y[i]) >
            2.0e-4f * (1.0f + fabsf(y[i]))) return 0;
    return 1;
}

static int compare_doubles(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int run_case(const kernel_case *item)
{
    if (item->dense) {
        item->dense(ga, gw, gc, HW1_M, HW1_K, HW1_N);
        return SPARSE_OK;
    }
    return item->sparse(ga, gs, gc, HW1_M);
}

static void set_threads(int threads)
{
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    if (omp_get_max_threads() != threads) {
        fprintf(stderr, "OpenMP setup mismatch: requested %d, got %d\n",
                threads, omp_get_max_threads());
        exit(2);
    }
#else
    (void)threads;
#endif
}

static void write_row(FILE *out, const char *run_id, const kernel_case *item,
                      const char *status, const double *samples, size_t count,
                      size_t repetitions)
{
    double median = 0.0, minimum = 0.0, maximum = 0.0, variation = 0.0;
    if (samples && count) {
        double sorted[SAMPLE_COUNT];
        memcpy(sorted, samples, count * sizeof(*samples));
        qsort(sorted, count, sizeof(*sorted), compare_doubles);
        median = sorted[count / 2];
        minimum = sorted[0];
        maximum = sorted[count - 1];
        variation = median ? (maximum - minimum) / median : 0.0;
    }

    fprintf(out, "%s,%s,%d,%d,%s,%d,%zu,%zu,",
            run_id, item->name, item->threads, item->tile_m, status,
            WARMUP_COUNT, count, repetitions);
    if (count)
        fprintf(out, "%.6f,%.6f,%.6f,%.6f\n",
                median, minimum, maximum, variation);
    else
        fputs(",,,,\n", out);
}

static int benchmark_one(FILE *out, const char *run_id,
                         const kernel_case *item, const float *reference)
{
    set_threads(item->threads);
    int rc = run_case(item);
    if (rc == SPARSE_UNIMPLEMENTED) {
        fprintf(stderr, "TODO / unavailable: %s\n", item->name);
        write_row(out, run_id, item, "TODO", NULL, 0, 0);
        return 0;
    }
    if (rc != SPARSE_OK || !compare_floats(gc, reference, HW1_M * HW1_N)) {
        fprintf(stderr, "FAIL correctness; not timing %s\n", item->name);
        write_row(out, run_id, item, "FAIL", NULL, 0, 0);
        return 1;
    }

    for (int warmup = 0; warmup < WARMUP_COUNT; ++warmup)
        (void)run_case(item);

    double start = now_seconds();
    (void)run_case(item);
    double single_us = (now_seconds() - start) * 1.0e6;
    size_t repetitions = single_us > TARGET_SAMPLE_US ? 1 :
        (size_t)(TARGET_SAMPLE_US / fmax(single_us, 1.0));
    if (!repetitions) repetitions = 1;
    if (repetitions > MAX_REPETITIONS) repetitions = MAX_REPETITIONS;

    double samples[SAMPLE_COUNT];
    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        start = now_seconds();
        for (size_t r = 0; r < repetitions; ++r) (void)run_case(item);
        samples[sample] = (now_seconds() - start) * 1.0e6 / repetitions;
        observable += gc[HW1_M * HW1_N - 1];
    }
    write_row(out, run_id, item, "pass", samples, SAMPLE_COUNT, repetitions);
    return 0;
}

int main(int argc, char **argv)
{
    const char *output_path = "results_raw.csv";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--output") && i + 1 < argc)
            output_path = argv[++i];
        else {
            fprintf(stderr, "Usage: %s [--output file.csv]\n", argv[0]);
            return 2;
        }
    }

    float *a = malloc(HW1_M * HW1_K * sizeof(*a));
    float *dense_w = malloc(HW1_K * HW1_N * sizeof(*dense_w));
    float *reference = malloc(HW1_M * HW1_N * sizeof(*reference));
    gc = malloc(HW1_M * HW1_N * sizeof(*gc));
    sparse_weights weights = {0};
    if (!a || !dense_w || !reference || !gc) {
        fputs("fixed-case allocation failed\n", stderr);
        return 2;
    }
    if (sparse_case_load_fixed(a, dense_w, &weights) != SPARSE_OK) {
        fputs("could not load the hard-coded case\n", stderr);
        return 2;
    }
    ga = a; gw = dense_w; gs = &weights;
    gemm_dense_scalar(a, dense_w, reference, HW1_M, HW1_K, HW1_N);

    static const kernel_case cases[] = {
        {"dense_scalar", 1, 0, gemm_dense_scalar, NULL},
        {"dense_tiled_neon_openmp_TN8_T4", 4, 0,
         gemm_dense_tiled_neon_openmp_TN8_T4, NULL},
        {"sparse_scalar", 1, 0, NULL, gemm_sparse_scalar},
        {"sparse_neon", 1, 0, NULL, gemm_sparse_neon},
        {"sparse_tiled_TM4", 1, 4, NULL, gemm_sparse_tiled},
        {"sparse_openmp_T1", 1, 0, NULL, gemm_sparse_openmp},
        {"sparse_openmp_T2", 2, 0, NULL, gemm_sparse_openmp},
        {"sparse_openmp_T4", 4, 0, NULL, gemm_sparse_openmp},
        {"sparse_openmp_tiled_neon_TM4_T1", 1, 4, NULL,
         gemm_sparse_openmp_tiled_neon},
        {"sparse_openmp_tiled_neon_TM4_T2", 2, 4, NULL,
         gemm_sparse_openmp_tiled_neon},
        {"sparse_openmp_tiled_neon_TM4_T4", 4, 4, NULL,
         gemm_sparse_openmp_tiled_neon}
    };

    FILE *out = fopen(output_path, "w");
    if (!out) { perror(output_path); return 2; }
    fputs("run_id,kernel,threads,tile_m,status,warmups,timed_samples,repetitions,"
          "latency_median_us,latency_min_us,latency_max_us,latency_variation\n", out);

    time_t current = time(NULL);
    struct tm local;
    localtime_r(&current, &local);
    char run_id[32];
    strftime(run_id, sizeof(run_id), "%Y%m%dT%H%M%S", &local);

    int failures = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
        failures += benchmark_one(out, run_id, &cases[i], reference);
    fclose(out);
    fprintf(stderr, "Latency results written to %s\n", output_path);

    sparse_weights_destroy(&weights);
    free(a); free(dense_w); free(reference); free(gc);
    return failures ? 1 : 0;
}
