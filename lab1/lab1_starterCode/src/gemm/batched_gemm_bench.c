#define _POSIX_C_SOURCE 200809L
#include "kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*bench_fn)(void);
static const float *a, *b;
static float *out;
static size_t batch, m, n, k, tile;
static volatile float sink;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void run_scalar(void) { gemm_batched_scalar(a,b,out,batch,m,n,k); }
static void run_tiled(void) { gemm_batched_tiled(a,b,out,batch,m,n,k,tile); }
static void run_batch_omp(void) { gemm_batched_omp_batch(a,b,out,batch,m,n,k,tile); }
static void run_tiles_omp(void) { gemm_batched_omp_tiles(a,b,out,batch,m,n,k,tile); }

static double time_us(bench_fn fn) {
    const int reps = 3;
    for (int i=0; i<2; ++i) fn();
    double start = now_seconds();
    for (int i=0; i<reps; ++i) fn();
    sink = out[(batch*m*n)-1];
    return (now_seconds()-start)*1e6/reps;
}

static int matches(const float *x, const float *y, size_t count) {
    for (size_t i=0; i<count; ++i)
        if (fabsf(x[i]-y[i]) > 2e-3f*(1.0f+fabsf(x[i]))) return 0;
    return 1;
}

int main(void) {
    const size_t sizes[] = {1, 4, 16, 64};
    m = n = k = 64;
    tile = 16;
    float *weights = malloc(k*n*sizeof(*weights));
    if (!weights) return 2;
    for (size_t i=0; i<k*n; ++i) weights[i] = (float)((int)(i%17)-8)*0.01f;
    b = weights;

    puts("Batched GEMM: A[B,M,K] * W[K,N] -> Y[B,M,N], FP32, M=N=K=64, tile=16");
    puts("B | method                  | latency (us) | GFLOP/s | speedup vs scalar");
    for (size_t si=0; si<sizeof(sizes)/sizeof(sizes[0]); ++si) {
        batch = sizes[si];
        float *input = malloc(batch*m*k*sizeof(*input));
        float *reference = malloc(batch*m*n*sizeof(*reference));
        out = malloc(batch*m*n*sizeof(*out));
        if (!input || !reference || !out) return 2;
        for (size_t i=0; i<batch*m*k; ++i) input[i] = (float)((int)(i%23)-11)*0.01f;
        a = input;

        run_scalar();
        gemm_batched_tiled(a,b,reference,batch,m,n,k,tile);
        if (!matches(out, reference, batch*m*n)) {
            fprintf(stderr, "tiled GEMM correctness failed at B=%zu\n", batch);
            return 1;
        }
        double baseline = time_us(run_scalar);
        bench_fn methods[] = {run_tiled, run_batch_omp, run_tiles_omp};
        const char *names[] = {"tiled serial", "OpenMP over batch", "OpenMP over tiles"};
        double macs = (double)batch*m*n*k;
        for (size_t mi=0; mi<3; ++mi) {
            methods[mi]();
            if (!matches(reference,out,batch*m*n)) {
                fprintf(stderr, "%s correctness failed at B=%zu\n", names[mi], batch);
                return 1;
            }
            double us = time_us(methods[mi]);
            printf("%zu | %-23s | %12.2f | %7.3f | %8.2fx\n",
                   batch,names[mi],us,2.0*macs/(us*1e3),baseline/us);
        }
        printf("%zu | %-23s | %12.2f | %7.3f | %8.2fx\n",
               batch,"scalar baseline",baseline,2.0*macs/(baseline*1e3),1.0);
        free(input); free(reference); free(out);
    }
    free(weights);
    return 0;
}
