#define _POSIX_C_SOURCE 200809L
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { WARMUPS = 20, SAMPLES = 100 };
typedef void (*kernel_fn)(const float *, const float *, float *, size_t, size_t);
static const size_t rows = 64, columns = 64;
static const float *matrix, *vector;
static float *output;

static double now_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) { perror("clock_gettime"); exit(EXIT_FAILURE); }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int compare_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static void report(const char *name, const double *v)
{
    double sorted[SAMPLES], mean = 0.0, variance = 0.0;
    for (int i = 0; i < SAMPLES; ++i) { sorted[i] = v[i]; mean += v[i] / SAMPLES; }
    qsort(sorted, SAMPLES, sizeof(sorted[0]), compare_double);
    for (int i = 0; i < SAMPLES; ++i) { double d = v[i] - mean; variance += d*d/SAMPLES; }
    printf("%-12s %10.3f %10.3f %10.3f %10.3f\n", name,
           (sorted[49]+sorted[50])*0.5, sqrt(variance), sorted[0], sorted[99]);
}
static int correct(void)
{
    for (size_t i = 0; i < rows; ++i) {
        float ref = 0.0f;
        for (size_t j = 0; j < columns; ++j) ref += matrix[i*columns+j] * vector[j];
        if (!isfinite(output[i]) || fabsf(output[i]-ref) > 1e-3f*(1.0f+fabsf(ref))) return 0;
    }
    return 1;
}
static void run_kernel(const char *name, kernel_fn fn)
{
    double samples[SAMPLES];
    for (int i=0; i<WARMUPS; ++i) fn(matrix,vector,output,rows,columns);
    if (!correct()) { fprintf(stderr,"%s correctness check failed\n",name); exit(EXIT_FAILURE); }
    for (int i=0; i<SAMPLES; ++i) {
        double start=now_seconds(); fn(matrix,vector,output,rows,columns);
        samples[i]=(now_seconds()-start)*1e6;
    }
    report(name,samples);
}
int main(void)
{
    float *a=malloc(rows*columns*sizeof(*a)), *x=malloc(columns*sizeof(*x));
    output=malloc(rows*sizeof(*output));
    if (!a||!x||!output) { fprintf(stderr,"Could not allocate GEMV benchmark arrays\n"); free(a);free(x);free(output);return 2; }
    matrix=a;vector=x;
    for(size_t i=0;i<rows*columns;i++) a[i]=(float)(i%13)*0.01f;
    for(size_t i=0;i<columns;i++) x[i]=(float)(i%7)*0.02f;
    puts("GEMV: y = A*x, row-major A[M,K], M=K=64");
    printf("Warmups: %d, timed samples: %d (microseconds)\n",WARMUPS,SAMPLES);
    puts("Method             median       std        min        max");
    run_kernel("Scalar",gemv_scalar);run_kernel("NEON",gemv_neon);
    free(a);free(x);free(output);return 0;
}
