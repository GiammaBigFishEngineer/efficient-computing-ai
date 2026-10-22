#define _POSIX_C_SOURCE 200809L
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { WARMUPS = 20, SAMPLES = 100 };
typedef float (*kernel_fn)(const float *,size_t);
static const float *input;
static size_t length;
static float result;
static double now_seconds(void)
{
    struct timespec ts;
    if(clock_gettime(CLOCK_MONOTONIC,&ts)!=0){perror("clock_gettime");exit(EXIT_FAILURE);}
    return (double)ts.tv_sec+(double)ts.tv_nsec*1e-9;
}
static int compare_double(const void *a,const void *b)
{
    double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);
}
static void report(const char *name,const double *v)
{
    double sorted[SAMPLES],mean=0.0,var=0.0;
    for(int i=0;i<SAMPLES;i++){sorted[i]=v[i];mean+=v[i]/SAMPLES;}
    qsort(sorted,SAMPLES,sizeof(sorted[0]),compare_double);
    for(int i=0;i<SAMPLES;i++){double d=v[i]-mean;var+=d*d/SAMPLES;}
    printf("%-16s %10.3f %10.3f %10.3f %10.3f\n",name,(sorted[49]+sorted[50])*0.5,sqrt(var),sorted[0],sorted[99]);
}
static void run_kernel(const char *name,kernel_fn fn,double reference,double magnitude)
{
    double samples[SAMPLES];
    for(int i=0;i<WARMUPS;i++)result=fn(input,length);
    if(!isfinite(result)||fabs((double)result-reference)>1e-5*(1.0+magnitude)){
        fprintf(stderr,"%s correctness check failed (got %.9g, expected %.9g)\n",name,result,reference);exit(EXIT_FAILURE);
    }
    for(int i=0;i<SAMPLES;i++){double start=now_seconds();result=fn(input,length);samples[i]=(now_seconds()-start)*1e6;}
    report(name,samples);
}
int main(void)
{
    length=1u<<20;
    float *x=malloc(length*sizeof(*x));
    if(!x){fprintf(stderr,"Could not allocate reduce-sum input\n");return 2;}
    input=x;
    double reference=0.0,magnitude=0.0;
    for(size_t i=0;i<length;i++){x[i]=(float)((int)(i%31)-15)*0.01f;reference+=x[i];magnitude+=fabs((double)x[i]);}
    printf("Sum of %zu FP32 elements\n",length);
    printf("Warmups: %d, timed samples: %d (microseconds; population std dev)\n",WARMUPS,SAMPLES);
    puts("Method                   median       std        min        max");
    run_kernel("Scalar",reduce_sum_scalar,reference,magnitude);
    run_kernel("NEON",reduce_sum_neon,reference,magnitude);
    free(x);return 0;
}
