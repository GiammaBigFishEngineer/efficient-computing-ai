#define _POSIX_C_SOURCE 200809L
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum { WARMUPS = 20, SAMPLES = 100 };
typedef void (*kernel_fn)(const float *,const float *,float *,size_t,size_t,size_t);
static const size_t m=48,n=48,k=48,tile=8;
static const float *a,*b,*reference;
static float *output;
static double now_seconds(void)
{
    struct timespec ts;
    if(clock_gettime(CLOCK_MONOTONIC,&ts)!=0){perror("clock_gettime");exit(EXIT_FAILURE);}
    return (double)ts.tv_sec+(double)ts.tv_nsec*1e-9;
}
static int compare_double(const void *a_,const void *b_)
{
    double x=*(const double*)a_,y=*(const double*)b_;return (x>y)-(x<y);
}
static void report(const char *name,const double *v)
{
    double sorted[SAMPLES],mean=0.0,var=0.0;
    for(int i=0;i<SAMPLES;i++){sorted[i]=v[i];mean+=v[i]/SAMPLES;}
    qsort(sorted,SAMPLES,sizeof(sorted[0]),compare_double);
    for(int i=0;i<SAMPLES;i++){double d=v[i]-mean;var+=d*d/SAMPLES;}
    printf("%-12s %10.3f %10.3f %10.3f %10.3f\n",name,(sorted[49]+sorted[50])*0.5,sqrt(var),sorted[0],sorted[99]);
}
static int correct(void)
{
    for(size_t i=0;i<m*n;i++)
        if(!isfinite(output[i])||fabsf(output[i]-reference[i])>1e-3f*(1.0f+fabsf(reference[i])))return 0;
    return 1;
}
static void run_kernel(const char *name,kernel_fn fn)
{
    double samples[SAMPLES];
    for(int i=0;i<WARMUPS;i++)fn(a,b,output,m,n,k);
    if(!correct()){fprintf(stderr,"%s correctness check failed\n",name);exit(EXIT_FAILURE);}
    for(int i=0;i<SAMPLES;i++){double start=now_seconds();fn(a,b,output,m,n,k);samples[i]=(now_seconds()-start)*1e6;}
    report(name,samples);
}
int main(void)
{
    float *aa=malloc(m*k*sizeof(*aa)),*bb=malloc(k*n*sizeof(*bb));
    float *ref=calloc(m*n,sizeof(*ref));output=malloc(m*n*sizeof(*output));
    if(!aa||!bb||!ref||!output){fprintf(stderr,"Could not allocate GEMM benchmark arrays\n");free(aa);free(bb);free(ref);free(output);return 2;}
    a=aa;b=bb;reference=ref;
    for(size_t i=0;i<m*k;i++)aa[i]=(float)(i%13)*0.01f;
    for(size_t i=0;i<k*n;i++)bb[i]=(float)(i%7)*0.02f;
    for(size_t i=0;i<m;i++)for(size_t j=0;j<n;j++)
        for(size_t q=0;q<k;q++)ref[i*n+j]+=a[i*k+q]*b[q*n+j];
    puts("GEMM: C = A*B, row-major, M=N=K=48; tiled size=8");
    printf("Warmups: %d, timed samples: %d (microseconds)\n",WARMUPS,SAMPLES);
    puts("Method             median       std        min        max");
    run_kernel("Scalar",gemm_scalar);
    /* Adapt the tiled entry point to the common benchmark function signature. */
    for(int i=0;i<WARMUPS;i++)gemm_tiled(a,b,output,m,n,k,tile);
    if(!correct()){fprintf(stderr,"Tiled GEMM correctness check failed\n");return 1;}
    {double samples[SAMPLES];for(int i=0;i<SAMPLES;i++){double start=now_seconds();gemm_tiled(a,b,output,m,n,k,tile);samples[i]=(now_seconds()-start)*1e6;}report("Tiled T=8",samples);}
    free(aa);free(bb);free(ref);free(output);return 0;
}
