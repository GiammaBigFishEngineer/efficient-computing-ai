#define _POSIX_C_SOURCE 200809L
#include "kernels.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// we initially iterate on warmups which defines the number of loops in which we
// execute the function and discard the timings --> we don't measure anything as
// data may not yet be in cache

// we actually record measures iterating on
// samples, which provide multiple timings because execution time varies
// slightly between runs.
enum { WARMUPS = 20, SAMPLES = 100 };

//
typedef void (*kernel_fn)(const float *, const float *, float *, size_t);

// input_a, input_b and output are POINTERS
static const float *input_a, *input_b;
static float *output;
static size_t length;

// this up here is global stuff --> now read the main

static double now_seconds(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    perror("clock_gettime");
    exit(EXIT_FAILURE);
  }
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int compare_double(const void *a, const void *b) {
  const double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static void report(const char *name, const double *values) {
  double sorted[SAMPLES], mean = 0.0, variance = 0.0;
  for (int i = 0; i < SAMPLES; ++i) {
    sorted[i] = values[i];
    mean += values[i] / SAMPLES;
  }
  qsort(sorted, SAMPLES, sizeof(sorted[0]), compare_double);
  for (int i = 0; i < SAMPLES; ++i) {
    const double delta = values[i] - mean;
    variance += delta * delta / SAMPLES;
  }
  printf("%-12s %10.3f %10.3f %10.3f %10.3f\n", name,
         (sorted[49] + sorted[50]) * 0.5, sqrt(variance), sorted[0],
         sorted[99]);
}

// run_kernel()
// now_seconds()
// report()
// compare double()

static void run_kernel(const char *name, kernel_fn fn) {
  double samples[SAMPLES];

  // warmup iterations
  for (int i = 0; i < WARMUPS; ++i)
    fn(input_a, input_b, output, length);

  // check correctness
  for (size_t i = 0; i < length; ++i) {
    if (!isfinite(output[i]) || output[i] != input_a[i] + input_b[i]) {
      fprintf(stderr, "%s correctness check failed at element %zu\n", name, i);
      exit(EXIT_FAILURE);
    }
  }

  // recall function 100 times and measure time --> compute stats over all runs
  for (int i = 0; i < SAMPLES; ++i) {
    const double start = now_seconds();
    fn(input_a, input_b, output, length);
    samples[i] = (now_seconds() - start) * 1e6;
  }
  report(name, samples);
}

// care only about this --> change length
int main(void) {
  // try:
  // 1024
  // 16384
  // 1048576
  length = 1048576;

  // allocate memory
  float *a = malloc(length * sizeof(*a));
  float *b = malloc(length * sizeof(*b));
  output = malloc(length * sizeof(*output));

  if (a == NULL || b == NULL || output == NULL) {
    // Print an error, release memory and stop
    fprintf(stderr, "Could not allocate vector-add benchmark arrays\n");
    free(a);
    free(b);
    free(output);
    return 2; // statement in a C function immediately stops the function and
              // sends the integer value 2 back to the caller
  }

  input_a = a;
  input_b = b;
  // fill the arrays with values --> NOTE: % gives the remainder after division
  for (size_t i = 0; i < length; ++i) {
    a[i] = (float)(i % 17);
    b[i] = (float)(i % 11);
  }

  // puts(...) = print without values
  // printf(...) = print + insert values into the text
  puts("Vector addition: y[i] = a[i] + b[i]");
  printf("Warmups: %d, timed samples: %d (microseconds)\n", WARMUPS, SAMPLES);
  puts("Method             median       std        min        max");

  // We recall the run_kernel helper function, which requires:
  // 1. A label to print.
  // 2. The function to execute.
  run_kernel("Scalar", vector_add_scalar);
  run_kernel("NEON", vector_add_neon);

  free(a);
  free(b);
  free(output);

  return 0;
}
