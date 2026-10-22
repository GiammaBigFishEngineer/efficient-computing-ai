#include "kernels.h"

// void: the function returns no value; it writes results into y
// const float *a and const float *b: give access to the input float arrays;
// const prevents modifying their elements through these parameters --> the
// function can read them, but cannot change them float *y: gives access to the
// output float array, which you can modify. NOTE: * means a is a pointer = it
// holds a memory address where a float is stored.

void vector_add_scalar(const float *a, const float *b, float *y, size_t n) {
  /* TODO 1 */
  // traditional scalar vector addition --> NO OPTIMIZATION

  /*
      the scalar version—adds one pair of numbers at a time, using an ordinary C
     loop. Arguments: a and b are read-only FP32 arrays of n elements; y is the
      writable FP32 result array of n elements. Compute y[i] = a[i] + b[i].
  */

  // ignore size_it, it's a common data type ... retarded C stuff
  for (size_t i = 0; i < n; ++i) {
    y[i] = a[i] + b[i];
  }
}
