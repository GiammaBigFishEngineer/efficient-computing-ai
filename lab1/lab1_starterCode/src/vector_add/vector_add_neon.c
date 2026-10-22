#include "kernels.h"
#include <arm_neon.h>

void vector_add_neon(const float *a, const float *b, float *y, size_t n) {
  /* TODO 2 — NEON vector addition */

  /*
    Arguments:
    - a and b are read-only FP32 arrays of n elements --> WE HAVE THE POINTERS
    - y is the writable FP32 result array. A NEON vector holds four FP32 values.

    example:
    float a[8] = {10, 20, 30, 40, 50, 60, 70, 80};
        NOTE: [8] in a declaration specifies the array’s size, it is a count,
                not an index --> index starting at 0 only for this specific case
                --> C is a social experiment

        example: each number is 5 bytes
        these are an example of pointers:
                [1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035]
                *a = 1000 and it's fixed


    float b[8] = { 1,  2,  3,  4,  5,  6,  7,  8};
            pointers: [2000, 2005, 2010, 2015, 2020, 2025, 2030, 2035]
            *b = 2000 and it's fixed

    n = 7

    we want to load the first 4 numbers of a and b--> to do so the idea is to
    a.load(STARTING pointer) and b.load(STARTING pointer) and it loads the
    element corresponding to the passed pointer AND the following 3 numbers
    (total of 4 numbers) automatically.
    NOTE: vld1q_f32 already means “load FOUR consecutive floats” and you give it
    only the starting address

    a.load(*a) --> 10, 20, 30, 40
    b.load(*b) --> 1,  2,  3,  4

    we sum lane-wise using vaddq_f32:
    10, 20, 30, 40
    +   +   +   +
    1,  2,  3,  4
    -------------
    11, 22, 33, 44

    we save the 4 values in a register with: vst1q_f32(*y, addition_result)

    then we want to move to the values
    50, 60, 70, 80 --> pointers: 1020, 1025, 1030, 1035
    and
    5,6,7,8 --> pointers: 2020, 2025, 2030, 2035
    The most logical thing would be to use another pointer, so use not *a but
    the next value which is at *a + number of jumps * how much each jump "costs"
    --> 1000 + 4 * 5 = 1020
    Guess what? Not at all, let's just sum to the initial pointer *a the index
    of the next value we want to consider: *a + 4 --> it starts from the value
    at pointer *a which is 10 and goes forward of 4 elements accessing 50

    In code this is: vld1q_f32(a + 4) --> vld1q_f32(a + i)
    This still devastates me because we're summing an index to a pointer which
    doesn't make sense

    Otherwise you could write it as:
    vld1q_f32(&a[i]);  // Same as vld1q_f32(a + i)
  */

  size_t i = 0;

  while (i + 4 <= n) {
    float32x4_t a_4 = vld1q_f32(&a[i]);
    // access to the element in position i of a and take
    // the pointer --> use it as starting point: load that
    // number and the following 3

    float32x4_t b_4 = vld1q_f32(&b[i]);
    // access to the element in position i of b and take
    // the pointer --> use it as starting point: load that
    // number and the following 3

    float32x4_t result = vaddq_f32(a_4, b_4); // sum those numbers lane-wise

    vst1q_f32(&y[i], result); // store the 4 values in result starting from the
                              // pointer corresponding to element i of y

    i += 4;
  }

  // NOTE
  // when you get out of the previous while it means that the remaining values
  // are less then 4 process the remaining values one pair at a time until the
  // index is too big

  while (i < n) {       // Handle any remaining elements
    y[i] = a[i] + b[i]; // Add one pair
    ++i;                // Advance by one element to the next pair
  }
}