#ifndef CASE_DATA_H
#define CASE_DATA_H

#include "sparse_gemm.h"

enum { HW1_M = 128, HW1_K = 512, HW1_N = 512, HW1_NNZB = 4096 };

/* Loads the fixed hard-coded case and expands its packed weights to dense W. */
int sparse_case_load_fixed(float *a, float *dense_w, sparse_weights *s);

#endif
