CC = gcc
TARGET := $(shell $(CC) -dumpmachine 2>/dev/null)
ifeq ($(findstring aarch64,$(TARGET)),)
$(error HW1 requires an AArch64 ARM GCC toolchain (detected '$(TARGET)'))
endif
OPT ?= -O0
CFLAGS = $(OPT) -Wall -Wextra -std=c11 -mcpu=cortex-a72 -Iinclude
OMPFLAGS = -fopenmp
LDFLAGS = -lm

COMMON = src/sparse_gemm/sparse_common.c src/sparse_gemm/dense_gemm.c \
         src/sparse_gemm/sparse_scalar.c src/sparse_gemm/sparse_neon.c \
         src/sparse_gemm/sparse_tiled.c src/sparse_gemm/sparse_openmp.c \
         src/sparse_gemm/sparse_openmp_tiled_neon.c src/sparse_gemm/case_data.c

.PHONY: all sparse_gemm test clean
all: sparse_gemm
sparse_gemm: src/sparse_gemm/benchmark

src/sparse_gemm/benchmark: $(COMMON) src/sparse_gemm/benchmark.c include/sparse_gemm.h
	$(CC) $(CFLAGS) $(OMPFLAGS) $(COMMON) src/sparse_gemm/benchmark.c -o $@ $(OMPFLAGS) $(LDFLAGS)

src/sparse_gemm/test_kernel: $(COMMON) src/sparse_gemm/test.c include/sparse_gemm.h
	$(CC) $(CFLAGS) $(OMPFLAGS) $(COMMON) src/sparse_gemm/test.c -o $@ $(OMPFLAGS) $(LDFLAGS)

test: src/sparse_gemm/test_kernel
	./src/sparse_gemm/test_kernel

clean:
	rm -f src/sparse_gemm/benchmark src/sparse_gemm/test_kernel results_raw.csv
