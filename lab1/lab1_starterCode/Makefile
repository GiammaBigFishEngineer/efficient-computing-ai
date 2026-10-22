CC = gcc

# Optimization level used throughout the lab.
# Override from the command line, e.g.:
#   make vector_add OPT=-O3
OPT ?= -O0

CFLAGS = $(OPT) -Wall -Wextra -std=c11 -mcpu=cortex-a72 -Iinclude
OMPFLAGS = -fopenmp
LDFLAGS = -lm

# ---------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------

VECTOR_SRC = src/vector_add/vector_add.c \
             src/vector_add/vector_add_neon.c

REDUCE_SUM_SRC = src/reduce_sum/reduce_sum_scalar.c \
                 src/reduce_sum/reduce_sum_neon.c

GEMV_SRC = src/gemv/gemv.c \
           src/gemv/gemv_neon.c

GEMM_SRC = src/gemm/gemm.c \
           src/gemm/gemm_tiled.c

GEMM_BATCHED_SRC = src/gemm/gemm_batched.c

BATCHED_GEMM_SRC = $(GEMM_SRC) $(GEMM_BATCHED_SRC)

HEADERS = include/kernels.h

# ---------------------------------------------------------------------
# Executables
# ---------------------------------------------------------------------

EXERCISE_BINS = src/vector_add/benchmark \
                src/reduce_sum/benchmark \
                src/gemv/benchmark \
                src/gemm/benchmark

BATCHED_GEMM_BINS = src/gemm/batched_gemm_bench \
                    src/gemm/batched_gemm_bench_omp

TEST_BINS = src/vector_add/test_kernel \
            src/reduce_sum/test_kernel \
            src/gemv/test_kernel \
            src/gemm/test_kernel

# ---------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------

.PHONY: all \
        vector_add reduce_sum gemv gemm \
        batched_gemm_bench batched_gemm_bench_omp \
        test run clean

# ---------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------

all: vector_add reduce_sum gemv gemm

# ---------------------------------------------------------------------
# Exercise targets
# ---------------------------------------------------------------------

vector_add: src/vector_add/benchmark

reduce_sum: src/reduce_sum/benchmark

gemv: src/gemv/benchmark

gemm: src/gemm/benchmark

# ---------------------------------------------------------------------
# Benchmark targets
# ---------------------------------------------------------------------

src/vector_add/benchmark: src/vector_add/benchmark.c $(VECTOR_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(VECTOR_SRC) $< -o $@ $(LDFLAGS)

src/reduce_sum/benchmark: src/reduce_sum/benchmark.c $(REDUCE_SUM_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(REDUCE_SUM_SRC) $< -o $@ $(LDFLAGS)

src/gemv/benchmark: src/gemv/benchmark.c $(GEMV_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(GEMV_SRC) $< -o $@ $(LDFLAGS)

src/gemm/benchmark: src/gemm/benchmark.c $(GEMM_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(GEMM_SRC) $< -o $@ $(LDFLAGS)

# ---------------------------------------------------------------------
# Batched GEMM benchmarks
# ---------------------------------------------------------------------

src/gemm/batched_gemm_bench: src/gemm/batched_gemm_bench.c \
                             $(BATCHED_GEMM_SRC) \
                             $(HEADERS)
	$(CC) $(CFLAGS) $(BATCHED_GEMM_SRC) $< -o $@ $(LDFLAGS)

src/gemm/batched_gemm_bench_omp: src/gemm/batched_gemm_bench.c \
                                 $(BATCHED_GEMM_SRC) \
                                 $(HEADERS)
	$(CC) $(CFLAGS) $(OMPFLAGS) \
		$(BATCHED_GEMM_SRC) $< \
		-o $@ $(OMPFLAGS) $(LDFLAGS)

batched_gemm_bench: src/gemm/batched_gemm_bench

batched_gemm_bench_omp: src/gemm/batched_gemm_bench_omp

# ---------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------

src/vector_add/test_kernel: src/vector_add/test.c $(VECTOR_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(VECTOR_SRC) $< -o $@ $(LDFLAGS)

src/reduce_sum/test_kernel: src/reduce_sum/test.c $(REDUCE_SUM_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(REDUCE_SUM_SRC) $< -o $@ $(LDFLAGS)

src/gemv/test_kernel: src/gemv/test.c $(GEMV_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(GEMV_SRC) $< -o $@ $(LDFLAGS)

src/gemm/test_kernel: src/gemm/test.c $(GEMM_SRC) $(GEMM_BATCHED_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(GEMM_SRC) $(GEMM_BATCHED_SRC) $< -o $@ $(LDFLAGS)

test: $(TEST_BINS)
	@set -e; \
	for test_bin in $(TEST_BINS); do \
		echo "Running $$test_bin"; \
		./$$test_bin; \
	done

# ---------------------------------------------------------------------
# Convenience target
# ---------------------------------------------------------------------

run: gemm
	./src/gemm/benchmark

# ---------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------

clean:
	rm -f $(EXERCISE_BINS) \
	      $(BATCHED_GEMM_BINS) \
	      $(TEST_BINS)