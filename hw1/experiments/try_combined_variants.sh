#!/usr/bin/env bash
# Compare NEON formulations of the combined kernel ON THE PI (GCC, -O0).
# Run from hw1/:   ./experiments/try_combined_variants.sh
#   only some:     VARIANTS="3 5" ./experiments/try_combined_variants.sh
# Does not touch src/; binaries and CSVs go to experiments/out/.
set -euo pipefail
cd "$(dirname "$0")/.."

CC="${CC:-gcc}"
CFLAGS="${CFLAGS:--O0 -Wall -Wextra -std=c11 -mcpu=cortex-a72 -Iinclude -fopenmp}"
LDFLAGS="${LDFLAGS:--fopenmp -lm}"
OUT=experiments/out
mkdir -p "$OUT"

COMMON="src/sparse_gemm/sparse_common.c src/sparse_gemm/dense_gemm.c
        src/sparse_gemm/sparse_scalar.c src/sparse_gemm/sparse_neon.c
        src/sparse_gemm/sparse_tiled.c src/sparse_gemm/sparse_openmp.c
        src/sparse_gemm/case_data.c"

name() {
    case "$1" in
        1) echo "laneq (current)" ;;      2) echo "vdupq + vfmaq (hint)" ;;
        3) echo "vfmaq_n, r loop" ;;      4) echo "vfmaq_n, unrolled" ;;
        5) echo "vmlaq_n, unrolled" ;;
    esac
}

printf "\n%-3s %-22s %-6s %10s %10s %10s\n" "V" "formulation" "test" "T1 (ms)" "T2 (ms)" "T4 (ms)"
for v in ${VARIANTS:-1 2 3 4 5}; do
    $CC $CFLAGS -DVARIANT=$v $COMMON experiments/combined_variants.c \
        src/sparse_gemm/test.c -o "$OUT/test_v$v" $LDFLAGS
    $CC $CFLAGS -DVARIANT=$v $COMMON experiments/combined_variants.c \
        src/sparse_gemm/benchmark.c -o "$OUT/bench_v$v" $LDFLAGS
    st=PASS
    "$OUT/test_v$v" >"$OUT/test_v$v.log" 2>&1 || st=FAIL
    "$OUT/bench_v$v" --output "$OUT/v$v.csv" 2>/dev/null || true
    get() { awk -F, -v k="$1" '$2==k {printf "%.1f", $9/1000}' "$OUT/v$v.csv"; }
    printf "%-3s %-22s %-6s %10s %10s %10s\n" "$v" "$(name $v)" "$st" \
        "$(get sparse_openmp_tiled_neon_TM4_T1)" \
        "$(get sparse_openmp_tiled_neon_TM4_T2)" \
        "$(get sparse_openmp_tiled_neon_TM4_T4)"
done
echo
echo "Reference (same runs): sparse_tiled_TM4 = $(awk -F, '$2=="sparse_tiled_TM4" {printf "%.1f", $9/1000}' "$OUT/v${v}.csv") ms," \
     "sparse_neon = $(awk -F, '$2=="sparse_neon" {printf "%.1f", $9/1000}' "$OUT/v${v}.csv") ms"
