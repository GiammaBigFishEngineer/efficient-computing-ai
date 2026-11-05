# HW1 – 4×4 block-sparse GEMM

> Full guide in Italian (Pi instructions, explanation of the kernels and NEON intrinsics, delivery checklist): [GUIDA.md](GUIDA.md)

Assignment: https://valentinopeluso.github.io/ECAI27/homework/homework-1.html

## Kernels

| File | What it does |
|---|---|
| `src/sparse_gemm/sparse_neon.c` | One `float32x4_t` accumulator per (row, 4-column group). For each block row r: `vdupq_n_f32(A[i,4q+r])` × `vld1q_f32(block row r)` via `vfmaq_f32`. One `vst1q_f32` per group. |
| `src/sparse_gemm/sparse_tiled.c` | Tile of exactly 4 input rows, scalar code (no NEON). Each 4-value weight row is loaded once into locals and reused for all 4 rows, so the weight and index stream is read M/4 times instead of M times. |
| `src/sparse_gemm/sparse_openmp.c` | Scalar sparse loop with a plain `#pragma omp parallel for` on the output-row loop. Each iteration writes its own row of C, so there are no races. |
| `src/sparse_gemm/sparse_openmp_tiled_neon.c` | Plain `#pragma omp parallel for` over independent 4-row tiles. Per group it keeps 4 NEON accumulators (the 4×4 output tile). Per block it loads the 4 A values of each row once (`vld1q_f32`) and each weight row once, then uses `vfmaq_laneq_f32` (lane-broadcast FMA) for all 4 rows. |

None of the kernels changes the interfaces, the packed BCC weights, or the block order. A trailing partial tile (m % 4 ≠ 0) falls back to a single-row path. This never happens for M = 128.

## Ideal operational intensity

Assumes compulsory traffic only, FP32 values and uint32 indices, each moved once.

- Dense: 2MKN = 67,108,864 FLOP / 4(MK+KN+MN) = 1,572,864 B → **42.7 FLOP/B**
- Sparse: 2·16·M·nnzb = 16,777,216 FLOP / (A + 16·nnzb values + nnzb row idx + N/4+1 col ptr + C) = 803,332 B → **20.9 FLOP/B** (the same for every sparse variant)

## Running on the Pi 4

```bash
sudo apt install python3-matplotlib zip   # once
GROUP=<n> ./run_pi.sh                      # build, test, 2 runs, analysis, PDF, zip
```

`analysis/analyze.py` computes median latency as the median of the per-run medians, and std as the standard deviation of the per-run medians. The CSV does not store individual samples, so more runs (`RUNS=5`) give a more meaningful std. `--report HW1_Group<n>.tex` (done by `run_pi.sh`) writes the numbers directly into the report table between the `% BEGIN RESULTS` / `% END RESULTS` markers. The plots are included from `analysis/*.png` when present.
