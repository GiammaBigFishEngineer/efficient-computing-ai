# HW1 sparse GEMM: benchmark summary

Inputs: results_run1.csv, results_run2.csv, results_run3.csv, results_run4.csv, results_run5.csv  
Case: M=128, K=512, N=512, nnzb=4096 (density 0.25)

Ideal OI (compulsory traffic, FP32 + uint32 indices):

- dense:  67,108,864 FLOP / 1,572,864 B = **42.67 FLOP/B**
- sparse: 16,777,216 FLOP / 803,332 B = **20.88 FLOP/B**  (4x fewer FLOP than dense)

Median = median of per-run medians; std = std of per-run medians.

| Kernel | OI (FLOP/B) | Median (ms) | Std (ms) | Min..max (ms) | GFLOP/s | vs dense scalar | vs sparse scalar |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense scalar | 42.7 | 563.1 | 14.5 | 532.5..580.4 | 0.12 | 1.00x | 0.17x |
| Dense tiled with NEON and OpenMP | 42.7 | 83.5 | 3.7 | 75.9..170.2 | 0.80 | 6.74x | 1.18x |
| Sparse scalar | 20.9 | 98.1 | 0.3 | 96.4..100.1 | 0.17 | 5.74x | 1.00x |
| Sparse NEON | 20.9 | 34.2 | 2.4 | 34.0..39.8 | 0.49 | 16.46x | 2.87x |
| Sparse tiled | 20.9 | 26.4 | 0.0 | 26.2..26.5 | 0.64 | 21.36x | 3.72x |
| Sparse OpenMP (1 thread) | 20.9 | 94.6 | 0.8 | 91.3..95.4 | 0.18 | 5.96x | 1.04x |
| Sparse OpenMP (2 threads) | 20.9 | 47.6 | 0.1 | 46.9..49.4 | 0.35 | 11.83x | 2.06x |
| Sparse OpenMP (4 threads) | 20.9 | 24.0 | 0.3 | 23.6..24.8 | 0.70 | 23.43x | 4.08x |
| Sparse Combined (1 thread) | 20.9 | 26.2 | 0.8 | 25.2..27.4 | 0.64 | 21.47x | 3.74x |
| Sparse Combined (2 threads) | 20.9 | 13.4 | 0.4 | 12.7..13.6 | 1.26 | 42.14x | 7.34x |
| Sparse Combined (4 threads) | 20.9 | 6.6 | 0.1 | 6.3..8.2 | 2.56 | 85.83x | 14.96x |

Faster dense baseline: **Dense tiled with NEON and OpenMP** (83.5 ms).
