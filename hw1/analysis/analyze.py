#!/usr/bin/env python3
"""HW1 sparse GEMM: benchmark analysis.

Reads one or more raw CSV files written by ./src/sparse_gemm/benchmark and
produces, in the output directory (default: analysis/):

  summary.csv          per-kernel statistics (machine readable)
  summary.md           the same as a Markdown table + speedups
  results_table.tex    the rows of the report table (also written straight
                       into the report with --report HW1_GroupN.tex)
  latency.png          median latency per kernel (min..max across samples)
  speedup.png          speedup of every kernel over sparse_scalar
  thread_scaling.png   OpenMP thread scaling (sparse_openmp vs combined)

Statistics
----------
The benchmark harness stores, per run and kernel, the median / min / max of
5 timed samples (each sample is itself the mean of `repetitions` calls).
Individual samples are not written to the CSV, so:

  median latency = median of the per-run medians over all CSV files
  std latency    = sample standard deviation (ddof=1) of the per-run medians
                   (needs >= 2 runs; with one run it is reported as n/a)

Usage
-----
  python3 analysis/analyze.py results_run1.csv results_run2.csv \\
      [-o analysis] [--report HW1_GroupN.tex]

Only the Python standard library + matplotlib are needed.
"""

import argparse
import csv
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------- problem ---
M, K, N, NNZB = 128, 512, 512, 4096
FP32 = 4   # bytes per float
IDX = 4    # bytes per uint32 index

# Ideal operational intensity: single-level memory, infinite registers, so
# every operand is moved exactly once (compulsory traffic only).
DENSE_FLOP = 2 * M * K * N
DENSE_BYTES = FP32 * (M * K + K * N + M * N)          # A + dense W + C
SPARSE_FLOP = 2 * M * NNZB * 16                        # 16 MACs per block/row
SPARSE_BYTES = (FP32 * M * K                           # A
                + FP32 * 16 * NNZB                     # block_values
                + IDX * NNZB                           # block_row_idx
                + IDX * (N // 4 + 1)                   # block_col_ptr
                + FP32 * M * N)                        # C
OI_DENSE = DENSE_FLOP / DENSE_BYTES
OI_SPARSE = SPARSE_FLOP / SPARSE_BYTES

# kernel name in CSV -> (label in the report, family, flop, oi)
KERNELS = OrderedDict([
    ("dense_scalar", ("Dense scalar", "dense", DENSE_FLOP, OI_DENSE)),
    ("dense_tiled_neon_openmp_TN8_T4",
     ("Dense tiled with NEON and OpenMP", "dense", DENSE_FLOP, OI_DENSE)),
    ("sparse_scalar", ("Sparse scalar", "sparse", SPARSE_FLOP, OI_SPARSE)),
    ("sparse_neon", ("Sparse NEON", "sparse", SPARSE_FLOP, OI_SPARSE)),
    ("sparse_tiled_TM4", ("Sparse tiled", "sparse", SPARSE_FLOP, OI_SPARSE)),
    ("sparse_openmp_T1",
     ("Sparse OpenMP (1 thread)", "sparse", SPARSE_FLOP, OI_SPARSE)),
    ("sparse_openmp_T2",
     ("Sparse OpenMP (2 threads)", "sparse", SPARSE_FLOP, OI_SPARSE)),
    ("sparse_openmp_T4",
     ("Sparse OpenMP (4 threads)", "sparse", SPARSE_FLOP, OI_SPARSE)),
    ("sparse_openmp_tiled_neon_TM4_T1",
     ("Sparse Combined (1 thread)", "sparse", SPARSE_FLOP, OI_SPARSE)),
    ("sparse_openmp_tiled_neon_TM4_T2",
     ("Sparse Combined (2 threads)", "sparse", SPARSE_FLOP, OI_SPARSE)),
    ("sparse_openmp_tiled_neon_TM4_T4",
     ("Sparse Combined (4 threads)", "sparse", SPARSE_FLOP, OI_SPARSE)),
])

# Rows of the report template, in its order.
REPORT_ROWS = [
    "dense_scalar", "dense_tiled_neon_openmp_TN8_T4", "sparse_scalar",
    "sparse_neon", "sparse_tiled_TM4", "sparse_openmp_T2", "sparse_openmp_T4",
    "sparse_openmp_tiled_neon_TM4_T2", "sparse_openmp_tiled_neon_TM4_T4",
]

# Chart styling (reference palette, light mode).
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FAMILY_COLOR = {"sparse": "#2a78d6", "dense": "#eb6834"}


def load_runs(paths):
    """Return {kernel: [row, ...]} with one row per run (status == pass)."""
    runs = {name: [] for name in KERNELS}
    for path in paths:
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                name = row["kernel"]
                if name not in runs:
                    print(f"warning: unknown kernel {name!r} in {path}",
                          file=sys.stderr)
                    continue
                if row["status"] != "pass":
                    print(f"warning: {name} is {row['status']} in {path}; "
                          "excluded", file=sys.stderr)
                    continue
                runs[name].append({
                    "file": str(path),
                    "median": float(row["latency_median_us"]),
                    "min": float(row["latency_min_us"]),
                    "max": float(row["latency_max_us"]),
                })
    return runs


def summarise(runs):
    stats = OrderedDict()
    for name, (label, family, flop, oi) in KERNELS.items():
        rows = runs[name]
        if not rows:
            continue
        medians = [r["median"] for r in rows]
        med_us = statistics.median(medians)
        std_us = statistics.stdev(medians) if len(medians) > 1 else None
        stats[name] = {
            "label": label, "family": family, "oi": oi, "runs": len(rows),
            "median_ms": med_us / 1e3,
            "std_ms": None if std_us is None else std_us / 1e3,
            "min_ms": min(r["min"] for r in rows) / 1e3,
            "max_ms": max(r["max"] for r in rows) / 1e3,
            "gflops": flop / (med_us * 1e-6) / 1e9,
        }
    ref = {k: stats[k]["median_ms"] for k in
           ("dense_scalar", "sparse_scalar") if k in stats}
    for s in stats.values():
        s["speedup_vs_dense_scalar"] = (ref["dense_scalar"] / s["median_ms"]
                                        if "dense_scalar" in ref else None)
        s["speedup_vs_sparse_scalar"] = (ref["sparse_scalar"] / s["median_ms"]
                                         if "sparse_scalar" in ref else None)
    return stats


def fmt(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_csv(stats, path):
    cols = ["kernel", "label", "runs", "oi_flop_per_byte", "median_ms",
            "std_ms", "min_ms", "max_ms", "gflops",
            "speedup_vs_dense_scalar", "speedup_vs_sparse_scalar"]
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for name, s in stats.items():
            wr.writerow([name, s["label"], s["runs"], f"{s['oi']:.3f}",
                         f"{s['median_ms']:.4f}",
                         "" if s["std_ms"] is None else f"{s['std_ms']:.4f}",
                         f"{s['min_ms']:.4f}", f"{s['max_ms']:.4f}",
                         f"{s['gflops']:.3f}",
                         fmt(s["speedup_vs_dense_scalar"], 3),
                         fmt(s["speedup_vs_sparse_scalar"], 3)])


def write_markdown(stats, inputs, path):
    lines = [
        "# HW1 sparse GEMM: benchmark summary", "",
        f"Inputs: {', '.join(Path(p).name for p in inputs)}  ",
        f"Case: M={M}, K={K}, N={N}, nnzb={NNZB} (density 0.25)", "",
        "Ideal OI (compulsory traffic, FP32 + uint32 indices):", "",
        f"- dense:  {DENSE_FLOP:,} FLOP / {DENSE_BYTES:,} B = "
        f"**{OI_DENSE:.2f} FLOP/B**",
        f"- sparse: {SPARSE_FLOP:,} FLOP / {SPARSE_BYTES:,} B = "
        f"**{OI_SPARSE:.2f} FLOP/B**  (4x fewer FLOP than dense)", "",
        "Median = median of per-run medians; std = std of per-run medians.", "",
        "| Kernel | OI (FLOP/B) | Median (ms) | Std (ms) | Min..max (ms) "
        "| GFLOP/s | vs dense scalar | vs sparse scalar |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in stats.values():
        lines.append(
            f"| {s['label']} | {s['oi']:.1f} | {s['median_ms']:.1f} | "
            f"{fmt(s['std_ms'])} | {s['min_ms']:.1f}..{s['max_ms']:.1f} | "
            f"{s['gflops']:.2f} | {fmt(s['speedup_vs_dense_scalar'], 2)}x | "
            f"{fmt(s['speedup_vs_sparse_scalar'], 2)}x |")
    dense = [s for s in stats.values() if s["family"] == "dense"]
    if dense:
        fastest = min(dense, key=lambda s: s["median_ms"])
        lines += ["", f"Faster dense baseline: **{fastest['label']}** "
                      f"({fastest['median_ms']:.1f} ms)."]
    Path(path).write_text("\n".join(lines) + "\n")


def tex_rows(stats):
    out = []
    for name in REPORT_ROWS:
        if name not in stats:
            out.append(f"{KERNELS[name][0]} & {KERNELS[name][3]:.1f} "
                       "& TODO & TODO \\\\")
            continue
        s = stats[name]
        out.append(f"{s['label']} & {s['oi']:.1f} & {s['median_ms']:.1f} "
                   f"& {fmt(s['std_ms'])} \\\\")
    return out


def write_tex(stats, path):
    out = ["% Generated by analysis/analyze.py -- do not edit by hand."]
    Path(path).write_text("\n".join(out + tex_rows(stats)) + "\n")


BEGIN_MARK, END_MARK = "% BEGIN RESULTS", "% END RESULTS"


def fill_report(stats, report):
    """Write the table rows directly into the report, between the markers."""
    text = Path(report).read_text()
    lines = text.split("\n")
    try:
        b = next(i for i, l in enumerate(lines) if l.startswith(BEGIN_MARK))
        e = next(i for i, l in enumerate(lines) if l.startswith(END_MARK))
    except StopIteration:
        sys.exit(f"{report}: markers '{BEGIN_MARK}' / '{END_MARK}' not found")
    lines[b + 1:e] = tex_rows(stats)
    Path(report).write_text("\n".join(lines))
    print(f"Filled results table in {report}", file=sys.stderr)


# ------------------------------------------------------------------ plots ---
def _plain_log(ax, axis):
    """Plain numbers (1, 2, 5, 10, ...) instead of 10^x on a log axis."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fmt_ = FuncFormatter(lambda v, _: f"{v:g}")
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    target.set_major_formatter(fmt_)
    target.set_minor_formatter(NullFormatter())


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def plot_latency(plt, stats, path):
    names = list(stats)[::-1]
    y = range(len(names))
    med = [stats[n]["median_ms"] for n in names]
    err = [[stats[n]["median_ms"] - stats[n]["min_ms"] for n in names],
           [stats[n]["max_ms"] - stats[n]["median_ms"] for n in names]]
    colors = [FAMILY_COLOR[stats[n]["family"]] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.barh(y, med, height=0.6, color=colors, edgecolor=SURFACE, linewidth=2,
            xerr=err, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2},
            zorder=2)
    for yi, v in zip(y, med):
        ax.text(v * 1.08, yi, f"{v:.1f}", va="center", fontsize=7, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels([stats[n]["label"] for n in names])
    ax.set_xscale("log")
    _plain_log(ax, "x")
    ax.set_xlabel("Median latency (ms, log scale; whiskers = min..max)")
    ax.set_title("Kernel latency, M=128, K=N=512, nnzb=4096", fontsize=10,
                 loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f])
               for f in ("dense", "sparse")]
    ax.legend(handles, ["Dense", "Sparse (25% blocks)"], frameon=False,
              fontsize=8, loc="lower right", labelcolor=INK2)
    ax.set_xlim(right=max(stats[n]["max_ms"] for n in names) * 2.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_speedup(plt, stats, path):
    if "sparse_scalar" not in stats:
        return
    names = list(stats)[::-1]
    y = range(len(names))
    sp = [stats[n]["speedup_vs_sparse_scalar"] for n in names]
    colors = [FAMILY_COLOR[stats[n]["family"]] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.barh(y, sp, height=0.6, color=colors, edgecolor=SURFACE, linewidth=2,
            zorder=2)
    ax.axvline(1.0, color=INK2, linewidth=1, linestyle="--", zorder=3)
    for yi, v in zip(y, sp):
        ax.text(v + max(sp) * 0.01, yi, f"{v:.2f}x", va="center", fontsize=7,
                color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels([stats[n]["label"] for n in names])
    ax.set_xlabel("Speedup over sparse scalar (higher is better)")
    ax.set_title("Speedup relative to the sparse scalar baseline",
                 fontsize=10, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_xlim(0, max(sp) * 1.15)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_scaling(plt, stats, path):
    series = [
        ("Sparse OpenMP", "sparse_openmp_T{}", "#2a78d6", "o"),
        ("Sparse Combined (tile 4 + NEON)", "sparse_openmp_tiled_neon_TM4_T{}",
         "#1baf7a", "s"),
    ]
    threads = [1, 2, 4]
    fig, ax = plt.subplots(figsize=(6.0, 3.8), facecolor=SURFACE)
    _style(ax)
    plotted = False
    for label, pattern, color, marker in series:
        pts = [(t, stats[pattern.format(t)]["median_ms"]) for t in threads
               if pattern.format(t) in stats]
        if not pts:
            continue
        plotted = True
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker,
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=label, zorder=3)
        ax.annotate(f"{ys[-1]:.1f} ms", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    fontsize=7, color=INK)
        # ideal linear scaling from the 1-thread point
        if xs[0] == 1:
            ax.plot(threads, [ys[0] / t for t in threads], color=color,
                    linewidth=1, linestyle=":", zorder=2)
    if not plotted:
        plt.close(fig)
        return
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _plain_log(ax, "y")
    ax.minorticks_off()
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Median latency (ms, log)")
    ax.set_title("Thread scaling (dotted = ideal linear)", fontsize=10,
                 loc="left")
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+", help="raw benchmark CSV files")
    ap.add_argument("-o", "--outdir", default="analysis")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--report", metavar="TEX",
                    help="LaTeX report whose results table is filled in place")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stats = summarise(load_runs(args.csv))
    if not stats:
        sys.exit("no passing kernels found in the given CSV files")

    write_csv(stats, out / "summary.csv")
    write_markdown(stats, args.csv, out / "summary.md")
    write_tex(stats, out / "results_table.tex")
    if args.report:
        fill_report(stats, args.report)

    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed: skipping plots "
                  "(pip install matplotlib)", file=sys.stderr)
        else:
            plot_latency(plt, stats, out / "latency.png")
            plot_speedup(plt, stats, out / "speedup.png")
            plot_scaling(plt, stats, out / "thread_scaling.png")

    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
