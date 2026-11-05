#!/usr/bin/env bash
# HW1 full pipeline. Run ON THE RASPBERRY PI 4, from the hw1/ directory:
#
#   GROUP=7 ./run_pi.sh            # 2 benchmark runs (default)
#   GROUP=7 RUNS=5 ./run_pi.sh     # more runs -> more reliable std
#
# Steps: build -> correctness tests -> benchmark runs -> analysis/plots ->
# report PDF (if pdflatex is available) -> HW1_Group<GROUP>.zip
set -euo pipefail
cd "$(dirname "$0")"

GROUP="${GROUP:-N}"
RUNS="${RUNS:-2}"

make clean
make sparse_gemm
make src/sparse_gemm/test_kernel
./src/sparse_gemm/test_kernel

csvs=()
for r in $(seq 1 "$RUNS"); do
    out="results_run${r}.csv"
    ./src/sparse_gemm/benchmark --output "$out"
    csvs+=("$out")
done

# needs matplotlib for the plots: sudo apt install python3-matplotlib
python3 analysis/analyze.py "${csvs[@]}" -o analysis

report="HW1_Group${GROUP}.tex"
if [ "$GROUP" != "N" ] && [ ! -f "$report" ]; then
    sed "s/Group\\\\texttt{N}/Group${GROUP}/g" HW1_GroupN.tex > "$report"
fi
if command -v pdflatex >/dev/null; then
    pdflatex -interaction=nonstopmode -halt-on-error "$report" >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error "$report" >/dev/null
else
    echo "pdflatex not found: compile ${report} elsewhere (copy analysis/ too)"
fi

zipname="HW1_Group${GROUP}.zip"
rm -f "$zipname"
files=(src/sparse_gemm/sparse_neon.c src/sparse_gemm/sparse_tiled.c
       src/sparse_gemm/sparse_openmp.c src/sparse_gemm/sparse_openmp_tiled_neon.c
       "${csvs[@]}" analysis/analyze.py analysis/summary.csv analysis/summary.md
       analysis/results_table.tex)
for f in analysis/*.png "HW1_Group${GROUP}.pdf"; do [ -f "$f" ] && files+=("$f"); done
zip -j "$zipname" "${files[@]}"
echo "Created $zipname"
