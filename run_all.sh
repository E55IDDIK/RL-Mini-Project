#!/bin/bash

set -euo pipefail

# always run from the repository root, whatever directory this was invoked from
cd "$(dirname "$0")"

# use python3 if available, else python (Ubuntu often has no bare `python`)
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "ERROR: no python interpreter found on PATH." >&2
    exit 1
fi
echo "Using interpreter: $($PY --version 2>&1)"

mkdir -p results figures checkpoints logs

# --- fail loudly if the checkpoints are missing --------------------------
# Without this the pipeline runs to completion but writes EMPTY csv files and
# blank figures, exiting 0 — a silent failure that is worse than a crash.
n_ckpt=$(find checkpoints -name '*_step*.pt' 2>/dev/null | wc -l)
if [ "$n_ckpt" -eq 0 ]; then
    echo "ERROR: no checkpoints found in checkpoints/." >&2
    echo "       Expected files like checkpoints/gnn_dqn_seed42_step20000.pt" >&2
    echo "       Train first, or restore the checkpoints shipped with this repo." >&2
    exit 1
fi
echo "Found $n_ckpt checkpoint(s)."

echo ""
echo "========================================="
echo "1. Evaluating all training checkpoints..."
echo "========================================="
$PY -m src.eval_all_checkpoints

echo ""
echo "========================================="
echo "2. Evaluating final models & baselines..."
echo "========================================="
$PY -m src.eval

echo ""
echo "========================================="
echo "3. Running ablation study..."
echo "========================================="
$PY -m src.ablation_observation_radius

echo ""
echo "========================================="
echo "4. Generating figures..."
echo "========================================="
$PY -m src.plot

# --- verify the artefacts actually exist and are non-trivial -------------
echo ""
echo "========================================="
echo "Verifying outputs"
echo "========================================="
status=0
for f in results/learning_curve_data.csv results/eval_comparison.csv results/ablation_observation_radius.csv; do
    if [ ! -s "$f" ]; then
        echo "  MISSING/EMPTY: $f" >&2; status=1
    else
        # header + at least one data row
        rows=$(($(wc -l < "$f") - 1))
        if [ "$rows" -lt 1 ]; then
            echo "  NO DATA ROWS: $f" >&2; status=1
        else
            echo "  OK  $f  ($rows rows)"
        fi
    fi
done
for f in figures/learning_curve_return.pdf figures/learning_curve_metrics.pdf figures/benchmark_comparison.pdf; do
    if [ ! -s "$f" ]; then
        echo "  MISSING/EMPTY: $f" >&2; status=1
    else
        echo "  OK  $f"
    fi
done

if [ "$status" -ne 0 ]; then
    echo ""
    echo "FAILED: some outputs were not produced." >&2
    exit 1
fi

echo ""
echo "Done. All results and figures regenerated."
echo "  results/  -> csv tables"
echo "  figures/  -> vector pdf figures"
