#!/usr/bin/env bash
set -euo pipefail

# Neuro pipeline demo: generate -> compress -> expand -> simulate -> analyze
# Writes outputs to out/neuro_pipeline

# Resolve repo root (script is in examples/neuro_pipeline)
ROOT_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"
PY_BIN="${PY_BIN:-python3}"
OUT_DIR="$ROOT_DIR/out/neuro_pipeline"

mkdir -p "$OUT_DIR"

echo "[1/5] Generate full neuro network -> $OUT_DIR/full.json"
"$PY_BIN" -m phi.cli fractal neuro generate \
  --output "$OUT_DIR/full.json" \
  --nodes 24 --model ws --ws-k 4 --ws-p 0.1 \
  --seed 5 --state-init random

echo "[2/5] Compress -> $OUT_DIR/model.json"
"$PY_BIN" -m phi.cli fractal neuro compress \
  --input "$OUT_DIR/full.json" \
  --model "$OUT_DIR/model.json" \
  --ratio 3 --method interp

echo "[3/5] Expand -> $OUT_DIR/recon.json"
"$PY_BIN" -m phi.cli fractal neuro expand \
  --model "$OUT_DIR/model.json" \
  --output "$OUT_DIR/recon.json" \
  --nodes 24 --method interp --seed 7

echo "[4/5] Simulate -> $OUT_DIR/traj.csv"
"$PY_BIN" -m phi.cli fractal neuro simulate \
  --model "$OUT_DIR/recon.json" \
  --output "$OUT_DIR/traj.csv" \
  --steps 10 --dt 0.1 --leak 0.1 --input-drive 0.05 --noise-std 0.0 --seed 42

echo "[5/5] Analyze A/B -> $OUT_DIR/metrics.csv"
"$PY_BIN" -m phi.cli fractal neuro analyze \
  --a "$OUT_DIR/full.json" \
  --b "$OUT_DIR/recon.json" \
  --output "$OUT_DIR/metrics.csv"

echo "Done. Outputs in: $OUT_DIR"
