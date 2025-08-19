# PHI Quickstart

Use this guide to set up the project-local virtual environment, install dependencies, run the CLI from any folder, and execute tests reliably.

## Prerequisites
- Python 3.12 recommended for “all-optional” installs (TensorFlow + Qiskit). Python 3.13 works but these two may be unavailable, so related tests will be skipped.
- macOS Apple Silicon uses `tensorflow-macos` wheels (already handled in requirements markers).

## 1) Create and use the project-local virtualenv

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

Use this same `.venv` for the entire project (CLI, tests, scripts) regardless of subfolder:

```bash
# Option A: explicit path each time (robust)
.venv/bin/python -m phi.cli --help

# Option B: activate once, then use python from anywhere
source .venv/bin/activate
python -m phi.cli --help
# when finished
deactivate
```

## 2) Sanity checks
```bash
# versions
.venv/bin/python --version
.venv/bin/python -c "import numpy, pandas, networkx, matplotlib; print('deps ok')"
```

## 3) Run the CLI (examples)
```bash
# See φ
.venv/bin/python -m phi.cli ratio

# Transform CSV
.venv/bin/python -m phi.cli transform \
  --input examples/example.csv \
  --output out_scaled.csv \
  --columns value \
  --op golden_scale --mode multiply
```

## 4) Run tests
```bash
# neuro-only first
.venv/bin/python -m pytest -vv -k neuro -rA -s
# full suite
.venv/bin/python -m pytest -q -rA
```

To get zero skipped tests (TensorFlow + Qiskit installed), prefer Python 3.12:
```bash
python3.12 -m venv .venv312
.venv312/bin/python -m pip install -U pip setuptools wheel
.venv312/bin/python -m pip install -r requirements.txt
.venv312/bin/python -m pytest -q -rA
```

## Troubleshooting
- Headless plots: set `MPLBACKEND=Agg` (CI already does this).
- Video writing: `imageio-ffmpeg` is installed via requirements; system ffmpeg also works.
- Audio I/O: input must be 16-bit PCM WAV. Convert externally if needed.
- Large workloads: prefer `ratio` strategy for tiny models and fast expansion; use `--length` to resample.

EOF

cat > docs/CLI-Cookbook.md << 'EOF'
# PHI CLI Cookbook

Task-oriented recipes. Prefix with the project venv for reliability:
```
.venv/bin/python -m phi.cli ...
```

## Timeseries end-to-end (phi strategy)
```bash
.venv/bin/python -m phi.cli fractal engine \
  --input examples/example.csv \
  --recon-output recon_phi_engine.csv \
  --columns value \
  --strategy phi --depth 4 --min-segment 3 \
  --smooth-window 5 --length 100 \
  --compare-plot engine_phi_compare.png \
  --analyze-output engine_phi_analysis.csv
```

## Timeseries end-to-end (ratio strategy)
```bash
.venv/bin/python -m phi.cli fractal engine \
  --input examples/example.csv \
  --recon-output recon_ratio_engine.csv \
  --columns value \
  --strategy ratio --ratio 3 --method interp \
  --length 100 \
  --compare-plot engine_ratio_compare.png \
  --analyze-output engine_ratio_analysis.csv
```

## Harmonizer (infra scheduling)
```bash
# Split schedule
.venv/bin/python -m phi.cli fractal harmonize \
  --model fractal_phi.json \
  --output harmonize_phi_split.csv \
  --schedule split --length 100 --total 1.0 --delta 0.1

# Backoff schedule
.venv/bin/python -m phi.cli fractal harmonize \
  --model fractal_phi.json \
  --output harmonize_phi_backoff.csv \
  --schedule backoff --length 100 --base 0.1 --max-delay 10 --beta 0.5
```

## Neuro pipeline (graph models)
```bash
# Generate (WS or BA) and preview
.venv/bin/python -m phi.cli fractal neuro generate \
  --output neuro_full.json \
  --nodes 500 --model ws --ws-k 10 --ws-p 0.1 --seed 42 \
  --state-init random --preview neuro_adj.png

# Compress and expand
.venv/bin/python -m phi.cli fractal neuro compress \
  --input neuro_full.json --model neuro_model.json \
  --ratio 4 --method interp

.venv/bin/python -m phi.cli fractal neuro expand \
  --model neuro_model.json --output neuro_recon.json \
  --nodes 500 --method interp --seed 42 --preview neuro_recon_adj.png

# Simulate and analyze
.venv/bin/python -m phi.cli fractal neuro simulate \
  --model neuro_recon.json --output neuro_states.csv \
  --steps 200 --dt 0.05 --leak 0.1 --input-drive 0.05 --noise-std 0.01 --seed 123

.venv/bin/python -m phi.cli fractal neuro analyze \
  --a neuro_full.json --b neuro_recon.json --output neuro_metrics.csv
```

## Image pipeline
```bash
.venv/bin/python -m phi.cli fractal image engine \
  --input mandelbrot.png --output mandelbrot_recon.png \
  --model image_model.json \
  --ratio 4 --method interp \
  --compare mandelbrot_compare.png \
  --analyze-output mandelbrot_analysis.csv
```

## Video pipeline
```bash
.venv/bin/python -m phi.cli fractal video engine \
  --input test.mp4 --output video_recon.mp4 \
  --model video_model.json \
  --spatial-ratio 2 --temporal-ratio 2 --method interp \
  --compare video_compare.png \
  --analyze-output video_analysis.csv --sample-frames 60
```

## 3D point clouds
```bash
.venv/bin/python -m phi.cli fractal three engine \
  --input tetra.ply --output tetra_recon.ply \
  --model three_model.json \
  --ratio 4 --method interp \
  --compare three_compare.png --axis z \
  --analyze-output three_metrics.csv --sample-points 1500 --nn-method auto
```

## AI bundles + Keras export
```bash
.venv/bin/python -m phi.cli fractal ai generate \
  --output ai_full.json --input-dim 16 --output-dim 4 \
  --depth 3 --base-width 64 --mode phi --seed 42

.venv/bin/python -m phi.cli fractal ai compress \
  --input ai_full.json --model ai_model.json \
  --ratio 2 --method interp

.venv/bin/python -m phi.cli fractal ai engine \
  --input ai_full.json --recon-output ai_recon.json \
  --model ai_model.json --ratio 2 --method interp \
  --analyze-output ai_metrics.csv --export-keras ai_model.h5
```

## Multiverse and Omniverse snippets
```bash
# Multiverse generate + preview mosaic
.venv/bin/python -m phi.cli fractal multiverse generate \
  --output multiverse_full.json --width 96 --height 96 --layers 9 --octaves 4 --seed 42 \
  --preview multiverse_mosaic.png

# Omniverse engine (similar flags)
.venv/bin/python -m phi.cli fractal omniverse engine \
  --input multiverse_full.json --recon-output multiverse_recon.json \
  --model multiverse_model.json --spatial-ratio 2 --layer-ratio 3 --method interp \
  --compare multiverse_compare.png --analyze multiverse_metrics.csv
```

## Cosmos/Quantum notes
- Cosmos: compress/expand/preview flows mirror multiverse/omniverse patterns.
- Quantum: see tests for programmatic APIs; for Qiskit-specific features ensure `qiskit-terra` (Python < 3.13).

Notes:
- `--analyze` is an alias for `--analyze-output` for all fractal engine subcommands.
- Prefer `.venv/bin/python -m ...` to ensure the correct interpreter.

EOF

cat > docs/Use-Cases.md << 'EOF'
# PHI Use Cases & Workflows

Practical scenarios with steps and expected outcomes. Includes ways to integrate with model training workflows.

## 1) Telemetry compression for long-term storage
- Problem: Reduce large time-series (metrics, logs) while retaining shape/trends.
- Steps:
  1. Compress: `fractal compress --strategy ratio --ratio N` (JSON model).
  2. Expand on demand: `fractal expand --method {interp|hold}` with optional `--length` to resample.
  3. Analyze fidelity: `fractal engine --compare-plot --analyze-output`.
- Outcome: Tiny model JSON, reconstructable CSV, metrics (MAE/RMSE/R2/corr), visualization PNGs.

## 2) One-shot timeseries reconstruction and evaluation
- Steps: `fractal engine --strategy {phi|ratio} --length ... --compare-plot ... --analyze-output ...`.
- Outcome: `recon_*.csv`, overlay plot, metrics CSV for QA/regression.

## 3) Harmonized infrastructure scheduling
- Steps: `fractal harmonize --schedule split|backoff` plus schedule-specific params.
- Outcome: CSV with per-timestep allocations or backoff delays for automation systems.

## 4) Neuro networks: structure, compression, simulation
- Steps:
  - Generate: `fractal neuro generate` with `--model ws|ba` and optional `--preview` adjacency.
  - Compress/Expand: `fractal neuro compress` → `fractal neuro expand`.
  - Simulate: `fractal neuro simulate` to produce state trajectories.
  - Compare: `fractal neuro analyze` on full vs reconstructed bundles.
- Outcome: Full/compressed/reconstructed JSONs, PNG previews, states CSV, metrics CSV.

## 5) Audio and video low-bitrate workflows
- Steps: `fractal audio engine` or `fractal video engine` with ratio/method; use `--compare` and `--analyze-output`.
- Outcome: Reconstructed media, side-by-side compare image, MSE/RMSE/PSNR metrics.

## 6) 3D point cloud compaction for robotics/LiDAR
- Steps: `fractal three compress` and `fractal three expand`; engine with `--analyze-output` and appropriate `--nn-method`.
- Outcome: Smaller `.ply` plus reconstruction; Chamfer-like metrics, preview images/plots.

## 7) Synthetic datasets and benchmarks
- Steps: `fractal multiverse generate` (3D stacks) and `fractal mandelbrot` (images/CSV grids).
- Outcome: Assets (JSON/PNG/CSV) to benchmark compression and analysis pipelines.

## 8) AI weight bundles for pruning + handoff
- Steps:
  1. Generate: `fractal ai generate` (architecture + random weights).
  2. Compress: `fractal ai compress` (ratio pruning across hidden layers).
  3. Expand: `fractal ai expand` to target widths; optionally use `fractal ai engine`.
  4. Export: `fractal ai engine --export-keras model.h5` for TensorFlow.
- Outcome: JSON bundles (full/compressed/recon) and `.h5`/`.keras` for training elsewhere.
- Notes:
  - Importing trained Keras models back into PHI JSON is not implemented (export is one-way).
  - Use metrics CSV to quantify pruning impact layer-by-layer.

## 9) Model QA, drift detection, and regression tests
- Steps: Run `fractal engine` regularly with `--analyze-output`. Store CSVs and plot trends.
- Outcome: Automated signals (MAE/RMSE/R2/corr) to catch regressions or data drift.

## 10) Training workflows leveraging PHI artifacts
- Patterns:
  - Data pre-processing: Use `fractal compress` to downsample large inputs for prototyping, then evaluate fidelity with `--analyze-output` before training full models.
  - Knowledge-preserving pruning: Use `fractal ai compress` to simulate structured pruning; export to Keras and fine-tune in TensorFlow.
  - Synthetic data: Use `multiverse`/`mandelbrot` generators to create controlled datasets for augmentation or curriculum training.
  - Neuro graphs: Use `neuro generate` to produce graph topologies for graph-based experiments (e.g., rate models or GNN prototypes). Load bundles with `phi.neuro.load_model` and convert to your GNN library format.
  - Multi-resolution curriculum: Train at multiple ratios and `--length`s to stabilize learning and reduce compute.
  - Augmentation by reconstruction: Expand with different methods (`interp` vs `hold`) and smoothing to create variants of signals/images.
- Outcome: Faster iterations, reproducible artifacts, and measurable trade-offs between compression and performance.

## Operational guidance
- Always invoke via the project venv (`.venv/bin/python -m phi.cli ...`), or activate once.
- Use Python 3.12 to enable TensorFlow and Qiskit; 3.13 is supported with those features skipped.
- Headless plots: `MPLBACKEND=Agg`.
- For larger jobs, prefer `ratio` strategy first, then try `phi` for higher fidelity with smoothing.