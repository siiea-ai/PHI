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
  4. Export: `fractal ai engine --export-keras model.keras` (native Keras format; if no extension is provided, `.keras` is appended).
- Outcome: JSON bundles (full/compressed/recon) and `.keras` for training elsewhere (legacy `.h5` supported; requires `h5py`).
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

## 11) Emergent neuro dynamics sweep (complexity metrics)
- Steps:
  - Run the example script to explore regimes across leak and noise while computing synchrony, metastability, PCA dominance, participation ratio, and spectral entropy.
  - Quick smoke test (small grid):
    ```bash
    .venv/bin/python examples/neuro_metrics_sweep.py \
      --nodes 256 --model ws --ws-k 12 --ws-p 0.1 \
      --steps 600 --burn-in 200 \
      --leak-grid 0.08 0.20 3 --noise-grid 0.00 0.10 3 \
      --out-prefix out/neuro_sweep_quick --plots
    ```
  - Full sweep (denser grid):
    ```bash
    .venv/bin/python examples/neuro_metrics_sweep.py \
      --nodes 256 --model ws --ws-k 12 --ws-p 0.1 \
      --steps 2000 --burn-in 500 \
      --leak-grid 0.05 0.25 9 --noise-grid 0.00 0.15 9 \
      --out-prefix out/neuro_sweep --plots
    ```
- Outcome:
  - CSV at `out/neuro_sweep*.csv` containing one row per (leak, noise) with metrics.
  - Heatmaps at `out/neuro_sweep*_<metric>.png` if `--plots` is used.
- Notes:
  - WS vs BA topologies: switch `--model ba --ba-m 4` to compare; small-world WS with moderate `ws_p` often shows higher metastability/participation.
  - Interpretation (guidelines):
    - High synchrony + low PR → overly ordered, low diversity.
    - High spectral entropy + high PR + moderate PCA1 → richer distributed dynamics.
    - Very high PCA1 or saturation → collapse into low-dimensional or unstable regimes.

## Operational guidance
- Always invoke via the project venv (`.venv/bin/python -m phi.cli ...`), or activate once.
- Use Python 3.12 to enable TensorFlow and Qiskit; 3.13 is supported with those features skipped.
- Headless plots: `MPLBACKEND=Agg`.
- For larger jobs, prefer `ratio` strategy first, then try `phi` for higher fidelity with smoothing.