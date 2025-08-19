# PHI: Golden Ratio Experiments

PHI is a playground to apply the golden ratio (φ) to data transformations and simple infrastructure heuristics. We start small: a Python library + CLI that computes φ, transforms CSV data, and offers φ-based utilities like golden backoff and golden split allocations.

```python
import math
phi = (1 + math.sqrt(5)) / 2
```

## Quickstart

1) Create a virtual environment and install dependencies (recommended):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Dependencies

Core (installed via requirements.txt):
- numpy, pandas, click, Pillow, imageio, imageio-ffmpeg, pytest

Optional (installed via requirements.txt to enable full functionality):
- SciPy: interpolation in `phi.cosmos`, `phi.multiverse`, `phi.omniverse` (ndimage.zoom), KD‑tree in `phi.three` (cKDTree). Fallbacks exist if missing.
- scikit-learn: optional NN backend in `phi.three` for metrics. Falls back if missing.
- Matplotlib: previews/plots in cosmos/multiverse/omniverse and 3D plotting in three. Some helpers require it and raise clear errors if absent.
- TensorFlow / tensorflow-macos: only for Keras export in `phi.ai.export_keras()` and `fractal ai engine --export-keras`. Not used otherwise.
- h5py: required when exporting Keras to `.h5`. Newer TF also supports `.keras` format.
- qiskit-terra: only to convert to a Qiskit circuit in `phi.quantum`. QASM export does not require it.
- pydub (+ ffmpeg): for reading non‑16‑bit WAV in `phi.audio`. 16‑bit PCM WAV works without it.

Platform notes:
- macOS Apple Silicon uses `tensorflow-macos`.
- Python 3.13: TensorFlow wheels may be unavailable; Keras export will be skipped unless a compatible TF is installed.

2) See φ:

```bash
.venv/bin/python -m phi.cli ratio
```

3) Transform a CSV using φ:

```bash
# Example CSV provided at examples/example.csv
.venv/bin/python -m phi.cli transform \
  --input examples/example.csv \
  --output out_scaled.csv \
  --columns value \
  --op golden_scale --mode multiply

.venv/bin/python -m phi.cli transform \
  --input examples/example.csv \
  --output out_smooth.csv \
  --columns value \
  --op fibonacci_smooth --window 5
```

- If `--columns` is omitted, numeric columns are auto-detected.
- Available ops: `golden_scale`, `golden_normalize`, `fibonacci_smooth`.

## Fractal compression/expansion

A φ-split fractal summarizes numeric columns by recursively partitioning data in golden-ratio proportions and storing simple stats (means/stds). Expansion reconstructs a piecewise-constant signal and optionally applies Fibonacci smoothing. This is lossy by design.

```bash
# Compress a column into a fractal model (JSON)
.venv/bin/python -m phi.cli fractal compress \
  --input examples/example.csv \
  --model fractal_model.json \
  --columns value \
  --depth 4 \
  --min-segment 3
  
# Expand (reconstruct) the fractal model back into a CSV
.venv/bin/python -m phi.cli fractal expand \
  --model fractal_model.json \
  --output fractal_recon.csv \
  --smooth-window 5 \
  --length 10   # optional; resamples to a target length
```

Tips:
- Lower `--min-segment` or higher `--depth` = more detail, larger model.
- Use `--columns` or omit to auto-pick numeric columns.
- Outputs are approximations; tune smoothing via `--smooth-window`.

### Strategies: phi vs ratio

You can now choose a compression strategy:

- `phi` (default): recursive φ-split tree storing means/stds. Good for harmonic hierarchical structure.
- `ratio`: keep every Nth sample (decimation). Tiny model; expansion via interpolation or step-hold.

Example (ratio strategy):

```bash
# Compress keeping every 3rd sample
.venv/bin/python -m phi.cli fractal compress \
  --input examples/example.csv \
  --model fractal_ratio.json \
  --columns value \
  --strategy ratio --ratio 3
  
# Expand with linear interpolation (default)
.venv/bin/python -m phi.cli fractal expand \
  --model fractal_ratio.json \
  --output recon_ratio_interp.csv \
  --length 50
  
# Or expand with step hold (staircase)
.venv/bin/python -m phi.cli fractal expand \
  --model fractal_ratio.json \
  --output recon_ratio_hold.csv \
  --length 50 \
  --method hold
```

## Fractal harmonizer

Derive harmonious infra schedules from a fractal-expanded series.

- Schedules:
  - `split`: golden resource split per timestep (alloc_a + alloc_b = total)
  - `backoff`: golden backoff delays modulated by the series

CLI (from a model file created via `fractal compress`):

```bash
# Split schedule from phi model
.venv/bin/python -m phi.cli fractal harmonize \
  --model fractal_phi.json \
  --output harmonize_phi_split.csv \
  --schedule split \
  --length 100 \
  --total 1.0 --delta 0.1

# Backoff schedule from phi model
.venv/bin/python -m phi.cli fractal harmonize \
  --model fractal_phi.json \
  --output harmonize_phi_backoff.csv \
  --schedule backoff \
  --length 100 \
  --base 0.1 --max-delay 10 --beta 0.5

# Split schedule from ratio model
.venv/bin/python -m phi.cli fractal harmonize \
  --model fractal_ratio.json \
  --output harmonize_ratio_split.csv \
  --schedule split \
  --length 100

# Backoff schedule from ratio model
.venv/bin/python -m phi.cli fractal harmonize \
  --model fractal_ratio.json \
  --output harmonize_ratio_backoff.csv \
  --schedule backoff \
  --length 100
```

Notes:
- Use `--column` to pick which column drives harmonization (defaults to first).
- For ratio models, expansion method can be chosen via `--method {interp,hold}`.
- `--smooth-window` tunes smoothing for phi strategy expansion.

## Fractal engine

Run compress + expand in one step with a single command, optionally saving the model and plotting the reconstruction. This wraps the lower-level `fractal compress` and `fractal expand` into one workflow.

```bash
# Phi strategy end-to-end
.venv/bin/python -m phi.cli fractal engine \
  --input examples/example.csv \
  --recon-output recon_phi.csv \
  --model fractal_phi.json \
  --columns value \
  --strategy phi --depth 4 --min-segment 3 \
  --smooth-window 5 \
  --length 100 \
  --plot engine_phi_plot.png

# Ratio strategy end-to-end (linear interpolation)
.venv/bin/python -m phi.cli fractal engine \
  --input examples/example.csv \
  --recon-output recon_ratio_interp.csv \
  --model fractal_ratio.json \
  --columns value \
  --strategy ratio --ratio 3 \
  --length 50 \
  --method interp \
  --plot engine_ratio_plot.png
```

Notes:
- `--model` is optional; when provided, the model bundle is saved as JSON.
- `--plot` saves a simple line plot (matplotlib if available, otherwise Pillow fallback).
- Omit `--columns` to auto-detect numeric columns.

### Engine analysis and compare plotting

Use `--compare-plot` to overlay original vs reconstructed, and `--analyze-output` to write a metrics CSV (mae, rmse, r2, corr, and basic stats).

```bash
# Phi strategy: compare + analyze
.venv/bin/python -m phi.cli fractal engine \
  --input examples/example.csv \
  --recon-output recon_phi_engine.csv \
  --columns value \
  --strategy phi \
  --length 20 \
  --compare-plot engine_phi_compare.png \
  --analyze-output engine_phi_analysis.csv

# Ratio strategy: compare + analyze
.venv/bin/python -m phi.cli fractal engine \
  --input examples/example.csv \
  --recon-output recon_ratio_engine.csv \
  --columns value \
  --strategy ratio --ratio 3 --method interp \
  --length 20 \
  --compare-plot engine_ratio_compare.png \
  --analyze-output engine_ratio_analysis.csv
```

## Image compression (ratio)

Compress images by downsampling with a ratio, then expand with interpolation or nearest-neighbor. Includes side-by-side comparison and metrics (MSE/RMSE/PSNR).

```bash
# Compress to a model (JSON)
.venv/bin/python -m phi.cli fractal image compress \
  --input mandelbrot.png \
  --model image_model.json \
  --ratio 4 --method interp

# Expand back to original size (or specify --width/--height)
.venv/bin/python -m phi.cli fractal image expand \
  --model image_model.json \
  --output mandelbrot_recon.png

# One-shot engine: compress + expand + compare + analyze
.venv/bin/python -m phi.cli fractal image engine \
  --input mandelbrot.png \
  --output mandelbrot_recon.png \
  --model image_model.json \
  --ratio 4 --method interp \
  --compare mandelbrot_compare.png \
  --analyze-output mandelbrot_analysis.csv
```

Notes:
- `--ratio` 2, 4, 8 ... increases compression (more loss).
- `--method` controls upsampling filter: `interp` (bilinear) or `nearest`.
- Comparison image places original and reconstructed side-by-side.

## Video compression (ratio)

Ratio-based video compression with spatial downsampling (every Nth pixel per axis) and temporal downsampling (every Nth frame). Expansion supports bilinear/linear interpolation or nearest/hold. Produces optional first-frame side-by-side compare image and metrics (MSE/RMSE/PSNR) over sampled frames.

```bash
# Compress to a model (JSON)
.venv/bin/python -m phi.cli fractal video compress \
  --input input.mp4 \
  --model video_model.json \
  --spatial-ratio 2 --temporal-ratio 2 --method interp

# Expand back to a playable video (MP4 or GIF). You can resize and/or change FPS.
.venv/bin/python -m phi.cli fractal video expand \
  --model video_model.json \
  --output recon.mp4 \
  --width 1280 --height 720 \
  --fps 24

# One-shot engine: compress + expand + compare + analyze
.venv/bin/python -m phi.cli fractal video engine \
  --input input.mp4 \
  --output recon.mp4 \
  --model video_model.json \
  --spatial-ratio 2 --temporal-ratio 2 --method interp \
  --compare video_compare.png \
  --analyze-output video_analysis.csv \
  --sample-frames 60
```

Notes:
- MP4 writing uses ffmpeg. If you don't have ffmpeg installed, install the Python bundle via `pip install imageio-ffmpeg` or install ffmpeg system-wide. GIF output generally works out of the box.
- `--method` controls both spatial and temporal upsampling: `interp` (bilinear spatial + linear temporal) or `nearest`/`hold` (step-wise).
- `--spatial-ratio` and `--temporal-ratio` increase compression at the cost of quality; larger values mean more loss.
- `--compare` saves a side-by-side image of the first frames; `--analyze-output` writes MSE/RMSE/PSNR across sampled frames.

## Multiverse compression (ratio)

Ratio-based compression of 3D stacks (layers of 2D fields). Spatial downsampling uses `--spatial-ratio`, and layer decimation uses `--layer-ratio`. Expansion supports bilinear (`interp`) or nearest-neighbor. Supports mosaic previews, side-by-side compare, and metrics.

```bash
# Generate a multiverse stack and preview mosaic
.venv/bin/python -m phi.cli fractal multiverse generate \
  --output multiverse_full.json \
  --width 96 --height 96 --layers 9 --octaves 4 --seed 42 \
  --preview multiverse_mosaic.png

# Compress to a model (JSON)
.venv/bin/python -m phi.cli fractal multiverse compress \
  --input multiverse_full.json \
  --model multiverse_model.json \
  --spatial-ratio 2 --layer-ratio 3 --method interp

# Expand back to original size (or specify --width/--height/--layers)
.venv/bin/python -m phi.cli fractal multiverse expand \
  --model multiverse_model.json \
  --output multiverse_recon.json \
  --method interp \
  --preview multiverse_recon_mosaic.png

# One-shot engine: compress + expand + compare + analyze
.venv/bin/python -m phi.cli fractal multiverse engine \
  --input multiverse_full.json \
  --recon-output multiverse_recon.json \
  --model multiverse_model.json \
  --spatial-ratio 2 --layer-ratio 3 --method interp \
  --compare multiverse_compare.png \
  --analyze multiverse_metrics.csv
```

Notes:
- `--method` controls upsampling: `interp` (bilinear) or `nearest`.
- Larger `--spatial-ratio`/`--layer-ratio` increase compression with more loss.
- `--compare` saves a side-by-side mosaic; `--analyze` or `--analyze-output` writes MSE/RMSE/PSNR and spectral metrics.

CLI note: For all fractal engine subcommands (`ai`, `cosmos`, `multiverse`, `omniverse`, `image`, `video`, `audio`, `three`, `quantum`), `--analyze` is an alias for `--analyze-output`.

## AI model compression (ratio)

Compress fully-connected neural nets by decimating hidden neurons (keep every Nth), then expand back via nearest or linear blend. Bundles are JSON with base64 numpy weights for portability. Optional Keras export.

```bash
# Generate a full model bundle
.venv/bin/python -m phi.cli fractal ai generate \
  --output ai_full.json \
  --input-dim 16 --output-dim 4 \
  --depth 3 --base-width 64 --mode phi \
  --seed 42

# Compress to a ratio model
.venv/bin/python -m phi.cli fractal ai compress \
  --input ai_full.json \
  --model ai_model.json \
  --ratio 2 --method interp

# Expand back to target hidden widths
.venv/bin/python -m phi.cli fractal ai expand \
  --model ai_model.json \
  --output ai_recon.json \
  --hidden 64,40,25 --method interp

# One-shot engine: compress + expand + analyze (and optional Keras export)
.venv/bin/python -m phi.cli fractal ai engine \
  --input ai_full.json \
  --recon-output ai_recon.json \
  --model ai_model.json \
  --ratio 2 --method interp \
  --analyze-output ai_metrics.csv \
  --export-keras ai_model.h5
```

Notes:
- Only the `ratio` strategy is implemented (educational demo).
- Metrics CSV reports per-layer MSE and totals; depends on pandas.
- Keras export requires TensorFlow installed (`pip install tensorflow`).

## 3D point clouds (ratio)

Ratio-based 3D point cloud compression by keeping every Nth point, with expansion via random interpolation or nearest resampling. I/O supports .ply (ASCII), .npz (compressed), and .npy. Includes optional 2D projection previews and Matplotlib 3D plots, plus an approximate symmetric Chamfer distance metric (selectable nearest-neighbor backend).

```bash
# Generate a Sierpinski tetrahedron (chaos game) and preview
.venv/bin/python -m phi.cli fractal three generate \
  --output tetra.ply \
  --points 20000 --seed 42 \
  --preview tetra_preview.png --axis z

# Compress to a model (JSON)
.venv/bin/python -m phi.cli fractal three compress \
  --input tetra.ply \
  --model three_model.json \
  --ratio 4 --method interp

# Expand back to a target count and preview
.venv/bin/python -m phi.cli fractal three expand \
  --model three_model.json \
  --output tetra_recon.ply \
  --points 20000 \
  --preview tetra_recon_preview.png --axis z

# One-shot engine: compress + expand + compare + analyze
.venv/bin/python -m phi.cli fractal three engine \
  --input tetra.ply \
  --output tetra_recon.ply \
  --model three_model.json \
  --ratio 4 --method interp \
  --compare three_compare.png --axis z \
  --analyze-output three_metrics.csv --sample-points 1500 --nn-method auto
```

Notes:
- I/O supports `.ply` (ASCII), `.npz` (compressed), and `.npy`.
- Optional 3D plotting via Matplotlib: use `--plot3d` to save a PNG or `--plot3d-show` to display.
- Metrics CSV computes an approximate symmetric Chamfer distance (RMSE and squared mean) using random subsampling; depends on pandas.
- Choose nearest-neighbor backend for metrics via `--nn-method {auto,kd,sklearn,brute}`. KD-tree (SciPy) and scikit-learn are optional; falls back to brute force.

## Audio compression (ratio)

Ratio-based audio compression (decimation) with expansion via linear interpolation or step hold. Requires 16-bit PCM WAV for I/O. Produces metrics (MSE/RMSE/PSNR) and optional waveform compare image.

```bash
# Compress WAV to a model (JSON)
.venv/bin/python -m phi.cli fractal audio compress \
  --input tone.wav \
  --model audio_model.json \
  --ratio 4 --method interp

# Expand back to original frames (or set --frames)
.venv/bin/python -m phi.cli fractal audio expand \
  --model audio_model.json \
  --output tone_recon.wav

# One-shot engine: compress + expand + compare + analyze
.venv/bin/python -m phi.cli fractal audio engine \
  --input tone.wav \
  --output tone_recon.wav \
  --model audio_model.json \
  --ratio 4 --method interp \
  --compare tone_compare.png \
  --analyze-output tone_analysis.csv
```

Notes:
- Input must be 16-bit PCM WAV. Use an external tool to convert if needed.
- `--method` controls expansion: `interp` (linear) or `hold` (step).
- Compare image overlays original vs reconstructed waveforms.

## Mandelbrot generator

Generate a Mandelbrot escape-count grid. Save as a PNG image, a CSV (x, y, iter), or both.

```bash
# Image (PNG)
.venv/bin/python -m phi.cli fractal mandelbrot \
  --output-image mandelbrot.png \
  --width 800 --height 800 --max-iter 512
  
# CSV of points and escape iterations
.venv/bin/python -m phi.cli fractal mandelbrot \
  --output-csv mandelbrot.csv \
  --width 300 --height 300 --max-iter 256
```

Options: `--xmin --xmax --ymin --ymax` control the viewport. Increase `--max-iter` for more detail (slower).

## Package Layout

- `phi/constants.py` — φ and related constants/utilities
- `phi/transforms.py` — data transforms (scale, normalize, Fibonacci smoothing)
- `phi/infra.py` — infra helpers (golden backoff, golden split)
- `phi/engine.py` — high-level engine to compress/expand/harmonize/plot
- `phi/cli.py` — CLI entrypoints (run with `python -m phi.cli ...`)
- `examples/` — sample data

## Infra Utilities (sneak peek)

- Golden backoff: delays that grow by φ each retry
- Golden split: partition resources into ~61.8% and ~38.2%

```python
from phi.infra import golden_backoff, golden_split

for attempt in range(5):
    delay = golden_backoff(attempt, base=0.1, max_delay=5)
    print(attempt, delay)

print(golden_split(100))  # -> (61.803..., 38.196...)
```

## Notes

- This is experimental and intentionally lightweight.
- Feel free to expand transforms and infra heuristics as we learn.
