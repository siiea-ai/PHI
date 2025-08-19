

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

## Neuro API (programmatic)
Use the programmatic API in `phi/neuro.py` via a ready-made script:

```bash
.venv/bin/python examples/neuro_programmatic.py \
  --outdir examples/neuro_out \
  --nodes 500 --model ws --ws-k 10 --ws-p 0.1 --seed 42 \
  --ratio 4 --method interp \
  --steps 200 --dt 0.05 --leak 0.1 --input-drive 0.05 --noise-std 0.01
```

Outputs written under `examples/neuro_out/`:
- `neuro_full.json`, `neuro_model.json`, `neuro_recon.json`
- `neuro_adj.png`, `neuro_recon_adj.png`
- `neuro_states.csv` (state trajectories)
- `neuro_metrics.csv` (MSE for states, L1 for degree histogram)

This script uses `phi.neuro.generate_full_network()`, `compress_network()`, `expand_network()`, `simulate_states()`, and `metrics_from_paths()` to mirror the CLI pipeline in pure Python.

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
