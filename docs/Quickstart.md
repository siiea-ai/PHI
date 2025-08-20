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

## Optional: LLM training (1.3B via LoRA)

For a reproducible workflow to fine-tune a ~1.3B parameter language model using Transformers + PEFT, see `docs/Training-1.3B-LLM.md`.

## Troubleshooting
- Headless plots: set `MPLBACKEND=Agg` (CI already does this).
- Video writing: `imageio-ffmpeg` is installed via requirements; system ffmpeg also works.
- Audio I/O: input must be 16-bit PCM WAV. Convert externally if needed.
- Large workloads: prefer `ratio` strategy for tiny models and fast expansion; use `--length` to resample.

