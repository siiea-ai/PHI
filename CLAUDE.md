# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Setup and Development
```bash
# Create virtual environment
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Run all tests
.venv/bin/python -m pytest -q -rA

# Run specific test module with verbose output
.venv/bin/python -m pytest -vv -k <module_name> -rA -s

# Run CLI
.venv/bin/python -m phi.cli --help
```

### Testing Strategy
- Use pytest for all tests
- Tests are organized by module under `tests/`
- Run focused tests during development: `.venv/bin/python -m pytest -vv -k <test_name> -rA -s`
- Always verify changes don't break existing tests

## Architecture Overview

### Core Design Principles
1. **Golden Ratio (φ) Integration**: All transformations and algorithms incorporate φ = 1.618... as a fundamental constant
2. **Modular Domain Support**: Each domain (image, audio, video, neuro, quantum) has its own module with consistent interfaces
3. **Lazy Imports**: Heavy dependencies (tensorflow, qiskit) are imported only when needed
4. **JSON Serialization**: Models use JSON with base64-encoded numpy arrays for portability

### Key Module Structure
```
phi/
├── constants.py      # PHI, INV_PHI, fibonacci sequences
├── transforms.py     # Core φ-based transformations
├── fractal.py       # Fractal compression strategies (φ-split, ratio)
├── engine.py        # High-level orchestration
├── signals.py       # Neuro-signal processing utilities
├── neuro/           # BCI and neuron network modules
│   └── bci.py      # Closed-loop BCI simulation
└── [domain].py      # Domain-specific implementations (image, audio, etc.)
```

### Compression Architecture
1. **Two Strategies**:
   - `phi-split`: Recursive partitioning using φ ratio
   - `ratio`: Decimation-based using φ for sampling

2. **Standard Flow**:
   - Compress: `data → fractal.compress() → compressed_model`
   - Expand: `compressed_model → fractal.expand() → reconstructed_data`
   - Models include metadata for reconstruction

### Neuro/BCI Components
- **Signal Analysis** (phi/signals.py): PSD, bandpower, PAC, complexity metrics
- **BCI Simulation** (phi/neuro/bci.py): Adaptive decoders, φ-scheduled controllers
- **Schedulers**: constant, cosine, cosine-with-φ-restarts

### Testing Approach
- Each module has comprehensive tests
- Tests compare φ-based methods against baselines
- Use fixed random seeds for reproducibility
- Log results to CSV/JSON for analysis