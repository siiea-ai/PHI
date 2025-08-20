# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-08-19
### Added
- CI: Added neuro CLI smoke tests to `.github/workflows/ci.yml`:
  - `python -m phi.cli neuro bci-sim --steps 30 ...`
  - Minimal `fractal neuro` pipeline (generate → compress → expand → simulate → analyze)
- Examples: New script `examples/neuro_pipeline/run.sh` for a runnable end-to-end neuro pipeline.
- Docs:
  - `docs/Quickstart.md`: BCI simulation and fractal neuro mini pipeline sections with copy-paste commands.
  - `docs/Consciousness-Roadmap.md`: Experimental Design Notes (BCI) linking φ schedulers to PSD/PAC/complexity hypotheses.

### Fixed
- Import bridge in `phi/neuro/__init__.py` by inserting dynamically loaded `phi/_neuro_impl` into `sys.modules` before `exec_module`, resolving `ImportError: 'NoneType' object has no attribute '__dict__'` in tests.

### Changed
- Bumped version in `pyproject.toml` to `0.1.1`.

### Testing
- Reinstalled in editable mode and ran full suite previously: 32 passed, 1 skipped. Added CI smoke for neuro CLI.

## [0.1.0] - 2025-08-01
- Initial public release of PHI: golden ratio experiments, signals, and neuro BCI simulations.
