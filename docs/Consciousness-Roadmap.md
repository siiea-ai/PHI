# PHI Roadmap: From Philosophical Claims to Testable Systems

This roadmap operationalizes the PHI project's goals: build real, reusable modules that turn claims about consciousness and complexity into testable predictions, with reproducible experiments.

## Principles
- Predict-first: each module defines explicit, falsifiable predictions.
- Reproducibility: fixed seeds, CSV/JSON logs, tests under `tests/`.
- Minimal deps: numpy/scipy/pandas/sklearn; optional heavy deps are lazy.
- Comparative baselines: evaluate φ strategies vs non-φ baselines.

## Core Hypotheses (H)
- H1 (Spectral 1/f): conscious-like information processing exhibits 1/f^k (k∈[0.5,2]) spectral law and modulations detectable via PSD slope.
- H2 (Cross-scale coupling): control policies aligned to φ schedules stabilize cross-scale coupling more efficiently than fixed schedules.
- H3 (Algorithmic complexity): regimes with higher effective integration show increases in Lempel–Ziv complexity and multiscale entropy at intermediate scales.

## Metrics and Features
Provided by `phi/signals.py`:
- PSD: Welch PSD, bandpower in canonical bands; `psd_slope()`.
- PAC: Tort modulation index; `pac_tort_mi()`.
- Complexity: Higuchi FD, sample entropy, multiscale entropy, LZC; `higuchi_fd()`, `multiscale_entropy()`, `lzc()`.
- Online features for control/decoding: `feature_vector()` and `compute_metrics()`.

## Closed-loop BCI Simulation
Provided by `phi/neuro/bci.py` with CLI in `phi/cli.py neuro bci-sim`.
- Environment: latent state y_t with drift/noise; neural-like window signal with theta/gamma components; gamma envelope modulated by σ(y_t).
- Decoder: online linear regression on `feature_vector()`.
- Controller: homeostatic control u_t = -g(t)·ŷ_t with schedulers:
  - Constant
  - Cosine
  - Cosine with φ restarts (period grows by φ)
- Logs: timeseries CSV and summary JSON when `--out-dir` is given.

## Experiments
- E1 (Scheduler efficacy): Compare constant vs cosine vs φ-restarts on identical seeds.
  - Script: `experiments/phi_coupling.py`
  - Output: `out/bci_compare.csv` with MSE, MAE, TTC per run
  - Prediction (P1): φ-restarts reduce TTC and MAE relative to fixed cosine at equal average gain.
- E2 (Noise SNR sweep): Vary `snr_scale` in `BCIConfig`; measure effect on PAC and decoding error.
  - Expected: higher SNR increases PAC_MI; decoding error drops until saturation.
- E3 (Drift robustness): Increase `drift`; test scheduler responsiveness.
  - Expected: φ-restarts handle nonstationarity (faster error recovery) compared to constant.

## How to Run
- Single simulation (CLI):
  ```bash
  python -m phi.cli neuro bci-sim --steps 500 --scheduler cosine_phi --out-dir out/bci_phi
  ```
- Benchmark comparison:
  ```bash
  python experiments/phi_coupling.py --steps 1000 --seeds 42,43,44 --schedulers constant,cosine,cosine_phi --out-csv out/bci_compare.csv --save-logs
  ```

## Validation
- Unit tests:
  - `tests/test_signals.py`: sanity for metrics and feature extraction
  - `tests/test_bci.py`: simulation shapes, metrics, and CLI path

## Next Work
- Packaging & tests: `pip install -e .` then `pytest -q` to ensure imports (`from phi ...`) resolve.
- Parameter sweeps: grid over `snr_scale`, `drift`, `ctrl_effect`, `base_gain`, `base_lr`; plot MAE/MSE/TTC vs params.
- Schedulers: add step/linear, cyclical with warm restarts, and adaptive φ factor; compare vs φ-restarts.
- Metrics: add DFA/Hurst, wavelet coherence, IRASA 1/f separation; evaluate relation to decoding error.
- Visualization: PAC phase histograms, feature trajectories, scheduler timelines; small dashboard.
- Data integration: loader for external EEG/MEG segments to compute `phi.signals` metrics and PAC.
- Repro harness: seed sweep runner + CSV aggregation + plotting scripts in `experiments/`.
