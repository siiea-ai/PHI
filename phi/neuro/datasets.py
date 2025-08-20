"""
Dataset utilities for BCI simulation outputs.

This module provides helpers to load a supervised dataset from artifacts
created by `phi.cli neuro bci-sim` when using the flags to save features,
windows, and a config JSON.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import json
import numpy as np
import pandas as pd


META_COLS = {"t", "y_true", "y", "y_hat", "err", "sched"}


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def load_bci_dataset(out_dir: str, use_features: bool = True, use_windows: bool = False) -> Dict[str, Any]:
    """
    Load a supervised dataset from a BCI simulation output directory.

    Parameters
    - out_dir: path that contains bci_timeseries.csv and optionally
      bci_features.csv, bci_windows.npz, and bci_config.json
    - use_features: if True and features CSV exists, return X_feat
    - use_windows: if True and windows NPZ exists, return X_win

    Returns a dict with keys:
    - y: (T,) float32 target vector (prefers y_true if present)
    - X_feat: (T, D) float32 feature matrix or None
    - X_win: (T, N) float32 window matrix or None (N = fs * window_sec)
    - meta: dict with config info (if available) and basic shapes
    """
    out_path = Path(out_dir)
    if not out_path.exists():
        raise FileNotFoundError(f"Output directory not found: {out_dir}")

    # Required timeseries
    ts_path = out_path / "bci_timeseries.csv"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing timeseries CSV: {ts_path}")
    ts = pd.read_csv(ts_path)

    # Targets: prefer y_true if present, else fall back to y
    y_col = "y_true" if "y_true" in ts.columns else "y"
    y = ts[y_col].to_numpy(dtype=np.float32)

    # Optional features
    X_feat = None
    feat_cols = []
    feat_path = out_path / "bci_features.csv"
    if use_features and feat_path.exists():
        fdf = pd.read_csv(feat_path)
        # Determine feature columns: use config feature_names when present
        cfg = _read_json(out_path / "bci_config.json")
        if cfg and isinstance(cfg.get("feature_names"), list) and len(cfg["feature_names"]) > 0:
            feat_cols = [c for c in cfg["feature_names"] if c in fdf.columns]
        else:
            # Fallback: anything not in metadata columns
            feat_cols = [c for c in fdf.columns if c not in META_COLS]
        X_feat = fdf[feat_cols].to_numpy(dtype=np.float32)

    # Optional windows
    X_win = None
    win_meta: Dict[str, Any] = {}
    win_path = out_path / "bci_windows.npz"
    if use_windows and win_path.exists():
        npz = np.load(win_path)
        X = npz.get("X")
        if X is not None:
            X_win = np.asarray(X, dtype=np.float32)
        win_meta = {k: npz[k].item() if npz[k].shape == () else npz[k].tolist() for k in npz.files if k != "X"}

    cfg = _read_json(out_path / "bci_config.json") or {}

    meta: Dict[str, Any] = {
        "out_dir": str(out_path),
        "timeseries_csv": str(ts_path),
        "features_csv": str(feat_path) if feat_path.exists() else None,
        "windows_npz": str(win_path) if win_path.exists() else None,
        "config_json": str(out_path / "bci_config.json") if (out_path / "bci_config.json").exists() else None,
        "feature_columns": feat_cols,
        "y_col": y_col,
        "T": int(y.shape[0]),
    }
    meta.update({"win_meta": win_meta})
    meta.update({"config": cfg})

    return {
        "y": y,
        "X_feat": X_feat,
        "X_win": X_win,
        "meta": meta,
    }
