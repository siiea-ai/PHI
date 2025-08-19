from __future__ import annotations

from typing import Any, Dict, Optional

import json
import numpy as np
import pandas as pd

from .constants import PHI, INV_PHI
from . import transforms as _T


Node = Dict[str, Any]
Model = Dict[str, Any]


def _phi_left_len(total: int) -> int:
    # Ensure both sides get at least one element when possible
    if total <= 2:
        return 1
    left = int(round(total * INV_PHI))
    left = max(1, min(total - 1, left))
    return left


def _build_tree(arr: np.ndarray, depth: int, min_segment: int) -> Node:
    n = int(arr.shape[0])
    node: Node = {
        "len": n,
        "mean": float(np.mean(arr)) if n > 0 else 0.0,
        "std": float(np.std(arr)) if n > 0 else 0.0,
    }

    if depth <= 0 or n <= min_segment:
        node["left"] = None
        node["right"] = None
        return node

    l = _phi_left_len(n)
    left_arr = arr[:l]
    right_arr = arr[l:]
    node["left"] = _build_tree(left_arr, depth - 1, min_segment)
    node["right"] = _build_tree(right_arr, depth - 1, min_segment)
    return node


def _expand_tree(node: Node) -> np.ndarray:
    n = int(node["len"]) if node.get("len") is not None else 0
    left = node.get("left")
    right = node.get("right")
    if not left and not right:
        return np.full((n,), float(node.get("mean", 0.0)), dtype=float)
    # Recurse; lengths are stored at each child
    left_arr = _expand_tree(left) if left else np.array([], dtype=float)
    right_arr = _expand_tree(right) if right else np.array([], dtype=float)
    return np.concatenate([left_arr, right_arr])


def phi_fractal_compress(series: pd.Series | np.ndarray, depth: int = 4, min_segment: int = 8) -> Model:
    """Build a phi-split fractal summary tree over a 1D numeric series.

    This is a hierarchical, self-similar partition that stores means/stds at nodes.
    It's lossy by design and serves as a compact, harmonic structure summary.
    """
    if isinstance(series, pd.Series):
        arr = series.astype(float).to_numpy()
    else:
        arr = np.asarray(series, dtype=float)

    root = _build_tree(arr, depth=depth, min_segment=min_segment)
    return {
        "version": 1,
        "strategy": "phi",
        "phi": PHI,
        "root_len": int(arr.shape[0]),
        "depth": int(depth),
        "min_segment": int(min_segment),
        "tree": root,
    }


def phi_fractal_expand(model: Model, smooth_window: Optional[int] = 5) -> pd.Series:
    """Reconstruct an approximate series from a fractal model.

    Produces a piecewise-constant reconstruction followed by optional
    Fibonacci smoothing for continuity.
    """
    root = model["tree"]
    arr = _expand_tree(root)
    s = pd.Series(arr)
    if smooth_window and smooth_window > 1:
        s = _T.fibonacci_smooth(s, window=int(smooth_window))
    return s


def ratio_fractal_compress(series: pd.Series | np.ndarray, ratio: int = 2) -> Model:
    """Simple fractal-like decimation: keep every `ratio`-th sample.

    Stores positions and values; expansion can interpolate or hold.
    """
    if ratio < 1:
        ratio = 1
    if isinstance(series, pd.Series):
        arr = series.astype(float).to_numpy()
    else:
        arr = np.asarray(series, dtype=float)

    n = int(arr.shape[0])
    idx = np.arange(0, n, ratio, dtype=int)
    vals = arr[idx] if n > 0 else np.array([], dtype=float)
    return {
        "version": 1,
        "strategy": "ratio",
        "phi": PHI,
        "original_length": n,
        "ratio": int(ratio),
        "indices": idx.tolist(),
        "values": vals.astype(float).tolist(),
    }


def ratio_fractal_expand(model: Model, length: Optional[int] = None, method: str = "interp") -> pd.Series:
    n0 = int(model.get("original_length", 0))
    idx = np.asarray(model.get("indices", []), dtype=float)
    vals = np.asarray(model.get("values", []), dtype=float)
    if length is None:
        length = n0
    length = int(length) if length is not None else n0
    if length <= 0:
        return pd.Series([], dtype=float)

    if idx.size == 0:
        return pd.Series(np.zeros(length, dtype=float))

    # Ensure first and last points cover the range for interpolation/hold
    if idx[0] != 0:
        idx = np.insert(idx, 0, 0.0)
        vals = np.insert(vals, 0, vals[0])
    if idx[-1] != (n0 - 1):
        idx = np.append(idx, float(n0 - 1))
        vals = np.append(vals, vals[-1])

    x_new = np.linspace(0.0, float(n0 - 1), num=length, endpoint=True)
    if method == "hold":
        # Step function: assign each x_new the value of the nearest sample on the left
        out = np.empty_like(x_new)
        j = 0
        for i, x in enumerate(x_new):
            while j + 1 < len(idx) and x >= idx[j + 1]:
                j += 1
            out[i] = vals[j]
    else:
        # Linear interpolation
        out = np.interp(x_new, idx, vals)
    return pd.Series(out)


def save_model(model: Model, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f)


def load_model(path: str) -> Model:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compress_dataframe(
    df: pd.DataFrame,
    columns: list[str] | None,
    depth: int = 4,
    min_segment: int = 8,
    *,
    strategy: str = "phi",
    ratio: int = 2,
) -> Dict[str, Model]:
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    models: Dict[str, Model] = {}
    for col in columns:
        if strategy == "phi":
            models[col] = phi_fractal_compress(df[col], depth=depth, min_segment=min_segment)
        elif strategy == "ratio":
            models[col] = ratio_fractal_compress(df[col], ratio=ratio)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return models


def expand_series(model: Model, *, length: Optional[int] = None, smooth_window: int = 5, method: str = "interp") -> pd.Series:
    strategy = model.get("strategy", "phi")
    if strategy == "phi":
        s = phi_fractal_expand(model, smooth_window=smooth_window)
        if length is not None and length != len(s):
            x_old = np.linspace(0.0, 1.0, num=len(s), endpoint=True)
            x_new = np.linspace(0.0, 1.0, num=int(length), endpoint=True)
            return pd.Series(np.interp(x_new, x_old, s.to_numpy()))
        return s
    elif strategy == "ratio":
        return ratio_fractal_expand(model, length=length, method=method)
    else:
        raise ValueError(f"Unknown strategy in model: {strategy}")


def expand_to_dataframe(
    models: Dict[str, Model],
    length: Optional[int] = None,
    smooth_window: int = 5,
    *,
    method: str = "interp",
) -> pd.DataFrame:
    cols = {}
    max_len = 0
    for name, m in models.items():
        s = expand_series(m, length=length, smooth_window=smooth_window, method=method)
        cols[name] = s
        max_len = max(max_len, len(s))

    # Align lengths if needed
    out = pd.DataFrame()
    for name, s in cols.items():
        if len(s) == max_len:
            out[name] = s.reset_index(drop=True)
        else:
            x_old = np.linspace(0.0, 1.0, num=len(s), endpoint=True)
            x_new = np.linspace(0.0, 1.0, num=max_len, endpoint=True)
            out[name] = pd.Series(np.interp(x_new, x_old, s.to_numpy()))
    return out
