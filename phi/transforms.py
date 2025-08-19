from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .constants import PHI, fibonacci_sequence


def golden_scale(series: pd.Series, factor: float = PHI, mode: str = "multiply") -> pd.Series:
    """Scale a numeric Series by the golden ratio (or a custom factor).

    mode="multiply"  -> series * factor
    mode="divide"    -> series / factor
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if mode not in {"multiply", "divide"}:
        raise ValueError("mode must be 'multiply' or 'divide'")
    if mode == "multiply":
        return series.astype(float) * float(factor)
    else:
        return series.astype(float) / float(factor)


def golden_normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize values into [0, PHI]."""
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = series.astype(float)
    min_v, max_v = float(s.min()), float(s.max())
    if max_v == min_v:
        return pd.Series([0.0] * len(s), index=s.index)
    norm01 = (s - min_v) / (max_v - min_v)
    return norm01 * PHI


def _normalized_fib_weights(length: int) -> np.ndarray:
    weights = np.array(fibonacci_sequence(length), dtype=float)
    s = weights.sum()
    return weights / s if s > 0 else weights


def fibonacci_smooth(series: pd.Series, window: int = 5) -> pd.Series:
    """Fibonacci-weighted moving average with centered window.

    - Uses dynamic weights near edges.
    - For non-numeric data, attempts to coerce to float.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    s = series.astype(float)

    def _apply_window(x: np.ndarray) -> float:
        w = _normalized_fib_weights(len(x))
        return float(np.dot(x, w)) if len(x) > 0 else np.nan

    # center=True yields symmetric windows; min_periods=1 handles edges
    return s.rolling(window=window, center=True, min_periods=1).apply(_apply_window, raw=True)


def _ensure_list(columns: Iterable[str] | None) -> List[str] | None:
    if columns is None:
        return None
    return [c for c in columns]


def apply_to_dataframe(
    df: pd.DataFrame,
    columns: Sequence[str] | None,
    op: str,
    **kwargs,
) -> pd.DataFrame:
    """Apply a transform to specified columns. Returns a copy.

    op in {"golden_scale", "golden_normalize", "fibonacci_smooth"}
    If columns is None, apply to numeric columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    cols = _ensure_list(columns)
    if cols is None:
        cols = df.select_dtypes(include=["number"]).columns.tolist()

    out = df.copy()

    for col in cols:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not in DataFrame")
        if op == "golden_scale":
            out[col] = golden_scale(out[col], **kwargs)
        elif op == "golden_normalize":
            out[col] = golden_normalize(out[col])
        elif op == "fibonacci_smooth":
            out[col] = fibonacci_smooth(out[col], **kwargs)
        else:
            raise ValueError("Unsupported op: " + op)

    return out
