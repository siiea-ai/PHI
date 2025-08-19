from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .constants import INV_PHI
from . import infra as infra_mod


def _normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]. If constant, return 0.5s.
    """
    s = s.astype(float)
    vmin = float(np.min(s)) if len(s) > 0 else 0.0
    vmax = float(np.max(s)) if len(s) > 0 else 1.0
    if vmax <= vmin:
        return pd.Series(np.full_like(s, 0.5, dtype=float), index=s.index)
    return (s - vmin) / (vmax - vmin)


def harmonize_resource_split(
    s: pd.Series,
    total: float = 1.0,
    delta: float = 0.1,
) -> pd.DataFrame:
    """Create a phi-based resource split schedule from a series.

    - Base split is golden: a ~= 0.618, b ~= 0.382.
    - The series (normalized to [0,1]) tilts the split by +/- delta around base a.
    - Resulting alloc_a + alloc_b = total for each step.
    """
    s_norm = _normalize_series(s)
    base_a = INV_PHI  # ~0.618
    # allow swing up to +/- delta absolute fraction around base
    a_frac = np.clip(base_a + (s_norm - 0.5) * 2.0 * delta, 0.0, 1.0)
    b_frac = 1.0 - a_frac
    alloc_a = total * a_frac
    alloc_b = total * b_frac
    out = pd.DataFrame({
        "t": np.arange(len(s_norm), dtype=int),
        "value": s.astype(float).to_numpy(),
        "value_norm": s_norm.to_numpy(),
        "alloc_a": alloc_a.to_numpy(),
        "alloc_b": alloc_b.to_numpy(),
    })
    return out


essential_backoff_columns = ["t", "value", "value_norm", "delay"]

def harmonize_backoff(
    s: pd.Series,
    base: float = 0.1,
    max_delay: float = 10.0,
    beta: float = 0.5,
) -> pd.DataFrame:
    """Create a golden backoff schedule modulated by the series.

    delay_t = golden_backoff(t) * (1 + beta * (1 - s_norm))
    - Higher s_norm -> smaller multiplier (more aggressive)
    - Lower s_norm -> larger multiplier (more conservative)
    """
    s_norm = _normalize_series(s)
    t = np.arange(len(s_norm), dtype=int)
    base_delays = np.array([infra_mod.golden_backoff(int(i), base=base, max_delay=max_delay) for i in t])
    mult = 1.0 + beta * (1.0 - s_norm.to_numpy())
    delay = base_delays * mult
    out = pd.DataFrame({
        "t": t,
        "value": s.astype(float).to_numpy(),
        "value_norm": s_norm.to_numpy(),
        "delay": delay,
    })
    return out
