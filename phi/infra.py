from __future__ import annotations

import math
from typing import Tuple

from .constants import PHI, INV_PHI


def golden_backoff(attempt: int, base: float = 0.1, max_delay: float = 10.0) -> float:
    """Golden-ratio exponential backoff.

    delay = base * (PHI ** attempt), clipped to max_delay.
    """
    if attempt < 0:
        attempt = 0
    delay = float(base) * (PHI ** float(attempt))
    return min(delay, float(max_delay))


def golden_split(total: float, larger_first: bool = True) -> Tuple[float, float]:
    """Split a total into approximately 61.8% and 38.2% parts.

    Let a be the larger part. Then a = total / PHI, b = total - a = total / PHI**2
    """
    a = total * INV_PHI  # ~0.618 * total
    b = total - a        # ~0.382 * total
    if larger_first:
        return (a, b)
    return (b, a)
