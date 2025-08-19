import math
from typing import List

# Golden ratio and related constants
PHI: float = (1.0 + math.sqrt(5.0)) / 2.0
INV_PHI: float = 1.0 / PHI  # ~0.618...


def fibonacci_sequence(n: int) -> List[int]:
    """Return the first n Fibonacci numbers starting at 1, 1, 2, ...
    For n <= 0, return [].
    """
    if n <= 0:
        return []
    if n == 1:
        return [1]
    seq = [1, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq
