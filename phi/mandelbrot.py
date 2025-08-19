from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def mandelbrot_escape_counts(
    xmin: float = -2.0,
    xmax: float = 1.0,
    ymin: float = -1.5,
    ymax: float = 1.5,
    width: int = 1000,
    height: int = 1000,
    max_iter: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Mandelbrot escape iteration counts.

    Returns (r1, r2, counts):
    - r1: 1D array of x-values of length `width`
    - r2: 1D array of y-values of length `height`
    - counts: 2D int array (height x width) of escape iterations (0..max_iter)
    """
    r1 = np.linspace(float(xmin), float(xmax), int(width))
    r2 = np.linspace(float(ymin), float(ymax), int(height))
    X, Y = np.meshgrid(r1, r2)
    C = X + 1j * Y

    Z = np.zeros_like(C, dtype=complex)
    counts = np.zeros(C.shape, dtype=np.int32)
    mask = np.ones(C.shape, dtype=bool)

    for n in range(1, int(max_iter) + 1):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped = np.greater(np.abs(Z), 2.0, where=mask)
        newly_escaped = escaped & mask
        counts[newly_escaped] = n
        mask &= ~escaped
        if not mask.any():
            break

    return r1, r2, counts


def normalized_counts(counts: np.ndarray, max_iter: int) -> np.ndarray:
    counts = counts.astype(float)
    return counts / float(max_iter)


def counts_to_dataframe(r1: np.ndarray, r2: np.ndarray, counts: np.ndarray) -> pd.DataFrame:
    H, W = counts.shape
    xx, yy = np.meshgrid(r1, r2)
    df = pd.DataFrame({
        "x": xx.ravel(),
        "y": yy.ravel(),
        "iter": counts.ravel(),
    })
    return df


def save_image(counts: np.ndarray, path: str, invert: bool = False) -> None:
    """Save counts as a grayscale PNG using Pillow.

    - Scales counts to 0..255 based on max value.
    - Flips vertically to mimic origin="lower" behavior.
    - Set invert=True to invert brightness (higher counts darker).
    """
    from PIL import Image, ImageOps

    arr = counts.astype(np.float32)
    maxv = float(arr.max()) if arr.size > 0 else 0.0
    if maxv > 0:
        arr = (arr / maxv) * 255.0
    arr = arr.astype(np.uint8)
    # Flip vertically so the lower y is at the bottom
    arr = np.flipud(arr)
    img = Image.fromarray(arr, mode="L")
    if invert:
        img = ImageOps.invert(img)
    img.save(path)
