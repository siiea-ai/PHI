"""
phi.signals

Robust neuro-signal utilities and metrics:
- Welch PSD and bandpower
- 1/f PSD slope (log-log)
- Phase-Amplitude Coupling (Tort MI)
- Higuchi Fractal Dimension
- Multiscale Entropy (sample entropy across scales)
- Lempel–Ziv Complexity (binary LZ76)

All functions operate on 1D numpy arrays (float64 recommended) and are designed
for small-to-medium scale analysis with minimal dependencies (numpy, scipy).

Example:
    import numpy as np
    from phi.signals import psd_slope, pac_tort_mi, higuchi_fd, multiscale_entropy, lzc

    fs = 256.0
    x = np.random.randn(int(30*fs))
    slope = psd_slope(x, fs, fmin=1.0, fmax=40.0)
    mi, _ = pac_tort_mi(x, fs)
    hfd = higuchi_fd(x)
    mse = multiscale_entropy(x, m=2, r=None, max_scale=10)
    c = lzc(x)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, welch, hilbert


# ---------------------------- Filtering & PSD ---------------------------- #

def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Args:
        x: 1D signal.
        fs: Sampling rate (Hz).
        low: Low cutoff (Hz).
        high: High cutoff (Hz).
        order: Filter order.
    Returns:
        Filtered signal, same shape as x.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("bandpass_filter expects a 1D array")
    nyq = 0.5 * fs
    low_n = max(1e-6, low / nyq)
    high_n = min(0.999999, high / nyq)
    if not (0 < low_n < high_n < 1):
        raise ValueError("Invalid band for bandpass_filter")
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, x)


def compute_psd_welch(x: np.ndarray, fs: float, nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PSD via Welch's method.

   Returns (freqs, psd)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("compute_psd_welch expects a 1D array")
    if nperseg is None:
        nperseg = min(1024, max(256, int(fs * 2)))
    # Ensure parameters are valid for short inputs
    effective = int(min(nperseg, x.size))
    effective = max(2, effective)
    noverlap = min(nperseg // 2, effective - 1)
    freqs, pxx = welch(x, fs=fs, nperseg=effective, noverlap=noverlap, detrend="constant")
    return freqs, pxx + 1e-20  # avoid log(0)


def bandpower_welch(x: np.ndarray, fs: float, band: Tuple[float, float], nperseg: Optional[int] = None) -> float:
    """Integrate the Welch PSD over a frequency band (absolute power)."""
    fmin, fmax = band
    freqs, pxx = compute_psd_welch(x, fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    # Integrate using trapezoidal rule
    _trap = getattr(np, "trapezoid", np.trapz)
    return float(_trap(pxx[mask], freqs[mask]))


def psd_slope(x: np.ndarray, fs: float, fmin: float = 1.0, fmax: float = 40.0, nperseg: Optional[int] = None) -> float:
    """Estimate 1/f PSD slope via linear regression in log-log space.

    Typical EEG shows approximately 1/f^k behavior with k in [0.5, 2]. The slope
    returned here is the coefficient for log10(P) ~ a + slope * log10(f).
    """
    freqs, pxx = compute_psd_welch(x, fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax) & (pxx > 0)
    if np.count_nonzero(mask) < 5:
        return float("nan")
    xf = np.log10(freqs[mask])
    yf = np.log10(pxx[mask])
    slope, intercept = np.polyfit(xf, yf, 1)
    return float(slope)


# ------------------------ Phase-Amplitude Coupling ----------------------- #

def pac_tort_mi(
    x: np.ndarray,
    fs: float,
    phase_band: Tuple[float, float] = (4.0, 8.0),
    amp_band: Tuple[float, float] = (30.0, 90.0),
    n_bins: int = 18,
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Tort et al. Modulation Index (MI) for PAC.

    Steps:
    - Bandpass to phase_band and amp_band
    - Phase = angle(hilbert(phase_signal))
    - Amp envelope = abs(hilbert(amp_signal))
    - Bin phases, compute mean amp per bin, normalize to prob. dist.
    - MI = (KL divergence of this distribution to uniform) / log(N)

    Returns (mi, extras) where extras contains phase_bins (rad), amp_dist.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("pac_tort_mi expects a 1D array")

    ph = bandpass_filter(x, fs, *phase_band)
    am = bandpass_filter(x, fs, *amp_band)

    ph_phase = np.angle(hilbert(ph))
    am_env = np.abs(hilbert(am)) + 1e-12

    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # mean amplitude per phase bin
    amp_means = np.zeros(n_bins)
    for i in range(n_bins):
        sel = (ph_phase >= bins[i]) & (ph_phase < bins[i + 1])
        if np.any(sel):
            amp_means[i] = np.mean(am_env[sel])
        else:
            amp_means[i] = 0.0

    if amp_means.sum() == 0.0:
        return 0.0, {"phase_bins": bin_centers, "amp_dist": amp_means}

    p = amp_means / amp_means.sum()
    u = np.ones_like(p) / len(p)
    kl = np.sum(p * np.log((p + 1e-12) / u))
    mi = float(kl / np.log(len(p)))
    return mi, {"phase_bins": bin_centers, "amp_dist": p}


# ----------------------- Higuchi Fractal Dimension ---------------------- #

def higuchi_fd(x: np.ndarray, kmax: int = 8) -> float:
    """Compute the Higuchi fractal dimension.

    Reference: Higuchi, T. Physica D (1988). Complexity ~ 1..2 for time series.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if N < (kmax + 1):
        return float("nan")

    L = np.zeros(kmax)
    x_mean = np.mean(x)
    for k in range(1, kmax + 1):
        Lmk = []
        for m in range(k):
            idx = np.arange(m, N, k)
            xk = x[idx]
            if xk.size < 2:
                continue
            diff = np.abs(np.diff(xk))
            n = xk.size
            norm = (N - 1) / (k * (n - 1))
            Lm = (diff.sum() * norm) if n > 1 else 0.0
            Lmk.append(Lm)
        if len(Lmk) == 0:
            L[k - 1] = np.nan
        else:
            L[k - 1] = np.mean(Lmk)

    # Linear fit log(L(k)) ~ a + b*log(1/k) => FD = -b
    ks = np.arange(1, kmax + 1)
    valid = np.isfinite(L) & (L > 0)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    y = np.log(L[valid])
    xlog = np.log(1.0 / ks[valid])
    b, a = np.polyfit(xlog, y, 1)
    return float(-b)


# ---------------------------- Sample Entropy ---------------------------- #

def _phi_m(x: np.ndarray, m: int, r: float) -> float:
    """Helper for sample entropy: count matches of length m within tolerance r."""
    N = x.size
    if N <= m + 1:
        return 0.0
    xm = np.array([x[i : i + m] for i in range(N - m + 1)])
    count = 0
    for i in range(N - m):
        dist = np.max(np.abs(xm[i + 1 :] - xm[i]), axis=1)
        count += np.count_nonzero(dist <= r)
    denom = (N - m) * (N - m - 1) / 2
    return count / max(1, int(denom))


def sample_entropy(x: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """Compute sample entropy (m, r). Returns -ln( A_{m+1} / A_m )."""
    x = np.asarray(x, dtype=float)
    if r is None:
        r = 0.15 * np.std(x)
    A_m = _phi_m(x, m, r)
    A_m1 = _phi_m(x, m + 1, r)
    if A_m1 == 0 or A_m == 0:
        return float("inf")
    return float(-np.log(A_m1 / A_m))


def multiscale_entropy(x: np.ndarray, m: int = 2, r: Optional[float] = None, max_scale: int = 10) -> Dict[int, float]:
    """Compute Multiscale Entropy (MSE) up to a maximum scale.

    Coarse-grain at each scale S by averaging non-overlapping windows of length S.
    Returns a dict: {scale -> SampEn}.
    """
    x = np.asarray(x, dtype=float)
    if r is None:
        r = 0.15 * np.std(x)
    mse: Dict[int, float] = {}
    for S in range(1, max_scale + 1):
        if S == 1:
            xS = x
        else:
            L = x.size // S
            if L < (m + 2):  # ensure enough data
                mse[S] = float("nan")
                continue
            xS = x[: L * S].reshape(L, S).mean(axis=1)
        mse[S] = sample_entropy(xS, m=m, r=r)
    return mse


# ------------------------- Lempel–Ziv Complexity ------------------------ #

def lzc(x: np.ndarray, normalize: bool = True, threshold: str = "median") -> float:
    """Compute Lempel–Ziv complexity (LZ76) on a binarized sequence.

    Args:
        x: 1D signal
        normalize: if True, normalize by (n / log2(n)) for comparability
        threshold: 'median' or 'mean' for binarization
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("lzc expects a 1D array")
    thr = np.median(x) if threshold == "median" else np.mean(x)
    s = (x > thr).astype(np.int8)
    n = s.size
    if n < 2:
        return 0.0

    # LZ76 parsing
        # LZ76 parsing
    i = 0
    c = 1
    k = 1
    k_max = 1
    while True:
        if (i + k > n) or (k_max + k > n):
            c += 1
            break
        if np.array_equal(s[i : i + k], s[k_max : k_max + k]):
            k += 1
        else:
            k_max += 1
            if k_max == i + k:
                c += 1
                i = i + k
                if i + 1 > n:
                    break
                k = 1
                k_max = i + 1
    if not normalize:
        return float(c)
    return float(c * np.log2(n) / n)


# --------------------------- Feature extraction ------------------------- #

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 90.0),
}


def feature_vector(x: np.ndarray, fs: float) -> Tuple[np.ndarray, List[str]]:
    """Compute a compact feature vector useful for online decoding.

    Includes bandpowers for delta/theta/alpha/beta/gamma, PSD slope, LZC, and
    PAC MI between theta phase and gamma amplitude.
    """
    feats: List[float] = []
    names: List[str] = []

    # Bandpowers
    for name, band in BANDS.items():
        feats.append(bandpower_welch(x, fs, band))
        names.append(f"bp_{name}")

    # PSD slope
    feats.append(psd_slope(x, fs))
    names.append("psd_slope")

    # LZC
    feats.append(lzc(x))
    names.append("lzc")

    # PAC (theta-gamma)
    mi, _ = pac_tort_mi(x, fs, phase_band=BANDS["theta"], amp_band=BANDS["gamma"])  # type: ignore
    feats.append(mi)
    names.append("pac_tg_mi")

    return np.asarray(feats, dtype=float), names


# ------------------------------ Utilities ------------------------------- #

@dataclass
class SignalMetrics:
    psd_slope: float
    pac_tg_mi: float
    higuchi_fd: float
    lzc: float
    bandpowers: Dict[str, float]


def compute_metrics(x: np.ndarray, fs: float) -> SignalMetrics:
    """Convenience wrapper computing several scalar metrics."""
    bp = {k: bandpower_welch(x, fs, v) for k, v in BANDS.items()}
    return SignalMetrics(
        psd_slope=psd_slope(x, fs),
        pac_tg_mi=pac_tort_mi(x, fs, phase_band=BANDS["theta"], amp_band=BANDS["gamma"])[0],
        higuchi_fd=higuchi_fd(x),
        lzc=lzc(x),
        bandpowers=bp,
    )
