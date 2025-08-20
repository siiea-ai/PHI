import numpy as np

from phi.signals import (
    compute_psd_welch,
    bandpower_welch,
    psd_slope,
    pac_tort_mi,
    higuchi_fd,
    sample_entropy,
    multiscale_entropy,
    lzc,
    feature_vector,
    compute_metrics,
)


def make_signal(fs: float = 256.0, seconds: float = 5.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    # mixture: theta 6 Hz + gamma 40 Hz with noise
    x = np.sin(2 * np.pi * 6.0 * t) + 0.5 * np.sin(2 * np.pi * 40.0 * t) + 0.3 * rng.standard_normal(n)
    return x, fs


def test_feature_vector_and_names():
    x, fs = make_signal()
    feats, names = feature_vector(x, fs)
    assert feats.ndim == 1
    assert len(feats) == len(names)
    # Expect 5 bandpowers + psd_slope + lzc + pac = 8
    assert len(feats) == 8
    assert len(set(names)) == len(names)


def test_metrics_are_finite():
    x, fs = make_signal()
    f, pxx = compute_psd_welch(x, fs)
    assert f.ndim == 1 and pxx.ndim == 1 and f.size == pxx.size and f.size > 0

    bp = bandpower_welch(x, fs, (8.0, 13.0))
    assert np.isfinite(bp)

    slope = psd_slope(x, fs)
    assert np.isfinite(slope)

    mi, extras = pac_tort_mi(x, fs)
    assert mi >= 0.0
    assert "phase_bins" in extras and "amp_dist" in extras

    hfd = higuchi_fd(x)
    assert np.isfinite(hfd)

    se = sample_entropy(x)
    assert np.isfinite(se) or np.isinf(se)

    mse = multiscale_entropy(x, max_scale=5)
    assert isinstance(mse, dict) and 1 in mse

    c = lzc(x)
    assert np.isfinite(c)

    m = compute_metrics(x, fs)
    assert np.isfinite(m.psd_slope)
    assert np.isfinite(m.pac_tg_mi)
    assert np.isfinite(m.higuchi_fd)
    assert np.isfinite(m.lzc)
    assert isinstance(m.bandpowers, dict) and len(m.bandpowers) >= 3
