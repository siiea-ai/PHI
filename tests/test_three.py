from pathlib import Path

import importlib.util
import numpy as np
import pytest

from phi import three as three_mod


def test_generators_basic_shapes():
    a = three_mod.generate_sierpinski_tetrahedron(n_points=2000, scale=1.0, seed=1)
    b = three_mod.generate_menger_sponge(n_points=1500, level=2, scale=1.0, seed=2)
    c = three_mod.generate_mandelbulb_julia(n_points=800, power=8, c=(0.2, 0.35, 0.0), seed=3)
    for pts in (a, b, c):
        assert isinstance(pts, np.ndarray)
        assert pts.shape[1] == 3
        assert pts.ndim == 2
        # Should be finite
        assert np.isfinite(pts).all()


def test_generic_io_roundtrip(tmp_path: Path):
    pts = three_mod.generate_sierpinski_tetrahedron(n_points=1000, scale=1.0, seed=0)

    # PLY
    ply_p = tmp_path / "cloud.ply"
    three_mod.save_point_cloud(pts, str(ply_p))
    pts_ply = three_mod.load_point_cloud(str(ply_p))
    assert pts_ply.shape[1] == 3

    # NPZ
    npz_p = tmp_path / "cloud.npz"
    three_mod.save_point_cloud(pts, str(npz_p))
    pts_npz = three_mod.load_point_cloud(str(npz_p))
    assert pts_npz.shape == pts_ply.shape

    # NPY
    npy_p = tmp_path / "cloud.npy"
    three_mod.save_point_cloud(pts, str(npy_p))
    pts_npy = three_mod.load_point_cloud(str(npy_p))
    assert pts_npy.shape == pts_ply.shape


def test_compress_expand_and_metrics(tmp_path: Path):
    pts = three_mod.generate_sierpinski_tetrahedron(n_points=3000, seed=0)
    cfg = three_mod.ThreeConfig(strategy="ratio", ratio=5, method="interp")
    bundle = three_mod.compress_point_cloud(pts, cfg)
    assert bundle["type"] == "phi-three-model"
    assert int(bundle["ratio"]) == 5

    recon = three_mod.expand_point_cloud(bundle, target_points=len(pts), method="interp")
    assert recon.shape[0] == len(pts)
    assert recon.shape[1] == 3

    # Save orig/recon and compute metrics (brute to avoid optional deps)
    orig_p = tmp_path / "orig.ply"
    recon_p = tmp_path / "recon.ply"
    three_mod.save_point_cloud(pts, str(orig_p))
    three_mod.save_point_cloud(recon, str(recon_p))

    mdf = three_mod.metrics_from_paths(str(orig_p), str(recon_p), sample_points=800, nn_method="brute")
    assert hasattr(mdf, "to_csv")
    assert set(["chamfer", "chamfer2"]).issubset(mdf.columns)


def test_metrics_kdtree_if_scipy(tmp_path: Path):
    # If SciPy is available, KD-tree path should run without error
    if importlib.util.find_spec("scipy") is None:
        pytest.skip("SciPy not installed")
    pts = three_mod.generate_sierpinski_tetrahedron(n_points=1200, seed=10)
    cfg = three_mod.ThreeConfig(strategy="ratio", ratio=3, method="nearest")
    bundle = three_mod.compress_point_cloud(pts, cfg)
    recon = three_mod.expand_point_cloud(bundle, target_points=len(pts), method="nearest")
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    three_mod.save_point_cloud(pts, str(a))
    three_mod.save_point_cloud(recon, str(b))
    mdf = three_mod.metrics_from_paths(str(a), str(b), sample_points=500, nn_method="kd")
    assert len(mdf) == 1


def test_metrics_sklearn_if_available(tmp_path: Path):
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn not installed")
    pts = three_mod.generate_sierpinski_tetrahedron(n_points=1200, seed=11)
    cfg = three_mod.ThreeConfig(strategy="ratio", ratio=3, method="nearest")
    bundle = three_mod.compress_point_cloud(pts, cfg)
    recon = three_mod.expand_point_cloud(bundle, target_points=len(pts), method="nearest")
    a = tmp_path / "a.ply"
    b = tmp_path / "b.ply"
    three_mod.save_point_cloud(pts, str(a))
    three_mod.save_point_cloud(recon, str(b))
    mdf = three_mod.metrics_from_paths(str(a), str(b), sample_points=500, nn_method="sklearn")
    assert len(mdf) == 1


def test_plotting_helper_handles_matplotlib(tmp_path: Path):
    pts = three_mod.generate_sierpinski_tetrahedron(n_points=1000, seed=7)
    out = tmp_path / "plot.png"
    try:
        import matplotlib.pyplot as _  # noqa: F401
    except Exception:
        # If matplotlib is missing, the helper should raise a clear error
        with pytest.raises(RuntimeError):
            three_mod.plot_point_cloud_matplotlib(pts, save_path=str(out), show=False, size=1.0)
        return

    # Matplotlib available: plotting should create a file without raising
    three_mod.plot_point_cloud_matplotlib(pts, save_path=str(out), show=False, size=1.0)
    assert out.exists()
