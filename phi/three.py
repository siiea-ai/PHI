from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class ThreeConfig:
    strategy: str = "ratio"           # only 'ratio' supported (educational)
    ratio: int = 4                     # keep every Nth point
    method: str = "interp"             # 'interp' (linear blend) or 'nearest'


# --------- Point cloud I/O (PLY ASCII + NPZ helpers) ---------

def save_point_cloud_ply(points: np.ndarray, path: str) -> None:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{float(x)} {float(y)} {float(z)}\n")


def load_point_cloud_ply(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        header = True
        count = None
        for line in f:
            line = line.strip()
            if header:
                if line.startswith("element vertex"):
                    try:
                        count = int(line.split()[-1])
                    except Exception:
                        count = None
                if line == "end_header":
                    header = False
                    break
        pts = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                pts.append((x, y, z))
            except Exception:
                continue
    arr = np.asarray(pts, dtype=np.float32)
    if count is not None and len(arr) != count:
        # proceed anyway; PLY counts can be off in hand-edited files
        pass
    return arr


def save_point_cloud_npz(points: np.ndarray, path: str) -> None:
    """Save point cloud to NPZ (compressed) with key 'points'."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    np.savez_compressed(path, points=pts)


def load_point_cloud_npz(path: str) -> np.ndarray:
    with np.load(path) as data:
        pts = data.get("points")
        if pts is None:
            # Support plain .npy too
            try:
                pts = np.asarray(data, dtype=np.float32)
            except Exception:
                raise ValueError("NPZ missing 'points' array")
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


def save_point_cloud(points: np.ndarray, path: str) -> None:
    """Generic saver based on extension: .ply (ASCII), .npz (compressed), .npy."""
    lp = path.lower()
    if lp.endswith(".npz"):
        save_point_cloud_npz(points, path)
    elif lp.endswith(".npy"):
        np.save(path, np.asarray(points, dtype=np.float32).reshape(-1, 3), allow_pickle=False)
    else:
        save_point_cloud_ply(points, path)


def load_point_cloud(path: str) -> np.ndarray:
    """Generic loader based on extension: .ply (ASCII), .npz (compressed), .npy."""
    lp = path.lower()
    if lp.endswith(".npz"):
        return load_point_cloud_npz(path)
    if lp.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
        return np.asarray(arr, dtype=np.float32).reshape(-1, 3)
    return load_point_cloud_ply(path)


# --------- Bundle I/O ---------

def _points_to_b64(points: np.ndarray) -> str:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    buf = io.BytesIO()
    # Save as .npy bytes for compactness and speed
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_points(b64_str: str) -> np.ndarray:
    data = base64.b64decode(b64_str.encode("ascii"))
    buf = io.BytesIO(data)
    arr = np.load(buf, allow_pickle=False)
    return np.asarray(arr, dtype=np.float32).reshape(-1, 3)


def save_model(bundle: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


def load_model(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------- Generator (Sierpinski tetrahedron via chaos game) ---------

def generate_sierpinski_tetrahedron(n_points: int = 10000, scale: float = 1.0, burn_in: int = 200, seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Regular tetrahedron vertices centered at origin
    verts = np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
    ], dtype=np.float32)
    verts *= (scale / np.sqrt(3.0))

    p = rng.uniform(-0.5, 0.5, size=3).astype(np.float32)
    out = []
    total = burn_in + n_points
    for i in range(total):
        v = verts[rng.integers(0, 4)]
        p = (p + v) * 0.5  # midpoint
        if i >= burn_in:
            out.append(p.copy())
    return np.asarray(out, dtype=np.float32)


def _menger_filled(ix: int, iy: int, iz: int, level: int) -> bool:
    """True if integer cell (ix,iy,iz) is part of Menger sponge (not removed)."""
    for _ in range(level):
        cx = ix % 3
        cy = iy % 3
        cz = iz % 3
        if (cx == 1) + (cy == 1) + (cz == 1) >= 2:
            return False
        ix //= 3
        iy //= 3
        iz //= 3
    return True


def generate_menger_sponge(n_points: int = 10000, level: int = 3, scale: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
    """Rejection-sampled Menger sponge point cloud (centered at origin).

    Uses integer grid at resolution 3**level and keeps cells that remain after
    the Menger removal rule. Adds uniform jitter inside each kept cell.
    """
    rng = np.random.default_rng(seed)
    n = int(3 ** int(level))
    pts = []
    max_tries = max(50 * n_points, 10000)
    tries = 0
    while len(pts) < n_points and tries < max_tries:
        tries += 1
        ix = int(rng.integers(0, n))
        iy = int(rng.integers(0, n))
        iz = int(rng.integers(0, n))
        if not _menger_filled(ix, iy, iz, int(level)):
            continue
        # Jitter inside the cell and normalize to [-0.5, 0.5]
        x = (ix + rng.random()) / n - 0.5
        y = (iy + rng.random()) / n - 0.5
        z = (iz + rng.random()) / n - 0.5
        pts.append((x * scale, y * scale, z * scale))
    return np.asarray(pts, dtype=np.float32)


def generate_mandelbulb_julia(
    n_points: int = 10000,
    power: int = 8,
    c: Tuple[float, float, float] = (0.2, 0.35, 0.0),
    bounds: float = 1.5,
    max_iter: int = 20,
    bail: float = 4.0,
    scale: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Simple 3D Julia-style set using Mandelbulb-like iteration.

    Iterates z -> z^power + c in spherical coordinates. Collects initial
    seeds that do not escape. Educational approximation; not optimized.
    """
    rng = np.random.default_rng(seed)
    out = []
    batch = 4096
    tries = 0
    def bulb_pow(z: np.ndarray) -> np.ndarray:
        # Convert to spherical
        x, y, zc = z[:, 0], z[:, 1], z[:, 2]
        r = np.sqrt(x * x + y * y + zc * zc) + 1e-12
        theta = np.arccos(np.clip(zc / r, -1.0, 1.0))
        phi = np.arctan2(y, x)
        rp = r ** power
        thetap = theta * power
        phip = phi * power
        sint = np.sin(thetap)
        return np.stack([
            rp * sint * np.cos(phip),
            rp * sint * np.sin(phip),
            rp * np.cos(thetap),
        ], axis=1)

    cvec = np.array(c, dtype=np.float32)[None, :]
    while len(out) < n_points and tries < 1000:
        tries += 1
        U = rng.uniform(-bounds, bounds, size=(batch, 3)).astype(np.float32)
        Z = U.copy()
        escaped = np.zeros((batch,), dtype=bool)
        for _ in range(max_iter):
            Z = bulb_pow(Z) + cvec
            r2 = np.sum(Z * Z, axis=1)
            escaped |= (r2 > bail * bail)
            if escaped.all():
                break
        keep = ~escaped
        if not np.any(keep):
            continue
        K = U[keep] * (scale / bounds)
        out.extend(K.tolist())
        if len(out) > n_points:
            out = out[:n_points]
    return np.asarray(out, dtype=np.float32)


# --------- Core compress/expand (ratio strategy) ---------

def compress_point_cloud(points: np.ndarray, config: Optional[ThreeConfig] = None) -> Dict:
    cfg = config or ThreeConfig()
    if cfg.strategy.lower() != "ratio":
        raise ValueError("Only 'ratio' strategy is supported for 3D")
    if cfg.ratio < 1:
        raise ValueError("ratio must be >= 1")

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    ds = pts[:: cfg.ratio]

    pmin = np.min(pts, axis=0).astype(np.float32).tolist()
    pmax = np.max(pts, axis=0).astype(np.float32).tolist()

    bundle = {
        "version": 1,
        "type": "phi-three-model",
        "strategy": "ratio",
        "ratio": int(cfg.ratio),
        "method": cfg.method,
        "bbox_min": pmin,
        "bbox_max": pmax,
        "orig_count": int(len(pts)),
        "ds_count": int(len(ds)),
        "points_b64": _points_to_b64(ds),
    }
    return bundle


def expand_point_cloud(bundle: Dict, target_points: Optional[int] = None, method: Optional[str] = None) -> np.ndarray:
    if bundle.get("type") != "phi-three-model":
        raise ValueError("Not a 3D model bundle")
    base = _b64_to_points(bundle.get("points_b64", ""))
    if base.size == 0:
        raise ValueError("Model contains no points")

    m = (method or bundle.get("method", "interp")).lower()
    ratio = int(bundle.get("ratio", 1))
    target = int(bundle.get("orig_count", len(base) * ratio)) if target_points is None else int(target_points)

    if target <= len(base):
        return base[:target]

    rng = np.random.default_rng()
    out = [p for p in base]
    if m == "nearest":
        reps = target - len(base)
        idx = rng.integers(0, len(base), size=reps)
        out.extend(base[idx])
    elif m == "interp":
        # Sample random linear blends between random pairs of base points
        need = target - len(base)
        i0 = rng.integers(0, len(base), size=need)
        i1 = rng.integers(0, len(base), size=need)
        a = rng.random(size=need, dtype=np.float32)
        blends = (1.0 - a)[:, None] * base[i0] + a[:, None] * base[i1]
        out.extend(blends)
    else:
        raise ValueError(f"Unknown method: {m}")

    return np.asarray(out[:target], dtype=np.float32)


# --------- Metrics + Compare ---------

def metrics_from_paths(
    orig_path: str,
    recon_path: str,
    sample_points: int = 2000,
    chunk: int = 256,
    seed: Optional[int] = None,
    nn_method: str = "auto",  # 'auto'|'kd'|'sklearn'|'brute'
) -> "pd.DataFrame":
    """Approximate symmetric Chamfer distance between two point clouds loaded from PLY files.

    Returns a pandas DataFrame with columns: chamfer2 (avg squared), chamfer (RMSE), samples_a, samples_b.
    """
    import pandas as pd  # local import to keep dependency optional

    A = load_point_cloud(orig_path)
    B = load_point_cloud(recon_path)
    rng = np.random.default_rng(seed)

    sa = min(sample_points, len(A))
    sb = min(sample_points, len(B))
    if sa == 0 or sb == 0:
        return pd.DataFrame([{"chamfer2": np.nan, "chamfer": np.nan, "samples_a": int(sa), "samples_b": int(sb)}])

    Ai = rng.choice(len(A), size=sa, replace=False)
    Bi = rng.choice(len(B), size=sb, replace=False)
    As = A[Ai]
    Bs = B[Bi]

    def nn_dist_sq(X: np.ndarray, Y: np.ndarray, chunk: int = 256) -> np.ndarray:
        m = nn_method.lower()
        # Try SciPy KD-tree
        if m in ("auto", "kd"):
            try:
                from scipy.spatial import cKDTree  # type: ignore
                tree = cKDTree(Y)
                try:
                    d, _ = tree.query(X, k=1, workers=-1)
                except TypeError:
                    # older SciPy uses 'n_jobs'
                    d, _ = tree.query(X, k=1)
                return (d.astype(np.float64) ** 2.0)
            except Exception:
                if m == "kd":
                    # explicit request but unavailable -> fall through to brute
                    pass
        # Try scikit-learn
        if m in ("auto", "sklearn"):
            try:
                from sklearn.neighbors import NearestNeighbors  # type: ignore
                nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
                nn.fit(Y)
                d, _ = nn.kneighbors(X, n_neighbors=1, return_distance=True)
                return (d[:, 0].astype(np.float64) ** 2.0)
            except Exception:
                if m == "sklearn":
                    pass
        # Brute-force fallback (chunked to bound memory)
        outs = []
        for st in range(0, len(X), chunk):
            x = X[st : st + chunk]
            diff = x[:, None, :] - Y[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            outs.append(d2.min(axis=1))
        return np.concatenate(outs, axis=0)

    d1 = nn_dist_sq(As, Bs, chunk=chunk)
    d2 = nn_dist_sq(Bs, As, chunk=chunk)
    chamfer2 = float(d1.mean() + d2.mean())
    chamfer = float(np.sqrt(chamfer2))

    return pd.DataFrame([
        {"chamfer2": chamfer2, "chamfer": chamfer, "samples_a": int(sa), "samples_b": int(sb)}
    ])


def _render_projection(points: np.ndarray, size: Tuple[int, int] = (400, 400), axis: str = "z") -> Image.Image:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if axis not in ("x", "y", "z"):
        axis = "z"
    # choose projection plane
    if axis == "z":
        P = pts[:, [0, 1]]
    elif axis == "y":
        P = pts[:, [0, 2]]
    else:
        P = pts[:, [1, 2]]

    # normalize to 0..1 with small padding
    pmin = P.min(axis=0)
    pmax = P.max(axis=0)
    span = np.maximum(pmax - pmin, 1e-6)
    U = (P - pmin) / span

    w, h = size
    img = Image.new("RGB", (w, h), color="white")
    draw = ImageDraw.Draw(img)
    # draw points
    xs = (U[:, 0] * (w - 1)).astype(int)
    ys = ( (1.0 - U[:, 1]) * (h - 1) ).astype(int)  # flip y for display
    for x, y in zip(xs, ys):
        img.putpixel((int(x), int(y)), (0, 0, 0))
    return img


def save_compare_projection_image(orig_path: str, recon_path: str, output_image: str, axis: str = "z", height: int = 400) -> None:
    A = load_point_cloud(orig_path)
    B = load_point_cloud(recon_path)
    # keep aspect 1:1 for each, final is side-by-side
    left = _render_projection(A, size=(height, height), axis=axis)
    right = _render_projection(B, size=(height, height), axis=axis)
    canvas = Image.new("RGB", (left.width + right.width, height), color="white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(output_image)


def save_projection_from_ply(ply_path: str, output_image: str, axis: str = "z", height: int = 400) -> None:
    """Render a single point cloud PLY file into a 2D projection image."""
    P = load_point_cloud(ply_path)
    img = _render_projection(P, size=(height, height), axis=axis)
    img.save(output_image)


def plot_point_cloud_matplotlib(points: np.ndarray, save_path: Optional[str] = None, show: bool = False, size: float = 1.0, elev: float = 20, azim: float = 30) -> None:
    """Optional Matplotlib 3D scatter helper.

    Import is local to keep matplotlib optional. If unavailable, raises a
    descriptive error. When save_path is provided, saves a PNG.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        raise RuntimeError("matplotlib is required for 3D plotting; pip install matplotlib") from e

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c='k', marker='.', alpha=0.8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close(fig)
