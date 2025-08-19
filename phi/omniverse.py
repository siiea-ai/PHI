from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class OmniverseConfig:
    strategy: str = "ratio"  # only 'ratio' supported
    spatial_ratio: int = 2    # keep every Nth pixel per axis
    layer_ratio: int = 1      # keep every Nth layer within each universe
    universe_ratio: int = 1   # keep every Nth universe
    method: str = "interp"    # 'interp' (bilinear/linear) or 'nearest'


# -----------------------------------------------------------------------------
# Helpers: base64 encode/decode numpy arrays as NPZ
# -----------------------------------------------------------------------------


def _arr_to_b64_npz(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, arr=arr.astype(np.float32, copy=False))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_npz_to_arr(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("ascii"))
    buf = io.BytesIO(raw)
    with np.load(buf) as data:
        arr = data["arr"]
    return np.asarray(arr, dtype=np.float32)


# -----------------------------------------------------------------------------
# Interpolation helpers
# -----------------------------------------------------------------------------


def _interp_resize2d(arr: np.ndarray, out_wh: Tuple[int, int], method: str = "interp") -> np.ndarray:
    h, w = arr.shape
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if (w, h) == (out_w, out_h):
        return arr.astype(np.float32, copy=False)
    try:
        from scipy.ndimage import zoom  # type: ignore

        zx = out_h / float(h)
        zy = out_w / float(w)
        order = 1 if method == "interp" else 0
        out = zoom(arr, (zx, zy), order=order, prefilter=False)
        out = np.asarray(out, dtype=np.float32)
        if out.shape != (out_h, out_w):
            out = out[:out_h, :out_w]
            if out.shape[0] < out_h or out.shape[1] < out_w:
                pad_h = max(0, out_h - out.shape[0])
                pad_w = max(0, out_w - out.shape[1])
                out = np.pad(out, ((0, pad_h), (0, pad_w)), mode="edge")
        return out
    except Exception:
        sx = max(1, int(round(out_w / w)))
        sy = max(1, int(round(out_h / h)))
        out = np.repeat(np.repeat(arr, sy, axis=0), sx, axis=1)
        return out[:out_h, :out_w].astype(np.float32)


def _interp_resize_along_axis(arr: np.ndarray, axis: int, out_len: int, method: str = "interp") -> np.ndarray:
    # Generic 1D interpolation/nearest along an axis for N-D arrays
    if out_len < 1:
        out_len = 1
    old_len = arr.shape[axis]
    if old_len == out_len:
        return arr.astype(np.float32, copy=False)

    if method == "nearest":
        idx = np.linspace(0, old_len - 1, out_len)
        idx = np.rint(idx).astype(int)
        idx = np.clip(idx, 0, old_len - 1)
        out = np.take(arr, idx, axis=axis)
        return np.asarray(out, dtype=np.float32)

    # linear interpolation
    x_old = np.linspace(0.0, 1.0, old_len, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    moved = np.moveaxis(arr, axis, 0)  # (old_len, ...)
    flat = moved.reshape(old_len, -1)
    out_flat = np.empty((out_len, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out_flat[:, c] = np.interp(x_new, x_old, flat[:, c])
    out_moved = out_flat.reshape((out_len,) + moved.shape[1:])
    out = np.moveaxis(out_moved, 0, axis)
    return out.astype(np.float32)


def _omni_resize(grid: np.ndarray, out_w: int, out_h: int, out_L: int, out_U: int, method: str = "interp") -> np.ndarray:
    # grid: (U, L, H, W) -> (U2, L2, H2, W2)
    U, L, H, W = grid.shape
    # spatial resize per (u, l)
    spatial = np.empty((U, L, out_h, out_w), dtype=np.float32)
    for u in range(U):
        for l in range(L):
            spatial[u, l] = _interp_resize2d(grid[u, l], (out_w, out_h), method=method)
    # resize layers then universes
    layers_resized = _interp_resize_along_axis(spatial, axis=1, out_len=out_L, method=method)
    uni_resized = _interp_resize_along_axis(layers_resized, axis=0, out_len=out_U, method=method)
    return np.asarray(uni_resized, dtype=np.float32)


# -----------------------------------------------------------------------------
# Generation (4D grid of fBm-like fields with cross-coupling)
# -----------------------------------------------------------------------------


def _generate_field(width: int, height: int, octaves: int, rng: np.random.Generator) -> np.ndarray:
    field = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0
    for o in range(octaves):
        ow = max(1, width // (2 ** o))
        oh = max(1, height // (2 ** o))
        base = rng.standard_normal((oh, ow), dtype=np.float32)
        up = _interp_resize2d(base, (width, height), method="interp")
        field += amp * up
        amp_sum += amp
        amp *= 0.5
    field /= max(amp_sum, 1e-6)
    fmin, fmax = float(field.min()), float(field.max())
    if fmax > fmin:
        field = (field - fmin) / (fmax - fmin)
    else:
        field.fill(0.5)
    return field.astype(np.float32)


def generate_full_grid(width: int, height: int, layers: int, universes: int, octaves: int = 4, seed: Optional[int] = None) -> Dict:
    if width < 1 or height < 1 or layers < 1 or universes < 1:
        raise ValueError("width, height, layers, universes must be >= 1")
    if octaves < 1:
        raise ValueError("octaves must be >= 1")
    rng = np.random.default_rng(seed)

    grid = np.empty((universes, layers, height, width), dtype=np.float32)
    for u in range(universes):
        for L in range(layers):
            # vary via rng; generate per (universe, layer)
            base = _generate_field(width, height, octaves, rng)
            # cross-layer coupling
            if L > 0:
                base = np.clip(0.7 * base + 0.3 * np.sin(3.0 * grid[u, L - 1]), 0.0, 1.0)
            # cross-universe coupling
            if u > 0:
                base = np.clip(0.75 * base + 0.25 * np.cos(2.5 * grid[u - 1, L]), 0.0, 1.0)
            grid[u, L] = base

    bundle = {
        "version": 1,
        "type": "omniverse_full",
        "width": int(width),
        "height": int(height),
        "layers": int(layers),
        "universes": int(universes),
        "field_dtype": "float32",
        "grid_npz_b64": _arr_to_b64_npz(grid),
        "params": {"octaves": int(octaves), "seed": (int(seed) if seed is not None else None)},
    }
    return bundle


# -----------------------------------------------------------------------------
# Compress / Expand
# -----------------------------------------------------------------------------


def compress_grid(full_bundle: Dict, config: Optional[OmniverseConfig] = None) -> Dict:
    if full_bundle.get("type") != "omniverse_full":
        raise ValueError("Not a full omniverse bundle")
    cfg = config or OmniverseConfig()
    if cfg.strategy.lower() != "ratio":
        raise ValueError("Only 'ratio' strategy is supported for omniverse")
    if cfg.spatial_ratio < 1 or cfg.layer_ratio < 1 or cfg.universe_ratio < 1:
        raise ValueError("ratios must be >= 1")

    w = int(full_bundle.get("width", 0))
    h = int(full_bundle.get("height", 0))
    L = int(full_bundle.get("layers", 0))
    U = int(full_bundle.get("universes", 0))
    grid = _b64_npz_to_arr(full_bundle["grid_npz_b64"])  # (U, L, h, w)

    w_ds = max(1, w // cfg.spatial_ratio)
    h_ds = max(1, h // cfg.spatial_ratio)
    L_ds = max(1, L // cfg.layer_ratio)
    U_ds = max(1, U // cfg.universe_ratio)

    # spatial decimation via striding, then layer and universe decimation
    ds_spatial = grid[:, :, ::cfg.spatial_ratio, ::cfg.spatial_ratio]
    if ds_spatial.shape[2:] != (h_ds, w_ds):
        tmp = np.empty((U, L, h_ds, w_ds), dtype=np.float32)
        for u in range(U):
            for l in range(L):
                tmp[u, l] = _interp_resize2d(grid[u, l], (w_ds, h_ds), method="nearest")
        ds_spatial = tmp

    idx_layers = np.arange(0, L, cfg.layer_ratio)
    if len(idx_layers) < L_ds:
        idx_layers = np.linspace(0, L - 1, L_ds)
        idx_layers = np.rint(idx_layers).astype(int)
    ds_L = ds_spatial[:, idx_layers, :, :]

    idx_universes = np.arange(0, U, cfg.universe_ratio)
    if len(idx_universes) < U_ds:
        idx_universes = np.linspace(0, U - 1, U_ds)
        idx_universes = np.rint(idx_universes).astype(int)
    ds = ds_L[idx_universes, :, :, :]  # (U_ds, L_ds, h_ds, w_ds)

    bundle = {
        "version": 1,
        "type": "omniverse_ratio",
        "strategy": "ratio",
        "spatial_ratio": int(cfg.spatial_ratio),
        "layer_ratio": int(cfg.layer_ratio),
        "universe_ratio": int(cfg.universe_ratio),
        "orig_size": [int(w), int(h), int(L), int(U)],
        "ds_size": [int(w_ds), int(h_ds), int(L_ds), int(U_ds)],
        "ds_dtype": "float32",
        "ds_npz_b64": _arr_to_b64_npz(ds),
        "config": {"method": cfg.method},
    }
    return bundle


essential_keys = ["omniverse_full", "omniverse_ratio"]


def expand_grid(bundle: Dict, target_size: Optional[Tuple[int, int, int, int]] = None, method: Optional[str] = None) -> Dict:
    btype = bundle.get("type")
    if btype == "omniverse_full":
        grid = _b64_npz_to_arr(bundle["grid_npz_b64"])  # (U, L, h, w)
        w = int(bundle.get("width", grid.shape[3]))
        h = int(bundle.get("height", grid.shape[2]))
        L = int(bundle.get("layers", grid.shape[1]))
        U = int(bundle.get("universes", grid.shape[0]))
        if target_size is not None:
            tw, th, tL, tU = int(target_size[0]), int(target_size[1]), int(target_size[2]), int(target_size[3])
            out = _omni_resize(grid, tw, th, tL, tU, method=(method or "interp"))
            return {
                "version": 1,
                "type": "omniverse_full",
                "width": int(tw),
                "height": int(th),
                "layers": int(tL),
                "universes": int(tU),
                "field_dtype": "float32",
                "grid_npz_b64": _arr_to_b64_npz(out),
                "params": bundle.get("params", {}),
            }
        else:
            return bundle

    if btype != "omniverse_ratio":
        raise ValueError("Not an omniverse ratio bundle")

    w, h, L, U = (int(bundle["orig_size"][0]), int(bundle["orig_size"][1]), int(bundle["orig_size"][2]), int(bundle["orig_size"][3]))
    ds = _b64_npz_to_arr(bundle["ds_npz_b64"])  # (U_ds, L_ds, h_ds, w_ds)
    out_w, out_h, out_L, out_U = (int(target_size[0]), int(target_size[1]), int(target_size[2]), int(target_size[3])) if target_size else (w, h, L, U)
    meth = (method or bundle.get("config", {}).get("method", "interp")).lower()
    grid = _omni_resize(ds, out_w, out_h, out_L, out_U, method=meth)
    grid = np.clip(grid, 0.0, 1.0)

    full = {
        "version": 1,
        "type": "omniverse_full",
        "width": int(out_w),
        "height": int(out_h),
        "layers": int(out_L),
        "universes": int(out_U),
        "field_dtype": "float32",
        "grid_npz_b64": _arr_to_b64_npz(grid),
        "params": {"expanded_from": "ratio", "spatial_ratio": int(bundle.get("spatial_ratio", 1)), "layer_ratio": int(bundle.get("layer_ratio", 1)), "universe_ratio": int(bundle.get("universe_ratio", 1)), **bundle.get("config", {})},
    }
    return full


# -----------------------------------------------------------------------------
# Visualization (mosaic) & Compare
# -----------------------------------------------------------------------------


def _field_to_image(field: np.ndarray, cmap: str = "viridis") -> Image.Image:
    a = np.asarray(field, dtype=np.float32)
    a = a - float(a.min())
    maxv = float(a.max())
    if maxv > 0:
        a = a / maxv
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    fig.set_size_inches(a.shape[1] / 100.0, a.shape[0] / 100.0)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(a, cmap=cmap, origin="lower")
    buf = io.BytesIO()
    fig.canvas.print_png(buf)
    plt.close(fig)
    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


def save_mosaic_from_model(bundle: Dict, output_path: str, cmap: str = "viridis", cols: Optional[int] = None, rows: Optional[int] = None) -> None:
    btype = bundle.get("type")
    if btype == "omniverse_full":
        grid = _b64_npz_to_arr(bundle["grid_npz_b64"])  # (U, L, h, w)
    elif btype == "omniverse_ratio":
        w, h, L, U = (int(bundle["orig_size"][0]), int(bundle["orig_size"][1]), int(bundle["orig_size"][2]), int(bundle["orig_size"][3]))
        ds = _b64_npz_to_arr(bundle["ds_npz_b64"])  # (U_ds, L_ds, h_ds, w_ds)
        grid = _omni_resize(ds, w, h, L, U, method=bundle.get("config", {}).get("method", "interp"))
    else:
        raise ValueError("Unknown bundle type")

    U, L, h, w = grid.shape
    # choose grid
    total = U * L
    if cols is None or rows is None:
        cols = int(np.ceil(np.sqrt(total)))
        rows = int(np.ceil(total / cols))

    # render each (u, l)
    tiles = []
    for u in range(U):
        for l in range(L):
            tiles.append(_field_to_image(grid[u, l], cmap=cmap))
    tile_w = max(img.width for img in tiles)
    tile_h = max(img.height for img in tiles)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color="white")

    for idx, img in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), resample=Image.BILINEAR)
        canvas.paste(img, (c * tile_w, r * tile_h))

    canvas.save(output_path)


def save_compare_mosaic(orig_bundle: Dict, recon_bundle: Dict, output_path: str, cmap: str = "viridis") -> None:
    if orig_bundle.get("type") != "omniverse_full" or recon_bundle.get("type") != "omniverse_full":
        raise ValueError("Both bundles must be full omniverse bundles")
    a = _b64_npz_to_arr(orig_bundle["grid_npz_b64"])  # (U, L, h, w)
    b = _b64_npz_to_arr(recon_bundle["grid_npz_b64"])  # (U, L, h, w)

    wa, ha, La, Ua = int(orig_bundle.get("width", a.shape[3])), int(orig_bundle.get("height", a.shape[2])), int(orig_bundle.get("layers", a.shape[1])), int(orig_bundle.get("universes", a.shape[0]))
    wb, hb, Lb, Ub = int(recon_bundle.get("width", b.shape[3])), int(recon_bundle.get("height", b.shape[2])), int(recon_bundle.get("layers", b.shape[1])), int(recon_bundle.get("universes", b.shape[0]))

    if (wa, ha, La, Ua) != (wb, hb, Lb, Ub):
        b = _omni_resize(b, wa, ha, La, Ua, method="interp")

    # build mosaics and stitch side-by-side
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p1 = td + "/orig.png"
        p2 = td + "/recon.png"
        save_mosaic_from_model({"type": "omniverse_full", "width": wa, "height": ha, "layers": La, "universes": Ua, "grid_npz_b64": _arr_to_b64_npz(a)}, p1, cmap=cmap)
        save_mosaic_from_model({"type": "omniverse_full", "width": wa, "height": ha, "layers": La, "universes": Ua, "grid_npz_b64": _arr_to_b64_npz(b)}, p2, cmap=cmap)
        img1 = Image.open(p1).convert("RGB")
        img2 = Image.open(p2).convert("RGB")
    h = max(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * h / img1.height), h), resample=Image.BILINEAR)
    img2 = img2.resize((int(img2.width * h / img2.height), h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (img1.width + img2.width, h), color="white")
    canvas.paste(img1, (0, 0))
    canvas.paste(img2, (img1.width, 0))
    canvas.save(output_path)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def _psnr(mse: float, max_i: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(max_i) - 10.0 * np.log10(mse))


def _radial_psd(a: np.ndarray) -> np.ndarray:
    f = np.fft.rfft2(a)
    psd2 = (f.real ** 2 + f.imag ** 2)
    h, w = a.shape
    ky = np.fft.fftfreq(h)
    kx = np.fft.rfftfreq(w)
    rr = np.sqrt((ky[:, None] ** 2) + (kx[None, :] ** 2))
    r = rr.flatten()
    p = psd2.flatten()
    bins = np.linspace(0, r.max() + 1e-9, num=64)
    idx = np.digitize(r, bins)
    out = np.zeros(len(bins) + 1, dtype=np.float64)
    counts = np.zeros(len(bins) + 1, dtype=np.int64)
    np.add.at(out, idx, p)
    np.add.at(counts, idx, 1)
    counts[counts == 0] = 1
    out = out / counts
    out = out[1:-1]
    s = np.sum(out)
    if s > 0:
        out = out / s
    return out.astype(np.float32)


def metrics_dataframe_from_grids(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    Ua, La, Ha, Wa = a.shape
    Ub, Lb, Hb, Wb = b.shape
    U = min(Ua, Ub)
    L = min(La, Lb)
    H = min(Ha, Hb)
    W = min(Wa, Wb)
    if U < 1 or L < 1 or H < 1 or W < 1:
        return pd.DataFrame([
            {"mse": np.nan, "rmse": np.nan, "psnr_db": np.nan, "spec_l1": np.nan, "spec_corr": np.nan, "width": 0, "height": 0, "layers": 0, "universes": 0}
        ])

    a = a[:U, :L, :H, :W].astype(np.float32)
    b = b[:U, :L, :H, :W].astype(np.float32)

    mses = []
    spec_l1s = []
    spec_corrs = []
    for u in range(U):
        for k in range(L):
            ak = a[u, k]
            bk = b[u, k]
            err = ak - bk
            mses.append(float(np.mean(err ** 2)))
            pa = _radial_psd(ak)
            pb = _radial_psd(bk)
            m = min(len(pa), len(pb))
            pa = pa[:m]
            pb = pb[:m]
            spec_l1s.append(float(np.mean(np.abs(pa - pb))))
            if np.std(pa) > 0 and np.std(pb) > 0:
                spec_corrs.append(float(np.corrcoef(pa, pb)[0, 1]))

    mse = float(np.mean(mses)) if mses else float("nan")
    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
    psnr = _psnr(mse, max_i=1.0) if np.isfinite(mse) else float("nan")
    spec_l1 = float(np.mean(spec_l1s)) if spec_l1s else float("nan")
    spec_corr = float(np.mean(spec_corrs)) if spec_corrs else float("nan")

    return pd.DataFrame([
        {
            "mse": mse,
            "rmse": rmse,
            "psnr_db": psnr,
            "spec_l1": spec_l1,
            "spec_corr": spec_corr,
            "width": int(W),
            "height": int(H),
            "layers": int(L),
            "universes": int(U),
        }
    ])


def metrics_from_paths(a_path: str, b_path: str) -> pd.DataFrame:
    a = load_model(a_path)
    b = load_model(b_path)
    if a.get("type") != "omniverse_full" or b.get("type") != "omniverse_full":
        raise ValueError("metrics_from_paths expects full omniverse bundles")
    sa = _b64_npz_to_arr(a["grid_npz_b64"])  # (U, L, h, w)
    sb = _b64_npz_to_arr(b["grid_npz_b64"])  # (U, L, h, w)
    return metrics_dataframe_from_grids(sa, sb)


# -----------------------------------------------------------------------------
# Model I/O
# -----------------------------------------------------------------------------


def save_model(bundle: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


def load_model(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
