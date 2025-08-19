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
class MultiverseConfig:
    strategy: str = "ratio"  # only 'ratio' supported
    spatial_ratio: int = 2    # keep every Nth pixel per axis
    layer_ratio: int = 1      # keep every Nth layer
    method: str = "interp"    # 'interp' (bilinear) or 'nearest'


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


def _interp_resize_layers(stack: np.ndarray, out_layers: int, method: str = "interp") -> np.ndarray:
    # stack: (L, H, W) -> (L2, H, W)
    L, H, W = stack.shape
    if L == out_layers:
        return stack.astype(np.float32, copy=False)
    if out_layers < 1:
        out_layers = 1
    if method == "nearest":
        idx = np.linspace(0, L - 1, out_layers)
        idx = np.rint(idx).astype(int)
        idx = np.clip(idx, 0, L - 1)
        return stack[idx, :, :].astype(np.float32)
    # linear interpolation along layer axis
    x_old = np.linspace(0.0, 1.0, L, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, out_layers, dtype=np.float32)
    out = np.empty((out_layers, H, W), dtype=np.float32)
    for i in range(H):
        # interpolate each row across layers for all columns at once
        s = stack[:, i, :]  # (L, W)
        # For numerical stability, do per-column linear interp
        for j in range(W):
            out[:, i, j] = np.interp(x_new, x_old, s[:, j])
    return out


def _stack_resize(stack: np.ndarray, out_w: int, out_h: int, out_layers: int, method: str = "interp") -> np.ndarray:
    L, H, W = stack.shape
    # resize spatial dims per layer then interpolate layers
    resized = np.empty((L, out_h, out_w), dtype=np.float32)
    for k in range(L):
        resized[k] = _interp_resize2d(stack[k], (out_w, out_h), method=method)
    return _interp_resize_layers(resized, out_layers, method=method)


# -----------------------------------------------------------------------------
# Generation (stack of fBm-like fields)
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


def generate_full_stack(width: int, height: int, layers: int, octaves: int = 4, seed: Optional[int] = None) -> Dict:
    if width < 1 or height < 1 or layers < 1:
        raise ValueError("width, height, layers must be >= 1")
    if octaves < 1:
        raise ValueError("octaves must be >= 1")
    rng = np.random.default_rng(seed)

    stack = np.empty((layers, height, width), dtype=np.float32)
    for L in range(layers):
        # vary seed via generator, generate per layer
        stack[L] = _generate_field(width, height, octaves, rng)
        # apply mild cross-layer nonlinearity to encourage diversity
        if L > 0:
            stack[L] = np.clip(0.7 * stack[L] + 0.3 * np.sin(3.0 * stack[L - 1]), 0.0, 1.0)

    bundle = {
        "version": 1,
        "type": "multiverse_full",
        "width": int(width),
        "height": int(height),
        "layers": int(layers),
        "field_dtype": "float32",
        "stack_npz_b64": _arr_to_b64_npz(stack),
        "params": {"octaves": int(octaves), "seed": (int(seed) if seed is not None else None)},
    }
    return bundle


# -----------------------------------------------------------------------------
# Compress / Expand
# -----------------------------------------------------------------------------


def compress_stack(full_bundle: Dict, config: Optional[MultiverseConfig] = None) -> Dict:
    if full_bundle.get("type") != "multiverse_full":
        raise ValueError("Not a full multiverse bundle")
    cfg = config or MultiverseConfig()
    if cfg.strategy.lower() != "ratio":
        raise ValueError("Only 'ratio' strategy is supported for multiverse")
    if cfg.spatial_ratio < 1 or cfg.layer_ratio < 1:
        raise ValueError("ratios must be >= 1")

    w = int(full_bundle.get("width", 0))
    h = int(full_bundle.get("height", 0))
    L = int(full_bundle.get("layers", 0))
    stack = _b64_npz_to_arr(full_bundle["stack_npz_b64"])  # (L, h, w)

    w_ds = max(1, w // cfg.spatial_ratio)
    h_ds = max(1, h // cfg.spatial_ratio)
    L_ds = max(1, L // cfg.layer_ratio)

    # spatial decimation via striding, then layer decimation
    ds_spatial = stack[:, ::cfg.spatial_ratio, ::cfg.spatial_ratio]
    if ds_spatial.shape[1:] != (h_ds, w_ds):
        # adjust using resize to exact ds size
        tmp = np.empty((L, h_ds, w_ds), dtype=np.float32)
        for k in range(L):
            tmp[k] = _interp_resize2d(stack[k], (w_ds, h_ds), method="nearest")
        ds_spatial = tmp

    idx_layers = np.arange(0, L, cfg.layer_ratio)
    if len(idx_layers) < L_ds:
        idx_layers = np.linspace(0, L - 1, L_ds)
        idx_layers = np.rint(idx_layers).astype(int)
    ds = ds_spatial[idx_layers, :, :]

    bundle = {
        "version": 1,
        "type": "multiverse_ratio",
        "strategy": "ratio",
        "spatial_ratio": int(cfg.spatial_ratio),
        "layer_ratio": int(cfg.layer_ratio),
        "orig_size": [int(w), int(h), int(L)],
        "ds_size": [int(w_ds), int(h_ds), int(L_ds)],
        "ds_dtype": "float32",
        "ds_npz_b64": _arr_to_b64_npz(ds),
        "config": {"method": cfg.method},
    }
    return bundle


essential_keys = ["multiverse_full", "multiverse_ratio"]


def expand_stack(bundle: Dict, target_size: Optional[Tuple[int, int, int]] = None, method: Optional[str] = None) -> Dict:
    btype = bundle.get("type")
    if btype == "multiverse_full":
        stack = _b64_npz_to_arr(bundle["stack_npz_b64"])  # (L, h, w)
        w = int(bundle.get("width", stack.shape[2]))
        h = int(bundle.get("height", stack.shape[1]))
        L = int(bundle.get("layers", stack.shape[0]))
        if target_size is not None:
            tw, th, tL = int(target_size[0]), int(target_size[1]), int(target_size[2])
            out = _stack_resize(stack, tw, th, tL, method=(method or "interp"))
            return {
                "version": 1,
                "type": "multiverse_full",
                "width": int(tw),
                "height": int(th),
                "layers": int(tL),
                "field_dtype": "float32",
                "stack_npz_b64": _arr_to_b64_npz(out),
                "params": bundle.get("params", {}),
            }
        else:
            return bundle

    if btype != "multiverse_ratio":
        raise ValueError("Not a multiverse ratio bundle")

    w, h, L = int(bundle["orig_size"][0]), int(bundle["orig_size"][1]), int(bundle["orig_size"][2])
    ds = _b64_npz_to_arr(bundle["ds_npz_b64"])  # (L_ds, h_ds, w_ds)
    out_w, out_h, out_L = (int(target_size[0]), int(target_size[1]), int(target_size[2])) if target_size else (w, h, L)
    meth = (method or bundle.get("config", {}).get("method", "interp")).lower()
    stack = _stack_resize(ds, out_w, out_h, out_L, method=meth)
    stack = np.clip(stack, 0.0, 1.0)

    full = {
        "version": 1,
        "type": "multiverse_full",
        "width": int(out_w),
        "height": int(out_h),
        "layers": int(out_L),
        "field_dtype": "float32",
        "stack_npz_b64": _arr_to_b64_npz(stack),
        "params": {"expanded_from": "ratio", "spatial_ratio": int(bundle.get("spatial_ratio", 1)), "layer_ratio": int(bundle.get("layer_ratio", 1)), **bundle.get("config", {})},
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
    if btype == "multiverse_full":
        stack = _b64_npz_to_arr(bundle["stack_npz_b64"])  # (L, h, w)
        L, h, w = stack.shape
    elif btype == "multiverse_ratio":
        w, h, L = int(bundle["orig_size"][0]), int(bundle["orig_size"][1]), int(bundle["orig_size"][2])
        ds = _b64_npz_to_arr(bundle["ds_npz_b64"])  # (L_ds, h_ds, w_ds)
        stack = _stack_resize(ds, w, h, L, method=bundle.get("config", {}).get("method", "interp"))
    else:
        raise ValueError("Unknown bundle type")

    # choose grid
    if cols is None or rows is None:
        # near-square grid
        cols = int(np.ceil(np.sqrt(stack.shape[0])))
        rows = int(np.ceil(stack.shape[0] / cols))

    # render each layer
    imgs = [
        _field_to_image(stack[k], cmap=cmap)
        for k in range(stack.shape[0])
    ]
    tile_w = max(img.width for img in imgs)
    tile_h = max(img.height for img in imgs)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color="white")

    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        if img.size != (tile_w, tile_h):
            img = img.resize((tile_w, tile_h), resample=Image.BILINEAR)
        canvas.paste(img, (c * tile_w, r * tile_h))

    canvas.save(output_path)


def save_compare_mosaic(orig_bundle: Dict, recon_bundle: Dict, output_path: str, cmap: str = "viridis") -> None:
    if orig_bundle.get("type") != "multiverse_full" or recon_bundle.get("type") != "multiverse_full":
        raise ValueError("Both bundles must be full multiverse bundles")
    a = _b64_npz_to_arr(orig_bundle["stack_npz_b64"])  # (L, h, w)
    b = _b64_npz_to_arr(recon_bundle["stack_npz_b64"])  # (L, h, w)

    wa, ha, La = int(orig_bundle.get("width", a.shape[2])), int(orig_bundle.get("height", a.shape[1])), int(orig_bundle.get("layers", a.shape[0]))
    wb, hb, Lb = int(recon_bundle.get("width", b.shape[2])), int(recon_bundle.get("height", b.shape[1])), int(recon_bundle.get("layers", b.shape[0]))

    if (wa, ha, La) != (wb, hb, Lb):
        b = _stack_resize(b, wa, ha, La, method="interp")

    # build mosaics and stitch side-by-side
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p1 = td + "/orig.png"
        p2 = td + "/recon.png"
        save_mosaic_from_model({"type": "multiverse_full", "width": wa, "height": ha, "layers": La, "stack_npz_b64": _arr_to_b64_npz(a)}, p1, cmap=cmap)
        save_mosaic_from_model({"type": "multiverse_full", "width": wa, "height": ha, "layers": La, "stack_npz_b64": _arr_to_b64_npz(b)}, p2, cmap=cmap)
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


def metrics_dataframe_from_stacks(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    La, Ha, Wa = a.shape
    Lb, Hb, Wb = b.shape
    L = min(La, Lb)
    H = min(Ha, Hb)
    W = min(Wa, Wb)
    if L < 1 or H < 1 or W < 1:
        return pd.DataFrame([
            {"mse": np.nan, "rmse": np.nan, "psnr_db": np.nan, "spec_l1": np.nan, "spec_corr": np.nan, "width": 0, "height": 0, "layers": 0}
        ])

    a = a[:L, :H, :W].astype(np.float32)
    b = b[:L, :H, :W].astype(np.float32)

    # per-layer metrics and aggregate
    mses = []
    spec_l1s = []
    spec_corrs = []
    for k in range(L):
        ak = a[k]
        bk = b[k]
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
        }
    ])


def metrics_from_paths(a_path: str, b_path: str) -> pd.DataFrame:
    a = load_model(a_path)
    b = load_model(b_path)
    if a.get("type") != "multiverse_full" or b.get("type") != "multiverse_full":
        raise ValueError("metrics_from_paths expects full multiverse bundles")
    sa = _b64_npz_to_arr(a["stack_npz_b64"])  # (L, h, w)
    sb = _b64_npz_to_arr(b["stack_npz_b64"])  # (L, h, w)
    return metrics_dataframe_from_stacks(sa, sb)


# -----------------------------------------------------------------------------
# Model I/O
# -----------------------------------------------------------------------------


def save_model(bundle: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


def load_model(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
