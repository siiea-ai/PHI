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
class CosmosConfig:
    strategy: str = "ratio"  # only 'ratio' supported for educational demo
    ratio: int = 2            # keep every Nth pixel per axis
    method: str = "interp"    # 'interp' (bilinear) or 'nearest'


# -----------------------------------------------------------------------------
# Helpers: base64 encode/decode numpy arrays as NPZ
# -----------------------------------------------------------------------------


def _arr_to_b64_npz(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    # store as float32 for compactness and consistency
    np.savez_compressed(buf, arr=arr.astype(np.float32, copy=False))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_npz_to_arr(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("ascii"))
    buf = io.BytesIO(raw)
    with np.load(buf) as data:
        arr = data["arr"]
    return np.asarray(arr, dtype=np.float32)


# -----------------------------------------------------------------------------
# Model I/O
# -----------------------------------------------------------------------------


def save_model(bundle: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


def load_model(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Generation (fBm via multi-octave noise + interpolation)
# -----------------------------------------------------------------------------


def _interp_resize(arr: np.ndarray, out_wh: Tuple[int, int], method: str = "interp") -> np.ndarray:
    h, w = arr.shape
    out_w, out_h = int(out_wh[0]), int(out_wh[1])
    if (w, h) == (out_w, out_h):
        return arr.astype(np.float32, copy=False)
    # Prefer SciPy for robust interpolation; fallback to nearest using repeat
    try:
        from scipy.ndimage import zoom  # type: ignore

        zx = out_h / float(h)
        zy = out_w / float(w)
        order = 1 if method == "interp" else 0
        out = zoom(arr, (zx, zy), order=order, prefilter=False)
        # Ensure exact shape (zoom can be off by 1)
        out = np.asarray(out, dtype=np.float32)
        if out.shape != (out_h, out_w):
            out = out[:out_h, :out_w]
            if out.shape[0] < out_h or out.shape[1] < out_w:
                pad_h = max(0, out_h - out.shape[0])
                pad_w = max(0, out_w - out.shape[1])
                out = np.pad(out, ((0, pad_h), (0, pad_w)), mode="edge")
        return out
    except Exception:
        # Nearest-only fallback
        sx = max(1, int(round(out_w / w)))
        sy = max(1, int(round(out_h / h)))
        out = np.repeat(np.repeat(arr, sy, axis=0), sx, axis=1)
        return out[:out_h, :out_w].astype(np.float32)


def generate_full_field(width: int, height: int, octaves: int = 4, seed: Optional[int] = None) -> Dict:
    if width < 1 or height < 1:
        raise ValueError("width and height must be >= 1")
    if octaves < 1:
        raise ValueError("octaves must be >= 1")
    rng = np.random.default_rng(seed)

    # fBm: sum of low-res noise upsampled to target, doubling frequency each octave
    field = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0
    for o in range(octaves):
        ow = max(1, width // (2 ** o))
        oh = max(1, height // (2 ** o))
        base = rng.standard_normal((oh, ow), dtype=np.float32)
        up = _interp_resize(base, (width, height), method="interp")
        field += amp * up
        amp_sum += amp
        amp *= 0.5  # reduce amplitude for higher frequencies

    field /= max(amp_sum, 1e-6)
    # Normalize to [0, 1]
    fmin, fmax = float(field.min()), float(field.max())
    if fmax > fmin:
        field = (field - fmin) / (fmax - fmin)
    else:
        field.fill(0.5)

    bundle = {
        "version": 1,
        "type": "cosmos_field_full",
        "width": int(width),
        "height": int(height),
        "field_dtype": "float32",
        "field_npz_b64": _arr_to_b64_npz(field),
        "params": {"octaves": int(octaves), "seed": (int(seed) if seed is not None else None)},
    }
    return bundle


# -----------------------------------------------------------------------------
# Compress / Expand
# -----------------------------------------------------------------------------


def compress_field(full_bundle: Dict, config: Optional[CosmosConfig] = None) -> Dict:
    if full_bundle.get("type") != "cosmos_field_full":
        raise ValueError("Not a full cosmos field bundle")
    cfg = config or CosmosConfig()
    if cfg.strategy.lower() != "ratio":
        raise ValueError("Only 'ratio' strategy is supported for cosmos")
    if cfg.ratio < 1:
        raise ValueError("ratio must be >= 1")

    w = int(full_bundle.get("width", 0))
    h = int(full_bundle.get("height", 0))
    field = _b64_npz_to_arr(full_bundle["field_npz_b64"])  # (h, w)

    w_ds = max(1, w // cfg.ratio)
    h_ds = max(1, h // cfg.ratio)
    ds = field[::cfg.ratio, ::cfg.ratio]
    # If stride skipped to very small, adjust via resize to exact ds size
    if ds.shape != (h_ds, w_ds):
        ds = _interp_resize(ds, (w_ds, h_ds), method="nearest")

    bundle = {
        "version": 1,
        "type": "cosmos_field_ratio",
        "strategy": "ratio",
        "ratio": int(cfg.ratio),
        "orig_size": [int(w), int(h)],
        "ds_size": [int(w_ds), int(h_ds)],
        "ds_dtype": "float32",
        "ds_npz_b64": _arr_to_b64_npz(ds),
        "config": {"method": cfg.method},
    }
    return bundle


def expand_field(bundle: Dict, target_size: Optional[Tuple[int, int]] = None, method: Optional[str] = None) -> Dict:
    btype = bundle.get("type")
    if btype == "cosmos_field_full":
        # Already full; optionally resize to target
        field = _b64_npz_to_arr(bundle["field_npz_b64"])  # (h, w)
        w = int(bundle.get("width", field.shape[1]))
        h = int(bundle.get("height", field.shape[0]))
        if target_size is not None:
            tw, th = int(target_size[0]), int(target_size[1])
            out = _interp_resize(field, (tw, th), method=(method or "interp"))
            return {
                "version": 1,
                "type": "cosmos_field_full",
                "width": int(tw),
                "height": int(th),
                "field_dtype": "float32",
                "field_npz_b64": _arr_to_b64_npz(out),
                "params": bundle.get("params", {}),
            }
        else:
            return bundle

    if btype != "cosmos_field_ratio":
        raise ValueError("Not a cosmos ratio bundle")

    w, h = int(bundle["orig_size"][0]), int(bundle["orig_size"][1])
    ds = _b64_npz_to_arr(bundle["ds_npz_b64"])  # (h_ds, w_ds)
    out_w, out_h = (int(target_size[0]), int(target_size[1])) if target_size else (w, h)
    meth = (method or bundle.get("config", {}).get("method", "interp")).lower()
    field = _interp_resize(ds, (out_w, out_h), method=meth)
    # Clip to [0,1]
    field = np.clip(field, 0.0, 1.0)

    full = {
        "version": 1,
        "type": "cosmos_field_full",
        "width": int(out_w),
        "height": int(out_h),
        "field_dtype": "float32",
        "field_npz_b64": _arr_to_b64_npz(field),
        "params": {"expanded_from": "ratio", "ratio": int(bundle.get("ratio", 1)), **bundle.get("config", {})},
    }
    return full


# -----------------------------------------------------------------------------
# Visualization & Compare
# -----------------------------------------------------------------------------


def _field_to_image(field: np.ndarray, cmap: str = "viridis") -> Image.Image:
    # Normalize to [0,1]
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


def save_image_from_model(bundle: Dict, output_path: str, cmap: str = "viridis") -> None:
    btype = bundle.get("type")
    if btype == "cosmos_field_full":
        field = _b64_npz_to_arr(bundle["field_npz_b64"])  # (h, w)
    elif btype == "cosmos_field_ratio":
        w, h = int(bundle["orig_size"][0]), int(bundle["orig_size"][1])
        field = _b64_npz_to_arr(bundle["ds_npz_b64"])  # visualize DS upsampled to orig size
        field = _interp_resize(field, (w, h), method=bundle.get("config", {}).get("method", "interp"))
    else:
        raise ValueError("Unknown bundle type")
    img = _field_to_image(field, cmap=cmap)
    img.save(output_path)


def save_compare_image(orig_bundle: Dict, recon_bundle: Dict, output_path: str, cmap: str = "viridis") -> None:
    if orig_bundle.get("type") != "cosmos_field_full":
        raise ValueError("orig_bundle must be a full cosmos field bundle")
    a = _b64_npz_to_arr(orig_bundle["field_npz_b64"])  # (h, w)

    if recon_bundle.get("type") != "cosmos_field_full":
        raise ValueError("recon_bundle must be a full cosmos field bundle")
    b = _b64_npz_to_arr(recon_bundle["field_npz_b64"])  # (h, w)

    aw, ah = int(orig_bundle.get("width", a.shape[1])), int(orig_bundle.get("height", a.shape[0]))
    bw, bh = int(recon_bundle.get("width", b.shape[1])), int(recon_bundle.get("height", b.shape[0]))

    if (aw, ah) != (bw, bh):
        b = _interp_resize(b, (aw, ah), method="interp")

    img_a = _field_to_image(a, cmap=cmap)
    img_b = _field_to_image(b, cmap=cmap)

    # Side-by-side
    h = max(img_a.height, img_b.height)
    scale_a = h / img_a.height
    scale_b = h / img_b.height
    a_w = max(1, int(img_a.width * scale_a))
    b_w = max(1, int(img_b.width * scale_b))
    a_resized = img_a.resize((a_w, h), resample=Image.BILINEAR)
    b_resized = img_b.resize((b_w, h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (a_resized.width + b_resized.width, h), color="white")
    canvas.paste(a_resized, (0, 0))
    canvas.paste(b_resized, (a_resized.width, 0))
    canvas.save(output_path)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def _psnr(mse: float, max_i: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return float(20.0 * np.log10(max_i) - 10.0 * np.log10(mse))


def _radial_psd(a: np.ndarray) -> np.ndarray:
    # Compute radially-averaged power spectral density
    f = np.fft.rfft2(a)
    psd2 = (f.real ** 2 + f.imag ** 2)
    h, w = a.shape
    # Build radial distances on the half-spectrum grid
    ky = np.fft.fftfreq(h)
    kx = np.fft.rfftfreq(w)
    # Broadcasting to form radius grid
    rr = np.sqrt((ky[:, None] ** 2) + (kx[None, :] ** 2))
    r = rr.flatten()
    p = psd2.flatten()
    # Bin by radius
    bins = np.linspace(0, r.max() + 1e-9, num=64)
    idx = np.digitize(r, bins)
    out = np.zeros(len(bins) + 1, dtype=np.float64)
    counts = np.zeros(len(bins) + 1, dtype=np.int64)
    np.add.at(out, idx, p)
    np.add.at(counts, idx, 1)
    counts[counts == 0] = 1
    out = out / counts
    out = out[1:-1]
    # Normalize
    s = np.sum(out)
    if s > 0:
        out = out / s
    return out.astype(np.float32)


def metrics_dataframe_from_fields(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h < 1 or w < 1:
        return pd.DataFrame([{"mse": np.nan, "rmse": np.nan, "psnr_db": np.nan, "spec_l1": np.nan, "spec_corr": np.nan, "width": 0, "height": 0}])
    a = a[:h, :w].astype(np.float32)
    b = b[:h, :w].astype(np.float32)
    err = a - b
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    psnr = _psnr(mse, max_i=1.0)

    pa = _radial_psd(a)
    pb = _radial_psd(b)
    # Align lengths
    m = min(pa.shape[0], pb.shape[0])
    pa = pa[:m]
    pb = pb[:m]
    spec_l1 = float(np.mean(np.abs(pa - pb)))
    if np.std(pa) > 0 and np.std(pb) > 0:
        spec_corr = float(np.corrcoef(pa, pb)[0, 1])
    else:
        spec_corr = float("nan")

    return pd.DataFrame([
        {
            "mse": mse,
            "rmse": rmse,
            "psnr_db": psnr,
            "spec_l1": spec_l1,
            "spec_corr": spec_corr,
            "width": int(w),
            "height": int(h),
        }
    ])


def metrics_from_paths(a_path: str, b_path: str) -> pd.DataFrame:
    a = load_model(a_path)
    b = load_model(b_path)
    if a.get("type") != "cosmos_field_full" or b.get("type") != "cosmos_field_full":
        raise ValueError("metrics_from_paths expects full cosmos field bundles")
    af = _b64_npz_to_arr(a["field_npz_b64"])  # (h, w)
    bf = _b64_npz_to_arr(b["field_npz_b64"])  # (h, w)
    return metrics_dataframe_from_fields(af, bf)
