from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class ImageConfig:
    strategy: str = "ratio"  # only 'ratio' supported for images (education/demo)
    ratio: int = 2            # keep every Nth pixel in each dimension
    method: str = "interp"    # 'interp' (bilinear) or 'nearest'


# ---- Model I/O helpers ----
def _img_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_png_to_img(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str.encode("ascii"))
    return Image.open(io.BytesIO(data)).convert("RGBA")


def save_model(bundle: Dict, path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


def load_model(path: str) -> Dict:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---- Core compress/expand ----
def compress_image(img: Image.Image, config: Optional[ImageConfig] = None) -> Dict:
    cfg = config or ImageConfig()
    if cfg.strategy.lower() != "ratio":
        raise ValueError("Only 'ratio' strategy is supported for images")
    if cfg.ratio < 1:
        raise ValueError("ratio must be >= 1")

    orig_mode = img.mode
    w, h = img.size
    # downsample by integer ratio using BOX filter (area resampling)
    w_ds = max(1, w // cfg.ratio)
    h_ds = max(1, h // cfg.ratio)
    ds = img.resize((w_ds, h_ds), resample=Image.BOX)

    bundle = {
        "version": 1,
        "type": "phi-image-model",
        "strategy": "ratio",
        "ratio": int(cfg.ratio),
        "mode": orig_mode,
        "orig_size": [int(w), int(h)],
        "ds_size": [int(w_ds), int(h_ds)],
        "ds_png_b64": _img_to_b64_png(ds),
        "config": {"method": cfg.method},
    }
    return bundle


def expand_image(bundle: Dict, target_size: Optional[Tuple[int, int]] = None, method: Optional[str] = None) -> Image.Image:
    if bundle.get("type") != "phi-image-model":
        raise ValueError("Not an image model bundle")
    ds_img = _b64_png_to_img(bundle["ds_png_b64"])  # RGBA
    orig_mode = bundle.get("mode", "RGB")
    ow, oh = bundle.get("orig_size", [ds_img.width, ds_img.height])

    # Determine output size
    if target_size is None:
        out_w, out_h = int(ow), int(oh)
    else:
        out_w, out_h = int(target_size[0]), int(target_size[1])

    meth = (method or bundle.get("config", {}).get("method", "interp")).lower()
    if meth == "interp":
        resample = Image.BILINEAR
    elif meth == "nearest":
        resample = Image.NEAREST
    else:
        raise ValueError("Unknown method: %s" % meth)

    up = ds_img.resize((out_w, out_h), resample=resample)
    return up.convert(orig_mode)


# ---- Compare + Metrics ----
def save_compare_side_by_side(orig: Image.Image, recon: Image.Image, output_path: str) -> None:
    # Put images side by side (align heights)
    h = max(orig.height, recon.height)
    scale_o = h / orig.height
    scale_r = h / recon.height
    o_w = max(1, int(orig.width * scale_o))
    r_w = max(1, int(recon.width * scale_r))
    o_resized = orig.resize((o_w, h), resample=Image.BILINEAR)
    r_resized = recon.resize((r_w, h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (o_resized.width + r_resized.width, h), color="white")
    canvas.paste(o_resized.convert("RGB"), (0, 0))
    canvas.paste(r_resized.convert("RGB"), (o_resized.width, 0))
    canvas.save(output_path)


def metrics_dataframe(orig: Image.Image, recon: Image.Image) -> pd.DataFrame:
    # Convert to common size and RGB
    w = min(orig.width, recon.width)
    h = min(orig.height, recon.height)
    if w < 1 or h < 1:
        return pd.DataFrame([{"mse": np.nan, "rmse": np.nan, "psnr_db": np.nan, "width": 0, "height": 0}])
    o = orig.resize((w, h), resample=Image.BILINEAR).convert("RGB")
    r = recon.resize((w, h), resample=Image.BILINEAR).convert("RGB")
    a = np.asarray(o, dtype=np.float32)
    b = np.asarray(r, dtype=np.float32)
    err = a - b
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    MAX_I = 255.0
    psnr = float(20.0 * np.log10(MAX_I) - 10.0 * np.log10(mse)) if mse > 0 else float("inf")
    return pd.DataFrame([
        {"mse": mse, "rmse": rmse, "psnr_db": psnr, "width": int(w), "height": int(h)}
    ])
