from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import imageio.v2 as imageio


@dataclass
class VideoConfig:
    strategy: str = "ratio"          # only 'ratio' supported (educational)
    spatial_ratio: int = 2            # keep every Nth pixel per axis
    temporal_ratio: int = 1           # keep every Nth frame (1 = keep all)
    method: str = "interp"            # 'interp' (bilinear spatial + linear temporal) or 'nearest'/'hold'
    frame_limit: Optional[int] = None # optional safety cap for frames during compression


# --------- Bundle I/O ---------

def save_model(bundle: Dict, path: str) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


def load_model(path: str) -> Dict:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------- Helpers: PNG frame <-> base64 ---------

def _frame_to_b64_png(frame: np.ndarray) -> str:
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_png_to_frame(b64_str: str) -> np.ndarray:
    data = base64.b64decode(b64_str.encode("ascii"))
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


# --------- Core compress/expand ---------

def compress_video(input_path: str, config: Optional[VideoConfig] = None) -> Dict:
    cfg = config or VideoConfig()
    if cfg.strategy.lower() != "ratio":
        raise ValueError("Only 'ratio' strategy is supported for video")
    if cfg.spatial_ratio < 1 or cfg.temporal_ratio < 1:
        raise ValueError("spatial_ratio and temporal_ratio must be >= 1")

    reader = imageio.get_reader(input_path)
    meta = reader.get_meta_data() or {}
    fps = float(meta.get("fps", 30.0))

    frames_b64: List[str] = []
    orig_frames = 0
    first_frame = None

    try:
        for idx, frame in enumerate(reader):
            orig_frames += 1
            if idx % cfg.temporal_ratio != 0:
                continue
            if first_frame is None:
                first_frame = frame
            # spatial downsample
            img = Image.fromarray(frame)
            w, h = img.size
            w_ds = max(1, w // cfg.spatial_ratio)
            h_ds = max(1, h // cfg.spatial_ratio)
            ds = img.resize((w_ds, h_ds), resample=Image.BOX)
            frames_b64.append(_frame_to_b64_png(np.asarray(ds)))
            # optional safety cap
            if cfg.frame_limit is not None and len(frames_b64) >= cfg.frame_limit:
                break
    finally:
        reader.close()

    if first_frame is None:
        raise ValueError("No frames read from video")

    o_w, o_h = Image.fromarray(first_frame).size
    ds0 = _b64_png_to_frame(frames_b64[0])
    d_w, d_h = Image.fromarray(ds0).size

    bundle = {
        "version": 1,
        "type": "phi-video-model",
        "strategy": "ratio",
        "spatial_ratio": int(cfg.spatial_ratio),
        "temporal_ratio": int(cfg.temporal_ratio),
        "method": cfg.method,
        "fps": fps,
        "orig_size": [int(o_w), int(o_h)],
        "ds_size": [int(d_w), int(d_h)],
        "orig_frames": int(orig_frames),
        "ds_frames": int(len(frames_b64)),
        "frames_png_b64": frames_b64,
    }
    return bundle


def expand_video(
    bundle: Dict,
    output_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    fps: Optional[float] = None,
    method: Optional[str] = None,
    target_frames: Optional[int] = None,
) -> None:
    if bundle.get("type") != "phi-video-model":
        raise ValueError("Not a video model bundle")

    # settings
    spatial_method = (method or bundle.get("method", "interp")).lower()
    temporal_method = spatial_method  # keep same keyword for simplicity
    fps_out = float(fps) if fps is not None else float(bundle.get("fps", 30.0))

    # decode downsampled frames and optionally upsample spatially
    ds_frames = [_b64_png_to_frame(b) for b in bundle.get("frames_png_b64", [])]
    if not ds_frames:
        raise ValueError("Model bundle contains no frames")

    if target_size is None:
        out_w, out_h = bundle.get("orig_size", [ds_frames[0].shape[1], ds_frames[0].shape[0]])
        out_w, out_h = int(out_w), int(out_h)
    else:
        out_w, out_h = int(target_size[0]), int(target_size[1])

    if spatial_method == "interp":
        resample = Image.BILINEAR
    elif spatial_method in ("nearest", "hold"):
        resample = Image.NEAREST
    else:
        raise ValueError(f"Unknown method: {spatial_method}")

    # pre-upsample spatially to target size
    up_frames = [
        np.asarray(Image.fromarray(fr).resize((out_w, out_h), resample=resample), dtype=np.uint8)
        for fr in ds_frames
    ]

    ds_count = len(up_frames)
    if target_frames is None:
        t_ratio = int(bundle.get("temporal_ratio", 1))
        out_count = int(bundle.get("orig_frames", ds_count * t_ratio))
    else:
        out_count = int(target_frames)

    writer = imageio.get_writer(output_path, fps=fps_out)
    try:
        if ds_count == 1:
            # single keyframe: just repeat
            for _ in range(out_count):
                writer.append_data(up_frames[0])
            return

        if temporal_method == "interp":
            # linear blend between neighboring frames
            # compute effective temporal ratio to span ds_count -> out_count
            # map t in [0, out_count-1] to s in [0, ds_count-1]
            denom = max(out_count - 1, 1)
            for t in range(out_count):
                s = (t / denom) * (ds_count - 1)
                i0 = int(np.floor(s))
                i1 = min(i0 + 1, ds_count - 1)
                a = float(s - i0)
                f0 = up_frames[i0].astype(np.float32)
                f1 = up_frames[i1].astype(np.float32)
                blend = np.clip((1.0 - a) * f0 + a * f1, 0, 255).astype(np.uint8)
                writer.append_data(blend)
        else:  # 'nearest'/'hold'
            # repeat each ds frame proportionally
            # default: repeat each by k where sum ~= out_count
            repeats = [out_count // ds_count] * ds_count
            for i in range(out_count % ds_count):
                repeats[i] += 1
            for fr, r in zip(up_frames, repeats):
                for _ in range(r):
                    writer.append_data(fr)
    finally:
        writer.close()


# --------- Metrics + Compare ---------

def metrics_from_paths(orig_path: str, recon_path: str, sample_frames: int = 60) -> "pd.DataFrame":
    """Compute average MSE/RMSE/PSNR over up to sample_frames evenly spaced frames."""
    import pandas as pd  # local import to avoid hard dependency when metrics not used
    r0 = imageio.get_reader(orig_path)
    r1 = imageio.get_reader(recon_path)
    try:
        n0 = r0.count_frames()
        n1 = r1.count_frames()
    except Exception:
        # some readers cannot count frames; fall back to metadata duration*fps approx
        n0 = int((r0.get_meta_data() or {}).get("nframes") or 0)
        n1 = int((r1.get_meta_data() or {}).get("nframes") or 0)
    if not n0 or not n1:
        # fallback: iterate and count minimal
        n0 = n1 = 0
        for _ in r0:
            n0 += 1
        for _ in r1:
            n1 += 1
        # reopen to read again
        r0.close(); r1.close()
        r0 = imageio.get_reader(orig_path)
        r1 = imageio.get_reader(recon_path)

    n = max(1, min(n0, n1))
    if n <= 0:
        return pd.DataFrame([{"mse": np.nan, "rmse": np.nan, "psnr_db": np.nan, "frames": 0}])

    # Choose up to sample_frames evenly spaced indices
    k = min(sample_frames, n)
    idxs = np.linspace(0, n - 1, num=k).astype(int)

    # Close initial readers; we'll reopen per-frame using paths for portability
    try:
        r0.close(); r1.close()
    except Exception:
        pass

    def frame_at(path: str, index: int) -> np.ndarray:
        rr = imageio.get_reader(path)
        try:
            # Try random access if supported by the backend
            try:
                return rr.get_data(index)
            except Exception:
                # Fallback: iterate to the desired index
                for i, fr in enumerate(rr):
                    if i == index:
                        return fr
        finally:
            rr.close()
        raise IndexError("frame index out of range")

    mse_acc = 0.0
    for i in idxs:
        a = frame_at(orig_path, int(i))
        b = frame_at(recon_path, int(i))
        # match size by downscale to smaller for fair compare
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        if h < 1 or w < 1:
            continue
        a_r = np.asarray(Image.fromarray(a).resize((w, h), resample=Image.BILINEAR), dtype=np.float32)
        b_r = np.asarray(Image.fromarray(b).resize((w, h), resample=Image.BILINEAR), dtype=np.float32)
        err = a_r - b_r
        mse_acc += float(np.mean(err ** 2))

    mse = mse_acc / float(k)
    rmse = float(np.sqrt(mse))
    MAX_I = 255.0
    psnr = float(20.0 * np.log10(MAX_I) - 10.0 * np.log10(mse)) if mse > 0 else float("inf")
    return pd.DataFrame([
        {"mse": mse, "rmse": rmse, "psnr_db": psnr, "frames": int(k)}
    ])


def save_compare_first_frame(orig_path: str, recon_path: str, output_image: str) -> None:
    r0 = imageio.get_reader(orig_path)
    r1 = imageio.get_reader(recon_path)
    try:
        f0 = next(iter(r0))
        f1 = next(iter(r1))
    finally:
        r0.close(); r1.close()
    o = Image.fromarray(f0).convert("RGB")
    r = Image.fromarray(f1).convert("RGB")
    h = max(o.height, r.height)
    def resize_to_h(im: Image.Image, h: int) -> Image.Image:
        scale = h / im.height
        return im.resize((max(1, int(im.width * scale)), h), resample=Image.BILINEAR)
    o2, r2 = resize_to_h(o, h), resize_to_h(r, h)
    canvas = Image.new("RGB", (o2.width + r2.width, h), color="white")
    canvas.paste(o2, (0, 0)); canvas.paste(r2, (o2.width, 0))
    canvas.save(output_image)
