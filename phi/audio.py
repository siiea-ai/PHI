from __future__ import annotations

import base64
import io
import json
import wave
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


@dataclass
class AudioConfig:
    strategy: str = "ratio"  # only 'ratio' supported (educational)
    ratio: int = 2            # keep every Nth sample
    method: str = "interp"    # 'interp' (linear) or 'hold'


# --------- Bundle I/O ---------

def save_model(bundle: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


def load_model(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------- WAV I/O (16-bit PCM) ---------

def load_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        if sampwidth != 2:
            raise ValueError("Only 16-bit PCM WAV is supported")
        raw = wf.readframes(nframes)
    arr = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels)
    else:
        arr = arr.reshape(-1, 1)
    return arr, framerate


def save_wav(path: str, data: np.ndarray, sample_rate: int) -> None:
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    data16 = np.asarray(np.clip(np.round(data), -32768, 32767), dtype=np.int16)
    n_channels = data16.shape[1]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data16.tobytes())


# --------- General audio loader (WAV/MP3/etc.) ---------
def load_audio(path: str) -> tuple[np.ndarray, int]:
    """
    Load audio from common formats.
    - If 16-bit PCM WAV, read via wave for zero-deps path.
    - Otherwise, fall back to pydub (requires ffmpeg), converting to 16-bit PCM.
    Returns (data[int16] of shape [frames, channels], sample_rate).
    """
    p = str(path).lower()
    # Try native WAV fast-path
    if p.endswith(".wav"):
        try:
            return load_wav(path)
        except Exception:
            # fall back to pydub if non-PCM16 or unsupported WAV subtype
            pass
    try:
        from pydub import AudioSegment  # lazy import to avoid hard dep
    except Exception as e:
        raise RuntimeError(
            "Reading non-16-bit WAV requires pydub and ffmpeg. "
            "Install with 'pip install pydub' and ensure ffmpeg is available."
        ) from e
    seg = AudioSegment.from_file(path)
    seg = seg.set_sample_width(2)  # 16-bit PCM
    sr = int(seg.frame_rate)
    ch = int(seg.channels)
    arr = np.array(seg.get_array_of_samples())
    if ch > 1:
        arr = arr.reshape(-1, ch)
    else:
        arr = arr.reshape(-1, 1)
    return arr.astype(np.int16), sr


# --------- Core compress/expand ---------

def compress_audio(data: np.ndarray, sample_rate: int, config: Optional[AudioConfig] = None) -> Dict:
    cfg = config or AudioConfig()
    if cfg.strategy.lower() != "ratio":
        raise ValueError("Only 'ratio' strategy is supported for audio")
    if cfg.ratio < 1:
        raise ValueError("ratio must be >= 1")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    ds = data[:: cfg.ratio]
    bundle = {
        "version": 1,
        "type": "phi-audio-model",
        "strategy": "ratio",
        "ratio": int(cfg.ratio),
        "method": cfg.method,
        "sample_rate": int(sample_rate),
        "channels": int(ds.shape[1]),
        "orig_frames": int(data.shape[0]),
        "ds_frames": int(ds.shape[0]),
        "pcm16_b64": base64.b64encode(ds.astype(np.int16).tobytes()).decode("ascii"),
    }
    return bundle


def expand_audio(bundle: Dict, target_frames: Optional[int] = None, method: Optional[str] = None) -> np.ndarray:
    if bundle.get("type") != "phi-audio-model":
        raise ValueError("Not an audio model bundle")
    ratio = int(bundle.get("ratio", 2))
    ch = int(bundle.get("channels", 1))
    orig_frames = int(bundle.get("orig_frames", 0))
    ds_frames = int(bundle.get("ds_frames", 0))
    meth = (method or bundle.get("method", "interp")).lower()

    ds_bytes = base64.b64decode(bundle["pcm16_b64"].encode("ascii"))
    ds = np.frombuffer(ds_bytes, dtype=np.int16)
    if ch > 1:
        ds = ds.reshape(-1, ch)
    else:
        ds = ds.reshape(-1, 1)

    out_len = int(target_frames) if target_frames is not None else max(orig_frames, ds.shape[0] * ratio)

    if meth == "hold":
        # Repeat each sample 'ratio' times, then trim/pad to out_len
        up = np.repeat(ds, repeats=ratio, axis=0)
        if up.shape[0] >= out_len:
            up = up[:out_len]
        else:
            pad = np.tile(ds[-1:], (out_len - up.shape[0], 1))
            up = np.vstack([up, pad])
    elif meth == "interp":
        # Linear interpolation up to out_len
        x_src = np.linspace(0, out_len - 1, num=ds.shape[0], dtype=np.float32)
        x_tgt = np.arange(out_len, dtype=np.float32)
        up = np.empty((out_len, ch), dtype=np.float32)
        for c in range(ch):
            up[:, c] = np.interp(x_tgt, x_src, ds[:, c].astype(np.float32))
    else:
        raise ValueError(f"Unknown method: {meth}")

    return np.asarray(np.round(up), dtype=np.int16)


# --------- Metrics + Compare ---------

def metrics_dataframe(orig: np.ndarray, recon: np.ndarray, sample_rate: int) -> pd.DataFrame:
    if orig.ndim == 1:
        orig = orig.reshape(-1, 1)
    if recon.ndim == 1:
        recon = recon.reshape(-1, 1)
    n = min(orig.shape[0], recon.shape[0])
    ch = orig.shape[1]
    if n == 0:
        return pd.DataFrame([{
            "mse": np.nan,
            "rmse": np.nan,
            "psnr_db": np.nan,
            "snr_db": np.nan,
            "stft_mse": np.nan,
            "frames": 0,
            "sample_rate": int(sample_rate),
            "channels": int(ch),
        }])
    a = orig[:n].astype(np.float32)
    b = recon[:n].astype(np.float32)
    err = a - b
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    MAX_A = 32767.0
    psnr = float(20.0 * np.log10(MAX_A) - 10.0 * np.log10(mse)) if mse > 0 else float("inf")
    # SNR (signal-to-noise ratio)
    sig_power = float(np.mean(a ** 2))
    noise_power = float(np.mean(err ** 2))
    snr = float(10.0 * np.log10(sig_power / noise_power)) if noise_power > 0 else float("inf")
    # STFT-based error on first channel using simple log-magnitude spectrogram
    stft_mse = _spectrogram_mse(a[:, 0], b[:, 0], sample_rate)
    return pd.DataFrame([
        {
            "mse": mse,
            "rmse": rmse,
            "psnr_db": psnr,
            "snr_db": snr,
            "stft_mse": stft_mse,
            "frames": int(n),
            "sample_rate": int(sample_rate),
            "channels": int(ch),
        }
    ])


def save_compare_plot(orig: np.ndarray, recon: np.ndarray, output_path: str, width: int = 1000, height: int = 300) -> None:
    # Simple waveform overlay using PIL
    if orig.ndim == 1:
        orig = orig.reshape(-1, 1)
    if recon.ndim == 1:
        recon = recon.reshape(-1, 1)
    n = min(orig.shape[0], recon.shape[0])
    if n < 2:
        img = Image.new("RGB", (width, height), "white")
        img.save(output_path)
        return

    # Use first channel for plot
    a = orig[:n, 0].astype(np.float32) / 32768.0
    b = recon[:n, 0].astype(np.float32) / 32768.0

    # Sample to width points
    idx = np.linspace(0, n - 1, num=width).astype(int)
    aa = a[idx]
    bb = b[idx]

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    mid = height // 2
    scale = 0.45 * height
    # axes
    draw.line([(0, mid), (width, mid)], fill=(220, 220, 220))

    def poly(points, color):
        for i in range(1, len(points)):
            draw.line([points[i - 1], points[i]], fill=color)

    pts_a = [(x, int(mid - aa[x] * scale)) for x in range(width)]
    pts_b = [(x, int(mid - bb[x] * scale)) for x in range(width)]
    poly(pts_a, (0, 92, 230))   # blue
    poly(pts_b, (230, 0, 0))    # red

    img.save(output_path)


def _stft_logmag(x: np.ndarray, n_fft: int = 1024, hop: int = 512) -> np.ndarray:
    """Compute a simple log-magnitude spectrogram for a mono signal."""
    if x.ndim > 1:
        x = x[:, 0]
    x = x.astype(np.float32)
    if x.shape[0] < n_fft:
        x = np.pad(x, (0, n_fft - x.shape[0]))
    w = np.hanning(n_fft).astype(np.float32)
    if x.shape[0] < n_fft:
        frames = 1
    else:
        frames = 1 + (x.shape[0] - n_fft) // hop
    mags = []
    for i in range(frames):
        start = i * hop
        frame = x[start:start + n_fft]
        if frame.shape[0] < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.shape[0]))
        spec = np.fft.rfft(frame * w)
        mags.append(np.log1p(np.abs(spec)))
    S = np.stack(mags, axis=1)  # [freq, time]
    return S


def _spectrogram_mse(a: np.ndarray, b: np.ndarray, sample_rate: int, n_fft: int = 1024, hop: int = 512) -> float:
    Sa = _stft_logmag(a, n_fft=n_fft, hop=hop)
    Sb = _stft_logmag(b, n_fft=n_fft, hop=hop)
    f = min(Sa.shape[0], Sb.shape[0])
    t = min(Sa.shape[1], Sb.shape[1])
    if f == 0 or t == 0:
        return float("nan")
    Da = Sa[:f, :t]
    Db = Sb[:f, :t]
    return float(np.mean((Da - Db) ** 2))


def save_spectrogram_compare(orig: np.ndarray, recon: np.ndarray, output_path: str, sample_rate: int, n_fft: int = 1024, hop: int = 512) -> None:
    """Save side-by-side spectrograms (orig | recon) as a PNG image."""
    if orig.ndim > 1:
        orig = orig[:, 0]
    if recon.ndim > 1:
        recon = recon[:, 0]
    Sa = _stft_logmag(orig, n_fft=n_fft, hop=hop)
    Sb = _stft_logmag(recon, n_fft=n_fft, hop=hop)
    f = min(Sa.shape[0], Sb.shape[0])
    t = min(Sa.shape[1], Sb.shape[1])
    Sa = Sa[:f, :t]
    Sb = Sb[:f, :t]
    vmin = float(min(Sa.min(), Sb.min()))
    vmax = float(max(Sa.max(), Sb.max()))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0

    def to_img(S: np.ndarray) -> Image.Image:
        X = (S - vmin) / (vmax - vmin + 1e-12)
        X = (np.clip(X, 0.0, 1.0) * 255.0).astype(np.uint8)
        X = np.flipud(X)  # freq axis: low at bottom
        return Image.fromarray(X, mode="L").convert("RGB")

    img_a = to_img(Sa)
    img_b = to_img(Sb)
    h = max(img_a.height, img_b.height)

    def pad_h(im: Image.Image, H: int) -> Image.Image:
        if im.height == H:
            return im
        new = Image.new("RGB", (im.width, H), "white")
        new.paste(im, (0, H - im.height))
        return new

    img_a = pad_h(img_a, h)
    img_b = pad_h(img_b, h)
    out = Image.new("RGB", (img_a.width + img_b.width, h), "white")
    out.paste(img_a, (0, 0))
    out.paste(img_b, (img_a.width, 0))
    out.save(output_path)
