from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure local package imports work when running via Streamlit
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# PHI modules
from phi.image import (
    ImageConfig,
    compress_image,
    expand_image,
    save_model as image_save_model,
    load_model as image_load_model,
    metrics_dataframe as image_metrics_df,
    save_compare_side_by_side as image_compare_side_by_side,
)
from phi.audio import (
    AudioConfig,
    load_audio,
    compress_audio,
    expand_audio,
    save_model as audio_save_model,
    load_model as audio_load_model,
    save_wav,
    metrics_dataframe as audio_metrics_df,
    save_compare_plot as audio_compare_plot,
    save_spectrogram_compare as audio_spectrogram_compare,
)
from phi.video import (
    VideoConfig,
    compress_video,
    expand_video,
    save_model as video_save_model,
    load_model as video_load_model,
    metrics_from_paths as video_metrics_df,
    save_compare_first_frame as video_compare_first_frame,
)
from phi.fractal import (
    compress_dataframe as fractal_compress_df,
    expand_to_dataframe as fractal_expand_df,
    phi_fractal_compress,
    ratio_fractal_compress,
    expand_series as fractal_expand_series,
    save_model as fractal_save_model,
    load_model as fractal_load_model,
)
from phi import transforms as T

# Base directories
BASE_UPLOAD = pathlib.Path("out/datahub/uploads")
BASE_HISTORY = pathlib.Path("out/datahub/history")

SUPPORTED_IMAGE = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def ensure_dirs() -> None:
    BASE_UPLOAD.mkdir(parents=True, exist_ok=True)
    BASE_HISTORY.mkdir(parents=True, exist_ok=True)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("..", ".")
    )


def save_uploaded_file(uploaded) -> pathlib.Path:
    ensure_dirs()
    ts = now_stamp()
    up_dir = BASE_UPLOAD / datetime.now().strftime("%Y%m%d")
    up_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{ts}_{slugify(uploaded.name)}"
    dest = up_dir / fname
    with open(dest, "wb") as f:
        f.write(uploaded.getbuffer())
    return dest


def write_manifest(history_dir: pathlib.Path, manifest: Dict) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    mpath = history_dir / "manifest.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def ls_history() -> List[pathlib.Path]:
    ensure_dirs()
    items: List[pathlib.Path] = []
    for p in BASE_HISTORY.glob("**/*/manifest.json"):
        items.append(p.parent)
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return items


def human_size(n: int) -> str:
    u = ["B", "KB", "MB", "GB"]
    i = 0
    x = float(n)
    while x >= 1024.0 and i < len(u) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.1f} {u[i]}"


st.set_page_config(page_title="Data Hub", page_icon="🗂️", layout="wide")

st.title("Data Hub")
st.caption("Upload data, run modules (Harmonizer, Fractal, Compress/Expand), and browse history.")

ensure_dirs()

# Environment checks
if shutil.which("ffmpeg") is None:
    st.warning("ffmpeg not detected on PATH. Reading MP3/FLAC/OGG/M4A and some videos may fail. On macOS: brew install ffmpeg")

with st.expander("Help & How-To", expanded=False):
    st.markdown(
        """
        - __Supported uploads__: Images (.png/.jpg/.jpeg/.bmp/.gif), Audio (.wav/.mp3/.flac/.ogg/.m4a), Video (.mp4/.mov/.avi/.mkv), Tables (.csv/.json).
        - __Where uploads go__: `out/datahub/uploads/<YYYYMMDD>/<timestamp>_<filename>`
        - __Run operations__:
          1) Upload a file in "Upload & Process".
          2) Module defaults to Auto. Adjust if needed.
          3) Set parameters and click the Run button.
          4) Outputs are saved to `out/datahub/history/<module>/<timestamp>_<slug>/` with a `manifest.json`.
        - __Harmonizer (Transforms)__:
          - `golden_scale`: multiply/divide by factor (default φ).
          - `golden_normalize`: min–max to [0, φ].
          - `fibonacci_smooth`: centered moving average with Fibonacci weights.
        - __Fractal__:
          - `phi` strategy: depth + min_segment; expansion can apply smoothing.
          - `ratio` strategy: keep every Nth; expand with interp/hold.
        - __Audio notes__: Non-16-bit WAV uses `pydub` + ffmpeg. Install ffmpeg if MP3/FLAC/OGG/M4A reads fail.
        - __Video notes__: We compute metrics over sampled frames and show a first-frame compare image.
        - __History__: Use the History tab to preview results and download artifacts.
        """
    )

# Tabs: Upload/Process | History
TAB = st.tabs(["Upload & Process", "History"])

with TAB[0]:
    st.header("Upload a file")
    uploaded = st.file_uploader(
        "Select a file",
        type=list({
            *{e[1:] for e in SUPPORTED_IMAGE},
            *{e[1:] for e in SUPPORTED_VIDEO},
            *{e[1:] for e in SUPPORTED_AUDIO},
            "csv", "json"
        }),
        accept_multiple_files=False,
    )

    if uploaded is not None:
        saved_path = save_uploaded_file(uploaded)
        st.success(f"Saved to {saved_path}")
        ext = saved_path.suffix.lower()

        st.subheader("Choose module and operation")
        module = st.selectbox(
            "Module",
            [
                "Auto (based on file)",
                "Image",
                "Audio",
                "Video",
                "Table (CSV/JSON rows)",
                "JSON Series",
                "Harmonizer (Transforms)",
                "Fractal",
            ],
            index=0,
        )

        # Auto-detect module by extension
        if module.startswith("Auto"):
            if ext in SUPPORTED_IMAGE:
                module = "Image"
            elif ext in SUPPORTED_AUDIO:
                module = "Audio"
            elif ext in SUPPORTED_VIDEO:
                module = "Video"
            elif ext == ".csv":
                module = "Table (CSV/JSON rows)"
            elif ext == ".json":
                # We'll peek into JSON structure later
                module = "Table (CSV/JSON rows)"
            else:
                st.warning("Unknown file type; select a module manually.")

        # Per-module UIs
        if module == "Image" and ext in SUPPORTED_IMAGE:
            st.markdown("---")
            st.subheader("Image: Compress → Expand → Compare/Measure")
            with Image.open(saved_path) as img:
                st.image(img, caption="Original image", use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                ratio = st.slider("Downsample ratio", 1, 8, 2)
            with c2:
                method = st.selectbox("Upsample method", ["interp", "nearest"], index=0)
            with c3:
                run = st.button("Run Image Pipeline")

            if run:
                hist_dir = BASE_HISTORY / "image" / f"{now_stamp()}_{saved_path.stem}"
                hist_dir.mkdir(parents=True, exist_ok=True)

                # Compress
                with Image.open(saved_path) as img:
                    bundle = compress_image(img, ImageConfig(ratio=ratio, method=method))
                model_path = hist_dir / "image_model.json"
                image_save_model(bundle, str(model_path))

                # Expand
                recon_img = expand_image(bundle, target_size=None, method=method)
                recon_path = hist_dir / "image_recon.png"
                recon_img.save(recon_path)

                # Compare + Metrics
                with Image.open(saved_path) as orig_img:
                    compare_path = hist_dir / "image_compare.png"
                    image_compare_side_by_side(orig_img, recon_img, str(compare_path))
                    dfm = image_metrics_df(orig_img, recon_img)
                metrics_path = hist_dir / "image_metrics.csv"
                dfm.to_csv(metrics_path, index=False)

                manifest = {
                    "module": "image",
                    "op": "compress_expand_compare",
                    "input": str(saved_path),
                    "params": {"ratio": ratio, "method": method},
                    "outputs": {
                        "model": str(model_path),
                        "recon": str(recon_path),
                        "compare": str(compare_path),
                        "metrics": str(metrics_path),
                    },
                    "created_at": now_stamp(),
                }
                write_manifest(hist_dir, manifest)

                st.success("Image pipeline complete.")
                st.image(str(compare_path), caption="Compare: Original | Recon", use_container_width=True)
                st.dataframe(dfm)

        elif module == "Audio" and ext in SUPPORTED_AUDIO:
            st.markdown("---")
            st.subheader("Audio: Compress → Expand → Compare/Measure")

            c1, c2, c3 = st.columns(3)
            with c1:
                ratio = st.slider("Downsample ratio", 1, 16, 2)
            with c2:
                method = st.selectbox("Upsample method", ["interp", "hold"], index=0)
            with c3:
                run = st.button("Run Audio Pipeline")

            if run:
                hist_dir = BASE_HISTORY / "audio" / f"{now_stamp()}_{saved_path.stem}"
                hist_dir.mkdir(parents=True, exist_ok=True)

                # Load audio (WAV native; others require pydub+ffmpeg)
                try:
                    data, sr = load_audio(str(saved_path))
                except Exception as e:
                    st.error(f"Failed to load audio: {e}")
                    st.stop()

                # Compress
                bundle = compress_audio(data, sr, AudioConfig(ratio=ratio, method=method))
                model_path = hist_dir / "audio_model.json"
                audio_save_model(bundle, str(model_path))

                # Expand
                recon = expand_audio(bundle, target_frames=None, method=method)
                recon_path = hist_dir / "audio_recon.wav"
                save_wav(str(recon_path), recon, sr)

                # Metrics + Compare plots
                dfm = audio_metrics_df(data, recon, sr)
                metrics_path = hist_dir / "audio_metrics.csv"
                dfm.to_csv(metrics_path, index=False)

                plot_path = hist_dir / "audio_compare.png"
                audio_compare_plot(data, recon, str(plot_path))
                spec_path = hist_dir / "audio_spectrogram_compare.png"
                try:
                    audio_spectrogram_compare(data, recon, str(spec_path), sample_rate=sr)
                except Exception:
                    # STFT may fail on super-short signals; ignore gracefully
                    pass

                manifest = {
                    "module": "audio",
                    "op": "compress_expand_compare",
                    "input": str(saved_path),
                    "params": {"ratio": ratio, "method": method},
                    "outputs": {
                        "model": str(model_path),
                        "recon": str(recon_path),
                        "metrics": str(metrics_path),
                        "waveform_plot": str(plot_path),
                        "spectrogram_plot": str(spec_path),
                    },
                    "created_at": now_stamp(),
                }
                write_manifest(hist_dir, manifest)

                st.success("Audio pipeline complete.")
                st.image(str(plot_path), caption="Waveform compare", use_container_width=True)
                if spec_path.exists():
                    st.image(str(spec_path), caption="Spectrogram compare", use_container_width=True)
                st.dataframe(dfm)

        elif module == "Video" and ext in SUPPORTED_VIDEO:
            st.markdown("---")
            st.subheader("Video: Compress → Expand → Compare/Measure")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                spatial_ratio = st.slider("Spatial ratio", 1, 8, 2)
            with c2:
                temporal_ratio = st.slider("Temporal ratio", 1, 8, 1)
            with c3:
                method = st.selectbox("Method", ["interp", "nearest", "hold"], index=0)
            with c4:
                frame_limit = st.number_input("Frame limit (optional)", min_value=0, value=0)
            run = st.button("Run Video Pipeline")

            if run:
                hist_dir = BASE_HISTORY / "video" / f"{now_stamp()}_{saved_path.stem}"
                hist_dir.mkdir(parents=True, exist_ok=True)

                cfg = VideoConfig(
                    spatial_ratio=int(spatial_ratio),
                    temporal_ratio=int(temporal_ratio),
                    method=method,
                    frame_limit=int(frame_limit) if int(frame_limit) > 0 else None,
                )

                # Compress
                bundle = compress_video(str(saved_path), cfg)
                model_path = hist_dir / "video_model.json"
                video_save_model(bundle, str(model_path))

                # Expand
                recon_path = hist_dir / "video_recon.mp4"
                expand_video(bundle, str(recon_path))

                # Compare + Metrics
                compare_path = hist_dir / "video_compare.png"
                video_compare_first_frame(str(saved_path), str(recon_path), str(compare_path))
                dfm = video_metrics_df(str(saved_path), str(recon_path))
                metrics_path = hist_dir / "video_metrics.csv"
                dfm.to_csv(metrics_path, index=False)

                manifest = {
                    "module": "video",
                    "op": "compress_expand_compare",
                    "input": str(saved_path),
                    "params": {
                        "spatial_ratio": spatial_ratio,
                        "temporal_ratio": temporal_ratio,
                        "method": method,
                        "frame_limit": int(frame_limit) if int(frame_limit) > 0 else None,
                    },
                    "outputs": {
                        "model": str(model_path),
                        "recon": str(recon_path),
                        "compare": str(compare_path),
                        "metrics": str(metrics_path),
                    },
                    "created_at": now_stamp(),
                }
                write_manifest(hist_dir, manifest)

                st.success("Video pipeline complete.")
                st.image(str(compare_path), caption="First-frame compare", use_container_width=True)
                st.dataframe(dfm)

        elif module in {"Table (CSV/JSON rows)", "Harmonizer (Transforms)", "Fractal"} and ext in {".csv", ".json"}:
            st.markdown("---")
            st.subheader("Table/Series operations")

            # Load into DataFrame or Series
            df: Optional[pd.DataFrame] = None
            series: Optional[pd.Series] = None
            source_kind = "table"

            try:
                if ext == ".csv":
                    df = pd.read_csv(saved_path)
                else:
                    with open(saved_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        if len(data) > 0 and isinstance(data[0], dict):
                            df = pd.DataFrame(data)
                        else:
                            # Assume numeric series
                            series = pd.Series(data, dtype=float)
                            source_kind = "series"
                    elif isinstance(data, dict):
                        # Try dict of arrays → DataFrame
                        try:
                            df = pd.DataFrame(data)
                        except Exception:
                            st.error("Unsupported JSON structure for automatic handling.")
                            st.stop()
                    else:
                        st.error("Unsupported JSON structure for automatic handling.")
                        st.stop()
            except Exception as e:
                st.error(f"Failed to read table/JSON: {e}")
                st.stop()

            if df is not None:
                st.write("Preview (first 100 rows):")
                st.dataframe(df.head(100))

                with st.expander("Harmonizer (Transforms)", expanded=True):
                    cols = st.multiselect("Columns (empty = numeric columns)", options=list(df.columns))
                    op = st.selectbox("Operation", ["golden_scale", "golden_normalize", "fibonacci_smooth"], index=0)
                    params: Dict[str, Union[str, int, float]] = {}
                    if op == "golden_scale":
                        params["factor"] = st.number_input("factor", value=float(1.61803398875))
                        params["mode"] = st.selectbox("mode", ["multiply", "divide"], index=0)
                    elif op == "fibonacci_smooth":
                        params["window"] = st.slider("window", 3, 51, 5, step=2)
                    run_tf = st.button("Run Transform")

                    if run_tf:
                        hist_dir = BASE_HISTORY / "table" / f"{now_stamp()}_{saved_path.stem}_transform"
                        hist_dir.mkdir(parents=True, exist_ok=True)
                        out_df = T.apply_to_dataframe(df, columns=cols if cols else None, op=op, **params)
                        out_path = hist_dir / "transformed.csv"
                        out_df.to_csv(out_path, index=False)
                        manifest = {
                            "module": "table",
                            "op": f"transform:{op}",
                            "input": str(saved_path),
                            "params": {"columns": cols if cols else None, **params},
                            "outputs": {"csv": str(out_path)},
                            "created_at": now_stamp(),
                        }
                        write_manifest(hist_dir, manifest)
                        st.success("Transform complete.")
                        st.dataframe(out_df.head(100))

                with st.expander("Fractal (Compact/Expand)", expanded=True):
                    strategy = st.selectbox("Strategy", ["phi", "ratio"], index=0)
                    depth = st.slider("depth (phi)", 1, 8, 4)
                    min_segment = st.slider("min_segment (phi)", 2, 64, 8)
                    ratio = st.slider("ratio (ratio)", 1, 16, 2)
                    method = st.selectbox("expand method (ratio)", ["interp", "hold"], index=0)
                    run_fr = st.button("Run Fractal")

                    if run_fr:
                        hist_dir = BASE_HISTORY / "table" / f"{now_stamp()}_{saved_path.stem}_fractal"
                        hist_dir.mkdir(parents=True, exist_ok=True)
                        models = fractal_compress_df(
                            df,
                            columns=None,
                            depth=depth,
                            min_segment=min_segment,
                            strategy=strategy,
                            ratio=ratio,
                        )
                        models_path = hist_dir / "models.json"
                        with open(models_path, "w", encoding="utf-8") as f:
                            json.dump(models, f)
                        recon_df = fractal_expand_df(models, length=None, smooth_window=5, method=method)
                        recon_path = hist_dir / "reconstructed.csv"
                        recon_df.to_csv(recon_path, index=False)

                        manifest = {
                            "module": "table",
                            "op": f"fractal:{strategy}",
                            "input": str(saved_path),
                            "params": {
                                "depth": depth,
                                "min_segment": min_segment,
                                "ratio": ratio,
                                "method": method,
                            },
                            "outputs": {"models": str(models_path), "reconstructed": str(recon_path)},
                            "created_at": now_stamp(),
                        }
                        write_manifest(hist_dir, manifest)
                        st.success("Fractal compact/expand complete.")
                        st.dataframe(recon_df.head(100))

            elif series is not None:
                st.write("Series length:", len(series))
                st.line_chart(series.head(min(len(series), 500)))

                with st.expander("Harmonizer (Transforms)", expanded=True):
                    op = st.selectbox("Operation", ["golden_scale", "golden_normalize", "fibonacci_smooth"], index=0)
                    params: Dict[str, Union[str, int, float]] = {}
                    if op == "golden_scale":
                        params["factor"] = st.number_input("factor", value=float(1.61803398875))
                        params["mode"] = st.selectbox("mode", ["multiply", "divide"], index=0)
                    elif op == "fibonacci_smooth":
                        params["window"] = st.slider("window", 3, 51, 5, step=2)
                    run_tf = st.button("Run Series Transform")

                    if run_tf:
                        hist_dir = BASE_HISTORY / "json_series" / f"{now_stamp()}_{saved_path.stem}_transform"
                        hist_dir.mkdir(parents=True, exist_ok=True)
                        if op == "golden_scale":
                            out_s = T.golden_scale(series, **params)
                        elif op == "golden_normalize":
                            out_s = T.golden_normalize(series)
                        else:
                            out_s = T.fibonacci_smooth(series, **params)
                        out_path = hist_dir / "series_transformed.csv"
                        out_s.to_csv(out_path, index=False, header=["value"])  # single column
                        manifest = {
                            "module": "json_series",
                            "op": f"transform:{op}",
                            "input": str(saved_path),
                            "params": params,
                            "outputs": {"csv": str(out_path)},
                            "created_at": now_stamp(),
                        }
                        write_manifest(hist_dir, manifest)
                        st.success("Series transform complete.")
                        st.dataframe(out_s.head(100))

                with st.expander("Fractal (Compact/Expand)", expanded=True):
                    strategy = st.selectbox("Strategy", ["phi", "ratio"], index=0)
                    depth = st.slider("depth (phi)", 1, 8, 4)
                    min_segment = st.slider("min_segment (phi)", 2, 64, 8)
                    ratio = st.slider("ratio (ratio)", 1, 16, 2)
                    method = st.selectbox("expand method (ratio)", ["interp", "hold"], index=0)
                    run_fr = st.button("Run Series Fractal")

                    if run_fr:
                        hist_dir = BASE_HISTORY / "json_series" / f"{now_stamp()}_{saved_path.stem}_fractal"
                        hist_dir.mkdir(parents=True, exist_ok=True)
                        if strategy == "phi":
                            model = phi_fractal_compress(series, depth=depth, min_segment=min_segment)
                        else:
                            model = ratio_fractal_compress(series, ratio=ratio)
                        model_path = hist_dir / "series_model.json"
                        fractal_save_model(model, str(model_path))
                        recon = fractal_expand_series(model, length=None, smooth_window=5, method=method)
                        recon_path = hist_dir / "series_reconstructed.csv"
                        recon.to_csv(recon_path, index=False, header=["value"])  # single column

                        manifest = {
                            "module": "json_series",
                            "op": f"fractal:{strategy}",
                            "input": str(saved_path),
                            "params": {
                                "depth": depth,
                                "min_segment": min_segment,
                                "ratio": ratio,
                                "method": method,
                            },
                            "outputs": {"model": str(model_path), "reconstructed": str(recon_path)},
                            "created_at": now_stamp(),
                        }
                        write_manifest(hist_dir, manifest)
                        st.success("Series fractal compact/expand complete.")
                        st.dataframe(recon.head(100))

        elif module == "JSON Series" and ext == ".json":
            st.info("Use the Table/Series handler above for JSON; it auto-detects list-of-numbers as series.")

with TAB[1]:
    st.header("History")
    hist_items = ls_history()
    if not hist_items:
        st.info("No history yet. Run an operation from Upload & Process.")
    else:
        # filter by module
        modules = sorted({p.parent.name for p in hist_items})
        sel_module = st.selectbox("Filter by module", options=["All"] + modules, index=0)
        for d in hist_items:
            if sel_module != "All" and d.parent.name != sel_module:
                continue
            mpath = d / "manifest.json"
            try:
                info = json.loads(mpath.read_text())
            except Exception:
                info = {"module": d.parent.name, "error": "manifest read failed"}
            with st.expander(f"{d.parent.name} | {d.name}", expanded=False):
                st.json(info)
                # quick preview if exists
                if info.get("module") == "image":
                    cmp = d / "image_compare.png"
                    if cmp.exists():
                        st.image(str(cmp), use_container_width=True)
                elif info.get("module") == "audio":
                    plot = d / "audio_compare.png"
                    if plot.exists():
                        st.image(str(plot), use_container_width=True)
                    spec = d / "audio_spectrogram_compare.png"
                    if spec.exists():
                        st.image(str(spec), use_container_width=True)
                elif info.get("module") == "video":
                    cmp = d / "video_compare.png"
                    if cmp.exists():
                        st.image(str(cmp), use_container_width=True)
                elif info.get("module") in {"table", "json_series"}:
                    # show first rows of any CSV
                    for f in sorted(d.glob("*.csv")):
                        try:
                            st.caption(f.name)
                            st.dataframe(pd.read_csv(f).head(50))
                        except Exception:
                            pass
                # download buttons
                files = list(sorted(d.glob("*")))
                if files:
                    st.write("Files:")
                    for f in files:
                        if f.is_file():
                            with open(f, "rb") as fh:
                                st.download_button(
                                    label=f"Download {f.name}",
                                    data=fh.read(),
                                    file_name=f.name,
                                    key=f"dl_{d.name}_{f.name}",
                                )
