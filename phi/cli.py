from __future__ import annotations

import sys
from typing import Optional
import json

import click

from .constants import PHI, INV_PHI
# Note: heavy submodules (pandas, transforms, fractal, engine, image, audio, mandelbrot)
# are imported lazily inside command functions to avoid requiring optional deps for
# unrelated commands. Video module is also imported lazily in its commands.


@click.group()
def main() -> None:
    """PHI CLI: golden ratio tools for data + infra experiments."""


@main.command()
def ratio() -> None:
    """Print φ and related constants."""
    click.echo(f"PHI (φ)     = {PHI}")
    click.echo(f"1/PHI       = {INV_PHI}")


@main.command()
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input CSV path")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output CSV path")
@click.option("--columns", type=str, default=None, help="Comma-separated columns to transform (default: numeric columns)")
@click.option("--op", type=click.Choice(["golden_scale", "golden_normalize", "fibonacci_smooth"], case_sensitive=False), required=True)
@click.option("--mode", type=click.Choice(["multiply", "divide"], case_sensitive=False), default="multiply", show_default=True, help="For golden_scale")
@click.option("--factor", type=float, default=PHI, show_default=True, help="Scaling factor for golden_scale")
@click.option("--window", type=int, default=5, show_default=True, help="Window for fibonacci_smooth")
@click.option("--infer-dtypes/--no-infer-dtypes", default=True, show_default=True, help="Infer dtypes when reading CSV")
def transform(input_path: str, output_path: str, columns: Optional[str], op: str, mode: str, factor: float, window: int, infer_dtypes: bool) -> None:
    """Apply a φ transform to selected columns and write a new CSV."""
    try:
        import pandas as pd
        from . import transforms
        read_kwargs = {"low_memory": False}
        if infer_dtypes:
            df = pd.read_csv(input_path, **read_kwargs)
        else:
            df = pd.read_csv(input_path, dtype=str, **read_kwargs)

        cols = None
        if columns:
            cols = [c.strip() for c in columns.split(",") if c.strip()]

        if op.lower() == "golden_scale":
            out = transforms.apply_to_dataframe(df, cols, op="golden_scale", factor=factor, mode=mode)
        elif op.lower() == "golden_normalize":
            out = transforms.apply_to_dataframe(df, cols, op="golden_normalize")
        elif op.lower() == "fibonacci_smooth":
            out = transforms.apply_to_dataframe(df, cols, op="fibonacci_smooth", window=window)
        else:
            raise click.UsageError(f"Unsupported op: {op}")

        out.to_csv(output_path, index=False)
        click.echo(f"Wrote: {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@main.group(name="fractal")
def fractal_cmd() -> None:
    """Fractal compression/expansion commands."""


@fractal_cmd.group("video")
def fractal_video() -> None:
    """Video compression/expansion (ratio strategy)."""


@fractal_video.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input video path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path")
@click.option("--spatial-ratio", type=int, default=2, show_default=True, help="Downsample every Nth pixel per axis")
@click.option("--temporal-ratio", type=int, default=1, show_default=True, help="Keep every Nth frame (1 = keep all)")
@click.option("--method", type=click.Choice(["interp", "nearest", "hold"], case_sensitive=False), default="interp", show_default=True, help="Expansion method to record in model")
@click.option("--frame-limit", type=int, default=None, help="Optional max number of downsampled frames to store")
def video_compress(input_path: str, model_path: str, spatial_ratio: int, temporal_ratio: int, method: str, frame_limit: Optional[int]) -> None:
    """Compress a video by spatial/temporal decimation into a JSON model (educational)."""
    try:
        from . import video as video_mod  # lazy import
        cfg = video_mod.VideoConfig(strategy="ratio", spatial_ratio=spatial_ratio, temporal_ratio=temporal_ratio, method=method.lower(), frame_limit=frame_limit)
        bundle = video_mod.compress_video(input_path, cfg)
        video_mod.save_model(bundle, model_path)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_video.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input JSON model path")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output video path (e.g., .mp4 or .gif)")
@click.option("--width", type=int, default=None, help="Target width (optional; must specify both width and height)")
@click.option("--height", type=int, default=None, help="Target height (optional; must specify both width and height)")
@click.option("--fps", type=float, default=None, help="Target FPS (optional; defaults to source fps recorded in model)")
@click.option("--method", type=click.Choice(["interp", "nearest", "hold"], case_sensitive=False), default=None, help="Override expansion method recorded in model")
@click.option("--frames", type=int, default=None, help="Target number of output frames (optional)")
def video_expand(model_path: str, output_path: str, width: Optional[int], height: Optional[int], fps: Optional[float], method: Optional[str], frames: Optional[int]) -> None:
    """Expand a video model back into a playable video file."""
    try:
        from . import video as video_mod  # lazy import
        bundle = video_mod.load_model(model_path)
        if (width is None) ^ (height is None):
            raise click.UsageError("Specify both --width and --height, or neither")
        target_size = (int(width), int(height)) if (width is not None and height is not None) else None
        video_mod.expand_video(
            bundle,
            output_path,
            target_size=target_size,
            fps=fps,
            method=(method.lower() if method else None),
            target_frames=frames,
        )
        click.echo(f"Wrote video: {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_video.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input video path")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output reconstructed video path (e.g., .mp4 or .gif)")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional model JSON save path")
@click.option("--spatial-ratio", type=int, default=2, show_default=True, help="Downsample every Nth pixel per axis")
@click.option("--temporal-ratio", type=int, default=1, show_default=True, help="Keep every Nth frame (1 = keep all)")
@click.option("--method", type=click.Choice(["interp", "nearest", "hold"], case_sensitive=False), default="interp", show_default=True, help="Expansion method")
@click.option("--width", type=int, default=None, help="Target width (optional; must specify both width and height)")
@click.option("--height", type=int, default=None, help="Target height (optional; must specify both width and height)")
@click.option("--fps", type=float, default=None, help="Target FPS (optional; defaults to source fps recorded in model)")
@click.option("--frames", type=int, default=None, help="Target number of output frames (optional)")
@click.option("--compare", "compare_path", type=click.Path(dir_okay=False), default=None, help="Optional side-by-side first-frame compare image output path")
@click.option("--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (MSE/RMSE/PSNR over sampled frames)")
@click.option("--sample-frames", type=int, default=60, show_default=True, help="Number of frames to sample for metrics")
def video_engine(
    input_path: str,
    output_path: str,
    model_path: Optional[str],
    spatial_ratio: int,
    temporal_ratio: int,
    method: str,
    width: Optional[int],
    height: Optional[int],
    fps: Optional[float],
    frames: Optional[int],
    compare_path: Optional[str],
    analyze_output: Optional[str],
    sample_frames: int,
) -> None:
    """Compress + expand a video; optionally save model, first-frame compare, and metrics."""
    try:
        from . import video as video_mod  # lazy import
        cfg = video_mod.VideoConfig(strategy="ratio", spatial_ratio=spatial_ratio, temporal_ratio=temporal_ratio, method=method.lower())
        bundle = video_mod.compress_video(input_path, cfg)
        if model_path:
            video_mod.save_model(bundle, model_path)
            click.echo(f"Wrote model: {model_path}")
        if (width is None) ^ (height is None):
            raise click.UsageError("Specify both --width and --height, or neither")
        target_size = (int(width), int(height)) if (width is not None and height is not None) else None
        video_mod.expand_video(
            bundle,
            output_path,
            target_size=target_size,
            fps=fps,
            method=method.lower(),
            target_frames=frames,
        )
        click.echo(f"Wrote video: {output_path}")
        if compare_path:
            video_mod.save_compare_first_frame(input_path, output_path, compare_path)
            click.echo(f"Wrote compare image: {compare_path}")
        if analyze_output:
            mdf = video_mod.metrics_from_paths(input_path, output_path, sample_frames=sample_frames)
            mdf.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@fractal_cmd.group("three")
def fractal_three() -> None:
    """3D point cloud compression/expansion (ratio strategy)."""


@fractal_three.command("generate")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output PLY path")
@click.option("--points", type=int, default=20000, show_default=True, help="Number of points to generate")
@click.option("--scale", type=float, default=1.0, show_default=True, help="Model scale")
@click.option("--seed", type=int, default=None, help="Random seed (optional)")
@click.option("--preview", type=click.Path(dir_okay=False), default=None, help="Optional 2D projection PNG path")
@click.option("--axis", type=click.Choice(["x", "y", "z"], case_sensitive=False), default="z", show_default=True, help="Projection axis for preview")
@click.option("--height", type=int, default=400, show_default=True, help="Preview image height")
def three_generate(output_path: str, points: int, scale: float, seed: Optional[int], preview: Optional[str], axis: str, height: int) -> None:
    """Generate a Sierpinski tetrahedron point cloud and save to PLY (optionally preview)."""
    try:
        from . import three as three_mod  # lazy import
        pts = three_mod.generate_sierpinski_tetrahedron(n_points=points, scale=scale, seed=seed)
        three_mod.save_point_cloud_ply(pts, output_path)
        click.echo(f"Wrote PLY: {output_path}")
        if preview:
            three_mod.save_projection_from_ply(output_path, preview, axis=axis.lower(), height=height)
            click.echo(f"Wrote preview: {preview}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_three.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input PLY path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path")
@click.option("--ratio", type=int, default=4, show_default=True, help="Keep every Nth point")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method to record in model")
def three_compress(input_path: str, model_path: str, ratio: int, method: str) -> None:
    """Compress a point cloud by point decimation into a JSON model (educational)."""
    try:
        from . import three as three_mod
        pts = three_mod.load_point_cloud_ply(input_path)
        cfg = three_mod.ThreeConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = three_mod.compress_point_cloud(pts, cfg)
        three_mod.save_model(bundle, model_path)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_three.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input JSON model path")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output PLY path")
@click.option("--points", type=int, default=None, help="Target number of output points (optional)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default=None, help="Override expansion method recorded in model")
@click.option("--preview", type=click.Path(dir_okay=False), default=None, help="Optional 2D projection PNG path")
@click.option("--axis", type=click.Choice(["x", "y", "z"], case_sensitive=False), default="z", show_default=True, help="Projection axis for preview")
@click.option("--height", type=int, default=400, show_default=True, help="Preview image height")
def three_expand(model_path: str, output_path: str, points: Optional[int], method: Optional[str], preview: Optional[str], axis: str, height: int) -> None:
    """Expand a 3D model back into a point cloud (PLY)."""
    try:
        from . import three as three_mod
        bundle = three_mod.load_model(model_path)
        recon = three_mod.expand_point_cloud(bundle, target_points=points, method=(method.lower() if method else None))
        three_mod.save_point_cloud_ply(recon, output_path)
        click.echo(f"Wrote PLY: {output_path}")
        if preview:
            three_mod.save_projection_from_ply(output_path, preview, axis=axis.lower(), height=height)
            click.echo(f"Wrote preview: {preview}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_three.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input PLY path")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output reconstructed PLY path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional model JSON save path")
@click.option("--ratio", type=int, default=4, show_default=True, help="Keep every Nth point")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method")
@click.option("--points", type=int, default=None, help="Target number of output points (optional)")
@click.option("--compare", "compare_path", type=click.Path(dir_okay=False), default=None, help="Optional side-by-side projection compare image")
@click.option("--axis", type=click.Choice(["x", "y", "z"], case_sensitive=False), default="z", show_default=True, help="Projection axis for compare image")
@click.option("--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (approx. symmetric Chamfer distance)")
@click.option("--sample-points", type=int, default=2000, show_default=True, help="Number of points to sample for metrics")
def three_engine(
    input_path: str,
    output_path: str,
    model_path: Optional[str],
    ratio: int,
    method: str,
    points: Optional[int],
    compare_path: Optional[str],
    axis: str,
    analyze_output: Optional[str],
    sample_points: int,
) -> None:
    """Compress + expand a point cloud; optionally save model, compare, and metrics."""
    try:
        from . import three as three_mod
        pts = three_mod.load_point_cloud_ply(input_path)
        cfg = three_mod.ThreeConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = three_mod.compress_point_cloud(pts, cfg)
        if model_path:
            three_mod.save_model(bundle, model_path)
            click.echo(f"Wrote model: {model_path}")
        target = points if points is not None else int(bundle.get("orig_count", len(pts)))
        recon = three_mod.expand_point_cloud(bundle, target_points=target, method=method.lower())
        three_mod.save_point_cloud_ply(recon, output_path)
        click.echo(f"Wrote PLY: {output_path}")
        if compare_path:
            three_mod.save_compare_projection_image(input_path, output_path, compare_path, axis=axis.lower())
            click.echo(f"Wrote compare image: {compare_path}")
        if analyze_output:
            mdf = three_mod.metrics_from_paths(input_path, output_path, sample_points=sample_points)
            mdf.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@fractal_cmd.command("mandelbrot")
@click.option("--xmin", type=float, default=-2.0, show_default=True)
@click.option("--xmax", type=float, default=1.0, show_default=True)
@click.option("--ymin", type=float, default=-1.5, show_default=True)
@click.option("--ymax", type=float, default=1.5, show_default=True)
@click.option("--width", type=int, default=1000, show_default=True)
@click.option("--height", type=int, default=1000, show_default=True)
@click.option("--max-iter", type=int, default=256, show_default=True)
@click.option("--output-image", type=click.Path(dir_okay=False), default=None, help="Path to save PNG image (optional)")
@click.option("--output-csv", type=click.Path(dir_okay=False), default=None, help="Path to save CSV of (x,y,iter) (optional)")
def mandelbrot(xmin: float, xmax: float, ymin: float, ymax: float, width: int, height: int, max_iter: int, output_image: Optional[str], output_csv: Optional[str]) -> None:
    """Generate Mandelbrot escape count grid and save as image and/or CSV."""
    try:
        from . import mandelbrot as mandelbrot_mod
        r1, r2, counts = mandelbrot_mod.mandelbrot_escape_counts(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            width=width, height=height, max_iter=max_iter,
        )
        if not output_image and not output_csv:
            raise click.UsageError("Provide at least one of --output-image or --output-csv")
        if output_image:
            mandelbrot_mod.save_image(counts, output_image)
            click.echo(f"Wrote image: {output_image}")
        if output_csv:
            df = mandelbrot_mod.counts_to_dataframe(r1, r2, counts)
            df.to_csv(output_csv, index=False)
            click.echo(f"Wrote CSV: {output_csv}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cmd.group("audio")
def fractal_audio() -> None:
    """Audio compression/expansion (ratio strategy, 16-bit PCM WAV)."""


@fractal_audio.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input WAV (16-bit PCM)")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path")
@click.option("--ratio", type=int, default=2, show_default=True, help="Keep every Nth sample")
@click.option("--method", type=click.Choice(["interp", "hold"], case_sensitive=False), default="interp", show_default=True, help="Expansion method to record in model")
def audio_compress(input_path: str, model_path: str, ratio: int, method: str) -> None:
    """Compress audio by decimation into a compact model (educational)."""
    try:
        from . import audio as audio_mod
        data, sr = audio_mod.load_wav(input_path)
        cfg = audio_mod.AudioConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = audio_mod.compress_audio(data, sr, cfg)
        audio_mod.save_model(bundle, model_path)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_audio.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input JSON model path")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output WAV path")
@click.option("--frames", type=int, default=None, help="Target number of frames (optional)")
@click.option("--method", type=click.Choice(["interp", "hold"], case_sensitive=False), default=None, help="Override method recorded in model")
def audio_expand(model_path: str, output_path: str, frames: Optional[int], method: Optional[str]) -> None:
    """Expand an audio model back into a WAV file."""
    try:
        from . import audio as audio_mod
        bundle = audio_mod.load_model(model_path)
        sr = int(bundle.get("sample_rate", 44100))
        recon = audio_mod.expand_audio(bundle, target_frames=frames, method=(method.lower() if method else None))
        audio_mod.save_wav(output_path, recon, sr)
        click.echo(f"Wrote WAV: {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_audio.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input WAV (16-bit PCM)")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output reconstructed WAV path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional model JSON save path")
@click.option("--ratio", type=int, default=2, show_default=True, help="Keep every Nth sample")
@click.option("--method", type=click.Choice(["interp", "hold"], case_sensitive=False), default="interp", show_default=True, help="Expansion method")
@click.option("--frames", type=int, default=None, help="Target number of frames (optional)")
@click.option("--compare", "compare_path", type=click.Path(dir_okay=False), default=None, help="Optional waveform compare image (PNG)")
@click.option("--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (MSE/RMSE/PSNR)")
def audio_engine(
    input_path: str,
    output_path: str,
    model_path: Optional[str],
    ratio: int,
    method: str,
    frames: Optional[int],
    compare_path: Optional[str],
    analyze_output: Optional[str],
) -> None:
    """Compress + expand audio; optionally save model, plot compare, and metrics."""
    try:
        from . import audio as audio_mod
        data, sr = audio_mod.load_wav(input_path)
        cfg = audio_mod.AudioConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = audio_mod.compress_audio(data, sr, cfg)
        if model_path:
            audio_mod.save_model(bundle, model_path)
            click.echo(f"Wrote model: {model_path}")
        recon = audio_mod.expand_audio(bundle, target_frames=(frames if frames is not None else data.shape[0]), method=method.lower())
        audio_mod.save_wav(output_path, recon, sr)
        click.echo(f"Wrote WAV: {output_path}")
        if compare_path:
            audio_mod.save_compare_plot(data, recon, compare_path)
            click.echo(f"Wrote compare image: {compare_path}")
        if analyze_output:
            mdf = audio_mod.metrics_dataframe(data, recon, sr)
            mdf.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@fractal_cmd.group("image")
def fractal_image() -> None:
    """Image compression/expansion (ratio strategy)."""


@fractal_image.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input image (PNG/JPEG)")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path")
@click.option("--ratio", type=int, default=2, show_default=True, help="Downsample ratio (keep every Nth pixel per axis)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion filter to record in model")
def image_compress(input_path: str, model_path: str, ratio: int, method: str) -> None:
    """Compress an image into a lightweight ratio model (educational)."""
    try:
        from PIL import Image  # local import to avoid hard dependency at import-time
        from . import image as image_mod
        img = Image.open(input_path)
        cfg = image_mod.ImageConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = image_mod.compress_image(img, cfg)
        image_mod.save_model(bundle, model_path)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_image.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input JSON model path")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output image path (PNG/JPEG)")
@click.option("--width", type=int, default=None, help="Target width (optional; must specify both width and height)")
@click.option("--height", type=int, default=None, help="Target height (optional; must specify both width and height)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default=None, help="Override method recorded in model")
def image_expand(model_path: str, output_path: str, width: Optional[int], height: Optional[int], method: Optional[str]) -> None:
    """Expand an image model back into an image file."""
    try:
        from . import image as image_mod
        bundle = image_mod.load_model(model_path)
        if (width is None) ^ (height is None):
            raise click.UsageError("Specify both --width and --height, or neither")
        target_size = (int(width), int(height)) if (width is not None and height is not None) else None
        recon = image_mod.expand_image(bundle, target_size=target_size, method=(method.lower() if method else None))
        recon.save(output_path)
        click.echo(f"Wrote image: {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_image.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input image (PNG/JPEG)")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output reconstructed image path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional model JSON save path")
@click.option("--ratio", type=int, default=2, show_default=True, help="Downsample ratio (keep every Nth pixel per axis)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method")
@click.option("--width", type=int, default=None, help="Target width (optional; must specify both width and height)")
@click.option("--height", type=int, default=None, help="Target height (optional; must specify both width and height)")
@click.option("--compare", "compare_path", type=click.Path(dir_okay=False), default=None, help="Optional side-by-side comparison image output path")
@click.option("--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (MSE/RMSE/PSNR)")
def image_engine(
    input_path: str,
    output_path: str,
    model_path: Optional[str],
    ratio: int,
    method: str,
    width: Optional[int],
    height: Optional[int],
    compare_path: Optional[str],
    analyze_output: Optional[str],
) -> None:
    """Compress + expand an image in one step; optionally save model, compare, and metrics."""
    try:
        from PIL import Image  # local import
        from . import image as image_mod
        img = Image.open(input_path)
        cfg = image_mod.ImageConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = image_mod.compress_image(img, cfg)
        if model_path:
            image_mod.save_model(bundle, model_path)
            click.echo(f"Wrote model: {model_path}")
        if (width is None) ^ (height is None):
            raise click.UsageError("Specify both --width and --height, or neither")
        target_size = (int(width), int(height)) if (width is not None and height is not None) else None
        recon = image_mod.expand_image(bundle, target_size=target_size, method=method.lower())
        recon.save(output_path)
        click.echo(f"Wrote image: {output_path}")
        if compare_path:
            image_mod.save_compare_side_by_side(img, recon, compare_path)
            click.echo(f"Wrote compare image: {compare_path}")
        if analyze_output:
            mdf = image_mod.metrics_dataframe(img, recon)
            mdf.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cmd.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input CSV path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path")
@click.option("--columns", type=str, default=None, help="Comma-separated columns to compress (default: numeric columns)")
@click.option("--depth", type=int, default=4, show_default=True, help="Fractal depth (phi strategy)")
@click.option("--min-segment", type=int, default=8, show_default=True, help="Minimum segment length to stop splitting (phi strategy)")
@click.option("--strategy", type=click.Choice(["phi", "ratio"], case_sensitive=False), default="phi", show_default=True, help="Compression strategy")
@click.option("--ratio", type=int, default=2, show_default=True, help="For strategy=ratio, keep every Nth sample")
@click.option("--infer-dtypes/--no-infer-dtypes", default=True, show_default=True, help="Infer dtypes when reading CSV")
def fractal_compress(input_path: str, model_path: str, columns: Optional[str], depth: int, min_segment: int, strategy: str, ratio: int, infer_dtypes: bool) -> None:
    """Compress selected columns into a phi-fractal JSON model."""
    try:
        import pandas as pd
        from . import fractal as fractal_mod
        read_kwargs = {"low_memory": False}
        if infer_dtypes:
            df = pd.read_csv(input_path, **read_kwargs)
        else:
            df = pd.read_csv(input_path, dtype=str, **read_kwargs)

        cols = None
        if columns:
            cols = [c.strip() for c in columns.split(",") if c.strip()]

        models = fractal_mod.compress_dataframe(
            df,
            columns=cols,
            depth=depth,
            min_segment=min_segment,
            strategy=strategy.lower(),
            ratio=ratio,
        )
        bundle = {
            "version": 1,
            "type": "phi-fractal-models",
            "phi": PHI,
            "input_rows": int(len(df)),
            "columns": list(models.keys()),
            "models": models,
        }
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cmd.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input JSON model path")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output CSV path")
@click.option("--length", type=int, default=None, help="Optional target output length (rows)")
@click.option("--smooth-window", type=int, default=5, show_default=True, help="Fibonacci smoothing window (phi strategy)")
@click.option("--method", type=click.Choice(["interp", "hold"], case_sensitive=False), default="interp", show_default=True, help="Expansion method for ratio strategy")
def fractal_expand(model_path: str, output_path: str, length: Optional[int], smooth_window: int, method: str) -> None:
    """Expand a phi-fractal JSON model back into an approximate CSV."""
    try:
        from . import fractal as fractal_mod
        with open(model_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        models = bundle.get("models")
        if not isinstance(models, dict):
            raise click.UsageError("Invalid model file: missing 'models' dict")

        df = fractal_mod.expand_to_dataframe(models, length=length, smooth_window=smooth_window, method=method.lower())
        df.to_csv(output_path, index=False)
        click.echo(f"Wrote: {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cmd.command("harmonize")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input JSON model path")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output CSV path for harmonized schedule")
@click.option("--column", type=str, default=None, help="Column name to drive harmonization (default: first model column)")
@click.option("--length", type=int, default=None, help="Optional target output length (rows)")
@click.option("--smooth-window", type=int, default=5, show_default=True, help="Fibonacci smoothing window (phi strategy)")
@click.option("--method", type=click.Choice(["interp", "hold"], case_sensitive=False), default="interp", show_default=True, help="Expansion method for ratio strategy")
@click.option("--schedule", type=click.Choice(["split", "backoff"], case_sensitive=False), required=True, help="Harmonization schedule to produce")
@click.option("--total", type=float, default=1.0, show_default=True, help="Total for split schedule (alloc_a + alloc_b = total)")
@click.option("--delta", type=float, default=0.1, show_default=True, help="Tilt amount around golden split for split schedule")
@click.option("--base", type=float, default=0.1, show_default=True, help="Base delay for backoff schedule")
@click.option("--max-delay", type=float, default=10.0, show_default=True, help="Max delay for backoff schedule")
@click.option("--beta", type=float, default=0.5, show_default=True, help="Series influence for backoff schedule")
def fractal_harmonize(
    model_path: str,
    output_path: str,
    column: Optional[str],
    length: Optional[int],
    smooth_window: int,
    method: str,
    schedule: str,
    total: float,
    delta: float,
    base: float,
    max_delay: float,
    beta: float,
) -> None:
    """Expand a fractal model and derive a harmonized infra schedule (split/backoff)."""
    try:
        from . import fractal as fractal_mod
        from . import harmonizer as harmonizer_mod
        with open(model_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        models = bundle.get("models")
        if not isinstance(models, dict):
            raise click.UsageError("Invalid model file: missing 'models' dict")
        df = fractal_mod.expand_to_dataframe(models, length=length, smooth_window=smooth_window, method=method.lower())

        # Pick driving column
        if column is None:
            # Prefer declared order in bundle if available
            cols_list = bundle.get("columns")
            if isinstance(cols_list, list) and cols_list:
                drive_col = cols_list[0]
            else:
                if not df.columns:
                    raise click.UsageError("Expanded dataframe has no columns")
                drive_col = str(df.columns[0])
        else:
            drive_col = column
        if drive_col not in df.columns:
            raise click.UsageError(f"Column '{drive_col}' not found in expanded dataframe")

        s = df[drive_col]
        if schedule.lower() == "split":
            out = harmonizer_mod.harmonize_resource_split(s, total=total, delta=delta)
        elif schedule.lower() == "backoff":
            out = harmonizer_mod.harmonize_backoff(s, base=base, max_delay=max_delay, beta=beta)
        else:
            raise click.UsageError(f"Unknown schedule: {schedule}")

        out.to_csv(output_path, index=False)
        click.echo(f"Wrote: {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cmd.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input CSV path")
@click.option("--recon-output", "recon_output", type=click.Path(dir_okay=False), required=True, help="Output CSV path for reconstructed data")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional path to save model JSON")
@click.option("--columns", type=str, default=None, help="Comma-separated columns to process (default: numeric columns)")
@click.option("--strategy", type=click.Choice(["phi", "ratio"], case_sensitive=False), default="phi", show_default=True)
@click.option("--depth", type=int, default=4, show_default=True, help="Fractal depth (phi strategy)")
@click.option("--min-segment", type=int, default=8, show_default=True, help="Minimum segment length (phi strategy)")
@click.option("--ratio", type=int, default=2, show_default=True, help="Decimation ratio (ratio strategy)")
@click.option("--length", type=int, default=None, help="Target expansion length (rows)")
@click.option("--smooth-window", type=int, default=5, show_default=True, help="Fibonacci smoothing window (phi strategy)")
@click.option("--method", type=click.Choice(["interp", "hold"], case_sensitive=False), default="interp", show_default=True, help="Expansion method for ratio strategy")
@click.option("--plot", "plot_path", type=click.Path(dir_okay=False), default=None, help="Optional path to save a plot of the reconstructed series")
@click.option("--plot-column", type=str, default=None, help="Column to plot (default: first reconstructed column)")
@click.option("--compare-plot", "compare_plot_path", type=click.Path(dir_okay=False), default=None, help="Optional path to save original vs reconstructed overlay plot")
@click.option("--compare-column", type=str, default=None, help="Column to compare (default: first common column)")
@click.option("--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional path to save analysis metrics CSV")
@click.option("--infer-dtypes/--no-infer-dtypes", default=True, show_default=True, help="Infer dtypes when reading CSV")
def fractal_engine(
    input_path: str,
    recon_output: str,
    model_path: Optional[str],
    columns: Optional[str],
    strategy: str,
    depth: int,
    min_segment: int,
    ratio: int,
    length: Optional[int],
    smooth_window: int,
    method: str,
    plot_path: Optional[str],
    plot_column: Optional[str],
    compare_plot_path: Optional[str],
    compare_column: Optional[str],
    analyze_output: Optional[str],
    infer_dtypes: bool,
) -> None:
    """Run compress+expand, optionally plot, compare, and analyze the reconstruction."""
    try:
        import pandas as pd
        from . import engine as engine_mod
        read_kwargs = {"low_memory": False}
        if infer_dtypes:
            df = pd.read_csv(input_path, **read_kwargs)
        else:
            df = pd.read_csv(input_path, dtype=str, **read_kwargs)

        cols = None
        if columns:
            cols = [c.strip() for c in columns.split(",") if c.strip()]

        cfg = engine_mod.FractalConfig(
            strategy=strategy.lower(),
            depth=depth,
            min_segment=min_segment,
            ratio=ratio,
            smooth_window=smooth_window,
            method=method.lower(),
        )
        eng = engine_mod.FractalEngine(cfg)
        eng.compress(df, columns=cols)
        if model_path:
            eng.save_model(model_path)
        recon_df = eng.expand(length=length)
        recon_df.to_csv(recon_output, index=False)
        click.echo(f"Wrote: {recon_output}")
        if plot_path:
            eng.plot_series(recon_df, output_path=plot_path, column=plot_column)
            click.echo(f"Wrote plot: {plot_path}")
        if compare_plot_path:
            eng.plot_compare(df, recon_df, output_path=compare_plot_path, column=compare_column)
            click.echo(f"Wrote compare plot: {compare_plot_path}")
        if analyze_output:
            metrics_df = eng.analyze(df, recon_df, columns=cols)
            metrics_df.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

