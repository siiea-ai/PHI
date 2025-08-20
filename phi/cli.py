from __future__ import annotations

import sys
import os
from typing import Optional, Tuple
import json
import time

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
@fractal_cmd.group("ai")
def fractal_ai() -> None:
    """AI model compression/expansion (ratio strategy)."""


@fractal_ai.command("generate")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path (full model)")
@click.option("--input-dim", type=int, required=True, help="Input dimension")
@click.option("--output-dim", type=int, required=True, help="Output dimension")
@click.option("--depth", type=int, default=3, show_default=True, help="Hidden depth (number of hidden layers)")
@click.option("--base-width", type=int, default=64, show_default=True, help="Base hidden width")
@click.option("--mode", type=click.Choice(["phi", "fibonacci", "fixed"], case_sensitive=False), default="phi", show_default=True, help="Hidden width schedule")
@click.option("--act-hidden", type=str, default="relu", show_default=True, help="Activation for hidden layers")
@click.option("--act-output", type=str, default="sigmoid", show_default=True, help="Activation for output layer")
@click.option("--seed", type=int, default=None, help="Random seed (optional)")
@click.option("--export-keras", type=click.Path(dir_okay=False), default=None, help="Optional path to export Keras model (.keras preferred; legacy .h5 supported). If no extension is provided, .keras is used.")
def ai_generate(output_path: str, input_dim: int, output_dim: int, depth: int, base_width: int, mode: str, act_hidden: str, act_output: str, seed: Optional[int], export_keras: Optional[str]) -> None:
    """Generate a full AI model bundle (JSON)."""
    try:
        from . import ai as ai_mod  # lazy import
        bundle = ai_mod.generate_full_model(
            input_dim=input_dim,
            output_dim=output_dim,
            depth=depth,
            base_width=base_width,
            mode=mode.lower(),
            act_hidden=act_hidden,
            act_output=act_output,
            seed=seed,
        )
        ai_mod.save_model(bundle, output_path)
        click.echo(f"Wrote model: {output_path}")
        if export_keras:
            ai_mod.export_keras(bundle, export_keras)
            click.echo(f"Wrote Keras model: {export_keras}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_ai.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full model JSON path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output compressed JSON model path")
@click.option("--ratio", type=int, default=2, show_default=True, help="Keep every Nth hidden neuron")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method to record in model")
def ai_compress(input_path: str, model_path: str, ratio: int, method: str) -> None:
    """Compress a full AI model bundle into a compact ratio model (educational)."""
    try:
        from . import ai as ai_mod  # lazy import
        full = ai_mod.load_model(input_path)
        cfg = ai_mod.AIConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = ai_mod.compress_model(full, cfg)
        ai_mod.save_model(bundle, model_path)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_ai.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input compressed JSON model path")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output full JSON model path")
@click.option("--hidden", type=str, default=None, help="Comma-separated target hidden widths (e.g., 64,32,16)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default=None, help="Override expansion method recorded in model")
@click.option("--seed", type=int, default=None, help="Random seed for stochastic expansion (optional)")
@click.option("--export-keras", type=click.Path(dir_okay=False), default=None, help="Optional path to export Keras model (.keras preferred; legacy .h5 supported). If no extension is provided, .keras is used.")
def ai_expand(model_path: str, output_path: str, hidden: Optional[str], method: Optional[str], seed: Optional[int], export_keras: Optional[str]) -> None:
    """Expand a compressed AI model bundle back to a full model bundle."""
    try:
        from . import ai as ai_mod  # lazy import
        bundle = ai_mod.load_model(model_path)
        target_hidden = None
        if hidden:
            try:
                target_hidden = [int(x.strip()) for x in hidden.split(",") if x.strip()]
            except Exception:
                raise click.UsageError("--hidden must be a comma-separated list of integers, e.g., 64,32,16")
        full = ai_mod.expand_model(bundle, target_hidden=target_hidden, method=(method.lower() if method else None), seed=seed)
        ai_mod.save_model(full, output_path)
        click.echo(f"Wrote model: {output_path}")
        if export_keras:
            ai_mod.export_keras(full, export_keras)
            click.echo(f"Wrote Keras model: {export_keras}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_ai.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full model JSON path")
@click.option("--recon-output", "recon_output", type=click.Path(dir_okay=False), required=True, help="Output full model JSON path (reconstructed)")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional path to save compressed model JSON")
@click.option("--ratio", type=int, default=2, show_default=True, help="Keep every Nth hidden neuron")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method")
@click.option("--analyze", "--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (per-layer MSE and totals)")
@click.option("--export-keras", type=click.Path(dir_okay=False), default=None, help="Optional path to export reconstructed Keras model (.keras preferred; legacy .h5 supported). If no extension is provided, .keras is used.")
def ai_engine(input_path: str, recon_output: str, model_path: Optional[str], ratio: int, method: str, analyze_output: Optional[str], export_keras: Optional[str]) -> None:
    """Compress + expand an AI model; optionally save compressed model, metrics, and Keras export."""
    try:
        from . import ai as ai_mod  # lazy import
        full_in = ai_mod.load_model(input_path)
        cfg = ai_mod.AIConfig(strategy="ratio", ratio=ratio, method=method.lower())
        comp = ai_mod.compress_model(full_in, cfg)
        if model_path:
            ai_mod.save_model(comp, model_path)
            click.echo(f"Wrote model: {model_path}")
        target_hidden = [int(x) for x in full_in.get("hidden", [])]
        full_out = ai_mod.expand_model(comp, target_hidden=target_hidden, method=method.lower())
        ai_mod.save_model(full_out, recon_output)
        click.echo(f"Wrote model: {recon_output}")
        if analyze_output:
            mdf = ai_mod.metrics_from_paths(input_path, recon_output)
            mdf.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
        if export_keras:
            ai_mod.export_keras(full_out, export_keras)
            click.echo(f"Wrote Keras model: {export_keras}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cmd.group("quantum")
def fractal_quantum() -> None:
    """Quantum circuit compression/expansion (ratio strategy)."""


@fractal_quantum.command("generate")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path (full circuit)")
@click.option("--qubits", type=int, required=True, help="Number of qubits")
@click.option("--depth", type=int, default=1, show_default=True, help="Pattern depth (repetitions)")
@click.option("--seed", type=int, default=None, help="Random seed (optional)")
@click.option("--export-qasm", type=click.Path(dir_okay=False), default=None, help="Optional path to export OpenQASM 2.0 (.qasm)")
def quantum_generate(output_path: str, qubits: int, depth: int, seed: Optional[int], export_qasm: Optional[str]) -> None:
    """Generate a full quantum circuit bundle (JSON)."""
    try:
        from . import quantum as qmod  # lazy import
        bundle = qmod.generate_full_circuit(num_qubits=qubits, depth=depth, seed=seed)
        qmod.save_model(bundle, output_path)
        click.echo(f"Wrote model: {output_path}")
        if export_qasm:
            qmod.export_qasm(bundle, export_qasm)
            click.echo(f"Wrote QASM: {export_qasm}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_quantum.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full circuit JSON path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output compressed JSON model path")
@click.option("--ratio", type=int, default=2, show_default=True, help="Keep every Nth qubit")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method to record in model")
def quantum_compress(input_path: str, model_path: str, ratio: int, method: str) -> None:
    """Compress a full quantum circuit into a compact ratio model (educational)."""
    try:
        from . import quantum as qmod  # lazy import
        full = qmod.load_model(input_path)
        cfg = qmod.QuantumConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = qmod.compress_circuit(full, cfg)
        qmod.save_model(bundle, model_path)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_quantum.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input compressed JSON model path (or full)")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output full circuit JSON path")
@click.option("--qubits", type=int, default=None, help="Target number of qubits (optional)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default=None, help="Override expansion method recorded in model")
@click.option("--export-qasm", type=click.Path(dir_okay=False), default=None, help="Optional path to export OpenQASM 2.0 (.qasm)")
def quantum_expand(model_path: str, output_path: str, qubits: Optional[int], method: Optional[str], export_qasm: Optional[str]) -> None:
    """Expand a compressed quantum circuit bundle back to a full circuit bundle."""
    try:
        from . import quantum as qmod  # lazy import
        bundle = qmod.load_model(model_path)
        full = qmod.expand_circuit(bundle, target_qubits=qubits, method=(method.lower() if method else None))
        qmod.save_model(full, output_path)
        click.echo(f"Wrote model: {output_path}")
        if export_qasm:
            qmod.export_qasm(full, export_qasm)
            click.echo(f"Wrote QASM: {export_qasm}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_quantum.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full circuit JSON path")
@click.option("--recon-output", "recon_output", type=click.Path(dir_okay=False), required=True, help="Output full circuit JSON path (reconstructed)")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional path to save compressed model JSON")
@click.option("--ratio", type=int, default=2, show_default=True, help="Keep every Nth qubit")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method")
@click.option("--analyze", "--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (gate counts and depth)")
@click.option("--export-qasm", type=click.Path(dir_okay=False), default=None, help="Optional path to export reconstructed OpenQASM 2.0 (.qasm)")
def quantum_engine(input_path: str, recon_output: str, model_path: Optional[str], ratio: int, method: str, analyze_output: Optional[str], export_qasm: Optional[str]) -> None:
    """Compress + expand a quantum circuit; optionally save compressed model, metrics, and QASM export."""
    try:
        from . import quantum as qmod  # lazy import
        full_in = qmod.load_model(input_path)
        cfg = qmod.QuantumConfig(strategy="ratio", ratio=ratio, method=method.lower())
        comp = qmod.compress_circuit(full_in, cfg)
        if model_path:
            qmod.save_model(comp, model_path)
            click.echo(f"Wrote model: {model_path}")
        target_qubits = int(full_in.get("num_qubits", 0))
        full_out = qmod.expand_circuit(comp, target_qubits=target_qubits, method=method.lower())
        qmod.save_model(full_out, recon_output)
        click.echo(f"Wrote model: {recon_output}")
        if analyze_output:
            mdf = qmod.metrics_from_paths(input_path, recon_output)
            mdf.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
        if export_qasm:
            qmod.export_qasm(full_out, export_qasm)
            click.echo(f"Wrote QASM: {export_qasm}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_quantum.command("export-qasm")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input circuit JSON path (full or compressed)")
@click.option("--output", "qasm_path", type=click.Path(dir_okay=False), required=True, help="Output OpenQASM 2.0 path (.qasm)")
def quantum_export_qasm(model_path: str, qasm_path: str) -> None:
    """Export a (full or compressed) circuit JSON to OpenQASM 2.0."""
    try:
        from . import quantum as qmod  # lazy import
        bundle = qmod.load_model(model_path)
        qmod.export_qasm(bundle, qasm_path)
        click.echo(f"Wrote QASM: {qasm_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cmd.group("cosmos")
def fractal_cosmos() -> None:
    """Cosmic field compression/expansion (ratio strategy)."""


@fractal_cosmos.command("generate")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path (full cosmos field)")
@click.option("--width", type=int, required=True, help="Field width")
@click.option("--height", type=int, required=True, help="Field height")
@click.option("--octaves", type=int, default=4, show_default=True, help="fBm octaves")
@click.option("--seed", type=int, default=None, help="Random seed (optional)")
@click.option("--preview", "preview_path", type=click.Path(dir_okay=False), default=None, help="Optional preview PNG path")
def cosmos_generate(output_path: str, width: int, height: int, octaves: int, seed: Optional[int], preview_path: Optional[str]) -> None:
    """Generate a full cosmos field bundle (JSON)."""
    try:
        from . import cosmos as cmod  # lazy import
        bundle = cmod.generate_full_field(width=width, height=height, octaves=octaves, seed=seed)
        cmod.save_model(bundle, output_path)
        click.echo(f"Wrote model: {output_path}")
        if preview_path:
            cmod.save_image_from_model(bundle, preview_path)
            click.echo(f"Wrote preview: {preview_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cosmos.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full cosmos JSON path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output compressed JSON model path")
@click.option("--ratio", type=int, default=2, show_default=True, help="Keep every Nth pixel per axis")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method to record in model")
def cosmos_compress(input_path: str, model_path: str, ratio: int, method: str) -> None:
    """Compress a full cosmos field into a compact ratio model (educational)."""
    try:
        from . import cosmos as cmod  # lazy import
        full = cmod.load_model(input_path)
        cfg = cmod.CosmosConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = cmod.compress_field(full, cfg)
        cmod.save_model(bundle, model_path)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cosmos.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input compressed JSON model path (or full)")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output full cosmos JSON path")
@click.option("--width", type=int, default=None, help="Target width (optional; must specify both width and height)")
@click.option("--height", type=int, default=None, help="Target height (optional; must specify both width and height)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default=None, help="Override expansion method recorded in model")
@click.option("--preview", "preview_path", type=click.Path(dir_okay=False), default=None, help="Optional preview PNG path")
def cosmos_expand(model_path: str, output_path: str, width: Optional[int], height: Optional[int], method: Optional[str], preview_path: Optional[str]) -> None:
    """Expand a compressed cosmos bundle back to a full bundle."""
    try:
        from . import cosmos as cmod  # lazy import
        bundle = cmod.load_model(model_path)
        if (width is None) ^ (height is None):
            raise click.UsageError("Specify both --width and --height, or neither")
        target_size = (int(width), int(height)) if (width is not None and height is not None) else None
        full = cmod.expand_field(bundle, target_size=target_size, method=(method.lower() if method else None))
        cmod.save_model(full, output_path)
        click.echo(f"Wrote model: {output_path}")
        if preview_path:
            cmod.save_image_from_model(full, preview_path)
            click.echo(f"Wrote preview: {preview_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cosmos.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full cosmos JSON path")
@click.option("--recon-output", "recon_output", type=click.Path(dir_okay=False), required=True, help="Output full cosmos JSON path (reconstructed)")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional path to save compressed model JSON")
@click.option("--ratio", type=int, default=2, show_default=True, help="Keep every Nth pixel per axis")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method")
@click.option("--compare", "compare_path", type=click.Path(dir_okay=False), default=None, help="Optional side-by-side compare image output path")
@click.option("--analyze", "--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (MSE/RMSE/PSNR + spectral)")
def cosmos_engine(input_path: str, recon_output: str, model_path: Optional[str], ratio: int, method: str, compare_path: Optional[str], analyze_output: Optional[str]) -> None:
    """Compress + expand a cosmos field; optionally save compressed model, metrics, and preview compare."""
    try:
        from . import cosmos as cmod  # lazy import
        full_in = cmod.load_model(input_path)
        cfg = cmod.CosmosConfig(strategy="ratio", ratio=ratio, method=method.lower())
        comp = cmod.compress_field(full_in, cfg)
        if model_path:
            cmod.save_model(comp, model_path)
            click.echo(f"Wrote model: {model_path}")
        ow = int(full_in.get("width", 0))
        oh = int(full_in.get("height", 0))
        full_out = cmod.expand_field(comp, target_size=(ow, oh), method=method.lower())
        cmod.save_model(full_out, recon_output)
        click.echo(f"Wrote model: {recon_output}")
        if compare_path:
            cmod.save_compare_image(full_in, full_out, compare_path)
            click.echo(f"Wrote compare: {compare_path}")
        if analyze_output:
            mdf = cmod.metrics_from_paths(input_path, recon_output)
            mdf.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cosmos.command("preview")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input cosmos JSON path (full or compressed)")
@click.option("--output", "image_path", type=click.Path(dir_okay=False), required=True, help="Output preview PNG path")
@click.option("--cmap", type=str, default="viridis", show_default=True, help="Matplotlib colormap name")
def cosmos_preview(model_path: str, image_path: str, cmap: str) -> None:
    """Export a preview image from a cosmos model (full or compressed)."""
    try:
        from . import cosmos as cmod  # lazy import
        bundle = cmod.load_model(model_path)
        cmod.save_image_from_model(bundle, image_path, cmap=cmap)
        click.echo(f"Wrote preview: {image_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cmd.group("multiverse")
def fractal_multiverse() -> None:
    """Multiverse stack compression/expansion (ratio strategy)."""


@fractal_multiverse.command("generate")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path (full multiverse stack)")
@click.option("--width", type=int, required=True, help="Field width")
@click.option("--height", type=int, required=True, help="Field height")
@click.option("--layers", type=int, required=True, help="Number of layers in the multiverse stack")
@click.option("--octaves", type=int, default=4, show_default=True, help="fBm octaves per layer")
@click.option("--seed", type=int, default=None, help="Random seed (optional)")
@click.option("--preview", "preview_path", type=click.Path(dir_okay=False), default=None, help="Optional mosaic PNG path")
def multiverse_generate(output_path: str, width: int, height: int, layers: int, octaves: int, seed: Optional[int], preview_path: Optional[str]) -> None:
    """Generate a full multiverse stack bundle (JSON)."""
    try:
        from . import multiverse as mmod  # lazy import
        bundle = mmod.generate_full_stack(width=width, height=height, layers=layers, octaves=octaves, seed=seed)
        mmod.save_model(bundle, output_path)
        click.echo(f"Wrote model: {output_path}")
        if preview_path:
            mmod.save_mosaic_from_model(bundle, preview_path)
            click.echo(f"Wrote mosaic: {preview_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_multiverse.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full multiverse JSON path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output compressed JSON model path")
@click.option("--spatial-ratio", type=int, default=2, show_default=True, help="Downsample every Nth pixel per axis")
@click.option("--layer-ratio", type=int, default=1, show_default=True, help="Keep every Nth layer (1=keep all)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method to record in model")
def multiverse_compress(input_path: str, model_path: str, spatial_ratio: int, layer_ratio: int, method: str) -> None:
    """Compress a full multiverse stack into a ratio model (educational)."""
    try:
        from . import multiverse as mmod  # lazy import
        full = mmod.load_model(input_path)
        cfg = mmod.MultiverseConfig(strategy="ratio", spatial_ratio=spatial_ratio, layer_ratio=layer_ratio, method=method.lower())
        bundle = mmod.compress_stack(full, cfg)
        mmod.save_model(bundle, model_path)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_multiverse.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input compressed JSON model path (or full)")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output full multiverse JSON path")
@click.option("--width", type=int, default=None, help="Target width (optional; must specify width, height, layers together)")
@click.option("--height", type=int, default=None, help="Target height (optional; must specify width, height, layers together)")
@click.option("--layers", type=int, default=None, help="Target layers (optional; must specify width, height, layers together)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default=None, help="Override expansion method recorded in model")
@click.option("--preview", "preview_path", type=click.Path(dir_okay=False), default=None, help="Optional mosaic PNG path")
def multiverse_expand(model_path: str, output_path: str, width: Optional[int], height: Optional[int], layers: Optional[int], method: Optional[str], preview_path: Optional[str]) -> None:
    """Expand a compressed multiverse bundle back to a full bundle."""
    try:
        from . import multiverse as mmod  # lazy import
        bundle = mmod.load_model(model_path)
        if any(v is not None for v in (width, height, layers)) and not all(v is not None for v in (width, height, layers)):
            raise click.UsageError("Specify all of --width, --height, and --layers together, or none")
        target_size = (int(width), int(height), int(layers)) if (width is not None and height is not None and layers is not None) else None
        full = mmod.expand_stack(bundle, target_size=target_size, method=(method.lower() if method else None))
        mmod.save_model(full, output_path)
        click.echo(f"Wrote model: {output_path}")
        if preview_path:
            mmod.save_mosaic_from_model(full, preview_path)
            click.echo(f"Wrote mosaic: {preview_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_multiverse.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full multiverse JSON path")
@click.option("--recon-output", "recon_output", type=click.Path(dir_okay=False), required=True, help="Output full multiverse JSON path (reconstructed)")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional path to save compressed model JSON")
@click.option("--spatial-ratio", type=int, default=2, show_default=True, help="Downsample every Nth pixel per axis")
@click.option("--layer-ratio", type=int, default=1, show_default=True, help="Keep every Nth layer (1=keep all)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method")
@click.option("--compare", "compare_path", type=click.Path(dir_okay=False), default=None, help="Optional side-by-side mosaic compare image path")
@click.option("--analyze", "--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (MSE/RMSE/PSNR + spectral)")
def multiverse_engine(input_path: str, recon_output: str, model_path: Optional[str], spatial_ratio: int, layer_ratio: int, method: str, compare_path: Optional[str], analyze_output: Optional[str]) -> None:
    """Compress + expand a multiverse; optionally save compressed model, metrics, and compare mosaic."""
    try:
        from . import multiverse as mmod  # lazy import
        full_in = mmod.load_model(input_path)
        cfg = mmod.MultiverseConfig(strategy="ratio", spatial_ratio=spatial_ratio, layer_ratio=layer_ratio, method=method.lower())
        comp = mmod.compress_stack(full_in, cfg)
        if model_path:
            mmod.save_model(comp, model_path)
            click.echo(f"Wrote model: {model_path}")
        ow = int(full_in.get("width", 0))
        oh = int(full_in.get("height", 0))
        oL = int(full_in.get("layers", 0))
        full_out = mmod.expand_stack(comp, target_size=(ow, oh, oL), method=method.lower())
        mmod.save_model(full_out, recon_output)
        click.echo(f"Wrote model: {recon_output}")
        if compare_path:
            mmod.save_compare_mosaic(full_in, full_out, compare_path)
            click.echo(f"Wrote compare: {compare_path}")
        if analyze_output:
            mdf = mmod.metrics_from_paths(input_path, recon_output)
            mdf.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_multiverse.command("preview")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input multiverse JSON path (full or compressed)")
@click.option("--output", "image_path", type=click.Path(dir_okay=False), required=True, help="Output mosaic PNG path")
@click.option("--cmap", type=str, default="viridis", show_default=True, help="Matplotlib colormap name")
def multiverse_preview(model_path: str, image_path: str, cmap: str) -> None:
    """Export a mosaic image from a multiverse model (full or compressed)."""
    try:
        from . import multiverse as mmod  # lazy import
        bundle = mmod.load_model(model_path)
        mmod.save_mosaic_from_model(bundle, image_path, cmap=cmap)
        click.echo(f"Wrote mosaic: {image_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cmd.group("omniverse")
def fractal_omniverse() -> None:
    """Omniverse grid compression/expansion (ratio strategy)."""


@fractal_omniverse.command("generate")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path (full omniverse grid)")
@click.option("--width", type=int, required=True, help="Field width")
@click.option("--height", type=int, required=True, help="Field height")
@click.option("--layers", type=int, required=True, help="Number of layers per universe")
@click.option("--universes", type=int, required=True, help="Number of universes")
@click.option("--octaves", type=int, default=4, show_default=True, help="fBm octaves per layer")
@click.option("--seed", type=int, default=None, help="Random seed (optional)")
@click.option("--preview", "preview_path", type=click.Path(dir_okay=False), default=None, help="Optional mosaic PNG path")
def omniverse_generate(output_path: str, width: int, height: int, layers: int, universes: int, octaves: int, seed: Optional[int], preview_path: Optional[str]) -> None:
    """Generate a full omniverse grid bundle (JSON)."""
    try:
        from . import omniverse as omod  # lazy import
        bundle = omod.generate_full_grid(width=width, height=height, layers=layers, universes=universes, octaves=octaves, seed=seed)
        omod.save_model(bundle, output_path)
        click.echo(f"Wrote model: {output_path}")
        if preview_path:
            omod.save_mosaic_from_model(bundle, preview_path)
            click.echo(f"Wrote mosaic: {preview_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_omniverse.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full omniverse JSON path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output compressed JSON model path")
@click.option("--spatial-ratio", type=int, default=2, show_default=True, help="Downsample every Nth pixel per axis")
@click.option("--layer-ratio", type=int, default=1, show_default=True, help="Keep every Nth layer (1=keep all)")
@click.option("--universe-ratio", type=int, default=1, show_default=True, help="Keep every Nth universe (1=keep all)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method to record in model")
def omniverse_compress(input_path: str, model_path: str, spatial_ratio: int, layer_ratio: int, universe_ratio: int, method: str) -> None:
    """Compress a full omniverse grid into a ratio model (educational)."""
    try:
        from . import omniverse as omod  # lazy import
        full = omod.load_model(input_path)
        cfg = omod.OmniverseConfig(strategy="ratio", spatial_ratio=spatial_ratio, layer_ratio=layer_ratio, universe_ratio=universe_ratio, method=method.lower())
        bundle = omod.compress_grid(full, cfg)
        omod.save_model(bundle, model_path)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_omniverse.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input compressed JSON model path (or full)")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output full omniverse JSON path")
@click.option("--width", type=int, default=None, help="Target width (optional; must specify width, height, layers, universes together)")
@click.option("--height", type=int, default=None, help="Target height (optional; must specify width, height, layers, universes together)")
@click.option("--layers", type=int, default=None, help="Target layers (optional; must specify width, height, layers, universes together)")
@click.option("--universes", type=int, default=None, help="Target universes (optional; must specify width, height, layers, universes together)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default=None, help="Override expansion method recorded in model")
@click.option("--preview", "preview_path", type=click.Path(dir_okay=False), default=None, help="Optional mosaic PNG path")
def omniverse_expand(model_path: str, output_path: str, width: Optional[int], height: Optional[int], layers: Optional[int], universes: Optional[int], method: Optional[str], preview_path: Optional[str]) -> None:
    """Expand a compressed omniverse bundle back to a full bundle."""
    try:
        from . import omniverse as omod  # lazy import
        bundle = omod.load_model(model_path)
        dims = (width, height, layers, universes)
        if any(v is not None for v in dims) and not all(v is not None for v in dims):
            raise click.UsageError("Specify all of --width, --height, --layers, and --universes together, or none")
        target_size = (int(width), int(height), int(layers), int(universes)) if all(v is not None for v in dims) else None
        full = omod.expand_grid(bundle, target_size=target_size, method=(method.lower() if method else None))
        omod.save_model(full, output_path)
        click.echo(f"Wrote model: {output_path}")
        if preview_path:
            omod.save_mosaic_from_model(full, preview_path)
            click.echo(f"Wrote mosaic: {preview_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_omniverse.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full omniverse JSON path")
@click.option("--recon-output", "recon_output", type=click.Path(dir_okay=False), required=True, help="Output full omniverse JSON path (reconstructed)")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional path to save compressed model JSON")
@click.option("--spatial-ratio", type=int, default=2, show_default=True, help="Downsample every Nth pixel per axis")
@click.option("--layer-ratio", type=int, default=1, show_default=True, help="Keep every Nth layer (1=keep all)")
@click.option("--universe-ratio", type=int, default=1, show_default=True, help="Keep every Nth universe (1=keep all)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method")
@click.option("--compare", "compare_path", type=click.Path(dir_okay=False), default=None, help="Optional side-by-side mosaic compare image path")
@click.option("--analyze", "--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (MSE/RMSE/PSNR + spectral)")
def omniverse_engine(input_path: str, recon_output: str, model_path: Optional[str], spatial_ratio: int, layer_ratio: int, universe_ratio: int, method: str, compare_path: Optional[str], analyze_output: Optional[str]) -> None:
    """Compress + expand an omniverse; optionally save compressed model, metrics, and compare mosaic."""
    try:
        from . import omniverse as omod  # lazy import
        full_in = omod.load_model(input_path)
        cfg = omod.OmniverseConfig(strategy="ratio", spatial_ratio=spatial_ratio, layer_ratio=layer_ratio, universe_ratio=universe_ratio, method=method.lower())
        comp = omod.compress_grid(full_in, cfg)
        if model_path:
            omod.save_model(comp, model_path)
            click.echo(f"Wrote model: {model_path}")
        ow = int(full_in.get("width", 0))
        oh = int(full_in.get("height", 0))
        oL = int(full_in.get("layers", 0))
        oU = int(full_in.get("universes", 0))
        full_out = omod.expand_grid(comp, target_size=(ow, oh, oL, oU), method=method.lower())
        omod.save_model(full_out, recon_output)
        click.echo(f"Wrote model: {recon_output}")
        if compare_path:
            omod.save_compare_mosaic(full_in, full_out, compare_path)
            click.echo(f"Wrote compare: {compare_path}")
        if analyze_output:
            mdf = omod.metrics_from_paths(input_path, recon_output)
            mdf.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_omniverse.command("preview")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input omniverse JSON path (full or compressed)")
@click.option("--output", "image_path", type=click.Path(dir_okay=False), required=True, help="Output mosaic PNG path")
@click.option("--cmap", type=str, default="viridis", show_default=True, help="Matplotlib colormap name")
def omniverse_preview(model_path: str, image_path: str, cmap: str) -> None:
    """Export a mosaic image from an omniverse model (full or compressed)."""
    try:
        from . import omniverse as omod  # lazy import
        bundle = omod.load_model(model_path)
        omod.save_mosaic_from_model(bundle, image_path, cmap=cmap)
        click.echo(f"Wrote mosaic: {image_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_cmd.group("neuro")
def fractal_neuro() -> None:
    """Neuro network compression/expansion (ratio strategy)."""


@fractal_neuro.command("generate")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path (full neuro network)")
@click.option("--nodes", type=int, required=True, help="Number of neurons (nodes)")
@click.option("--model", type=click.Choice(["ws", "ba"], case_sensitive=False), default="ws", show_default=True, help="Graph generator model: Watts-Strogatz (ws) or Barabasi-Albert (ba)")
@click.option("--ws-k", type=int, default=10, show_default=True, help="WS: each node connected to k nearest neighbors in ring topology")
@click.option("--ws-p", type=float, default=0.1, show_default=True, help="WS: rewiring probability")
@click.option("--ba-m", type=int, default=3, show_default=True, help="BA: edges to attach from a new node to existing nodes")
@click.option("--seed", type=int, default=None, help="Random seed (optional)")
@click.option("--state-init", type=click.Choice(["random", "zeros"], case_sensitive=False), default="random", show_default=True, help="Initial neuron state initialization")
@click.option("--preview", "preview_path", type=click.Path(dir_okay=False), default=None, help="Optional adjacency PNG path")
def neuro_generate(output_path: str, nodes: int, model: str, ws_k: int, ws_p: float, ba_m: int, seed: Optional[int], state_init: str, preview_path: Optional[str]) -> None:
    """Generate a full neuro network bundle (JSON)."""
    try:
        from . import neuro as nmod  # lazy import
        bundle = nmod.generate_full_network(
            nodes=nodes,
            model=model.lower(),
            ws_k=ws_k,
            ws_p=ws_p,
            ba_m=ba_m,
            seed=seed,
            state_init=state_init.lower(),
        )
        nmod.save_model(bundle, output_path)
        click.echo(f"Wrote model: {output_path}")
        if preview_path:
            nmod.save_adjacency_image(bundle, preview_path)
            click.echo(f"Wrote adjacency: {preview_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_neuro.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full neuro JSON path")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output compressed JSON model path")
@click.option("--ratio", type=int, default=4, show_default=True, help="Keep every Nth neuron")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method to record in model")
def neuro_compress(input_path: str, model_path: str, ratio: int, method: str) -> None:
    """Compress a full neuro network into a compact ratio model (educational)."""
    try:
        from . import neuro as nmod  # lazy import
        full = nmod.load_model(input_path)
        cfg = nmod.NeuroConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = nmod.compress_network(full, cfg)
        nmod.save_model(bundle, model_path)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_neuro.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input neuro JSON path (compressed or full)")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output full neuro JSON path")
@click.option("--nodes", type=int, default=None, help="Target number of neurons (optional)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default=None, help="Override expansion method recorded in model")
@click.option("--seed", type=int, default=None, help="Random seed for stochastic topology regeneration (optional)")
@click.option("--preview", "preview_path", type=click.Path(dir_okay=False), default=None, help="Optional adjacency PNG path")
def neuro_expand(model_path: str, output_path: str, nodes: Optional[int], method: Optional[str], seed: Optional[int], preview_path: Optional[str]) -> None:
    """Expand a neuro model (compressed or full) to a full network bundle."""
    try:
        from . import neuro as nmod  # lazy import
        bundle = nmod.load_model(model_path)
        full = nmod.expand_network(bundle, target_nodes=nodes, method=(method.lower() if method else None), seed=seed)
        nmod.save_model(full, output_path)
        click.echo(f"Wrote model: {output_path}")
        if preview_path:
            nmod.save_adjacency_image(full, preview_path)
            click.echo(f"Wrote adjacency: {preview_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_neuro.command("simulate")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input neuro network path (PHI JSON, CSV edgelist, NPY/NPZ adjacency, simple JSON)")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output CSV path of state trajectories")
@click.option("--steps", type=int, default=100, show_default=True, help="Simulation steps (outputs steps+1 rows including initial state)")
@click.option("--dt", type=float, default=0.1, show_default=True, help="Time step size")
@click.option("--leak", type=float, default=0.1, show_default=True, help="Leak term coefficient")
@click.option("--input-drive", type=float, default=0.0, show_default=True, help="Fallback scalar external input drive added to all neurons")
@click.option("--drive-signal", type=click.Path(exists=True, dir_okay=False), default=None, help="Optional 1D drive signal path (audio/csv/npy/npz/json/image/video)")
@click.option("--drive-signal-normalize/--no-drive-signal-normalize", default=True, show_default=True, help="Normalize 1D drive signal before scaling")
@click.option("--drive-signal-scale", type=float, default=1.0, show_default=True, help="Scale factor for 1D drive signal")
@click.option("--pulse", is_flag=True, default=False, show_default=True, help="Generate a periodic pulse drive if no drive signal is provided")
@click.option("--pulse-period", type=int, default=100, show_default=True, help="Pulse period (steps)")
@click.option("--pulse-width", type=int, default=10, show_default=True, help="Pulse width (steps)")
@click.option("--pulse-amplitude", type=float, default=1.0, show_default=True, help="Pulse amplitude")
@click.option("--pulse-kind", type=click.Choice(["rect", "tri", "sine"], case_sensitive=False), default="rect", show_default=True, help="Pulse kind")
@click.option("--drive-2d", "drive_2d_path", type=click.Path(exists=True, dir_okay=False), default=None, help="Optional 2D drive array path (time x nodes) [csv/npy/npz/json]")
@click.option("--state-init", type=click.Choice(["random", "zeros"], case_sensitive=False), default="random", show_default=True, help="Initial neuron state if network is not a PHI JSON bundle")
@click.option("--net-seed", type=int, default=None, help="Random seed for network state init if needed")
@click.option("--noise-std", type=float, default=0.0, show_default=True, help="Gaussian noise std added to input drive")
@click.option("--seed", type=int, default=None, help="Random seed for input noise (optional)")
def neuro_simulate(
    model_path: str,
    output_path: str,
    steps: int,
    dt: float,
    leak: float,
    input_drive: float,
    drive_signal: Optional[str],
    drive_signal_normalize: bool,
    drive_signal_scale: float,
    pulse: bool,
    pulse_period: int,
    pulse_width: int,
    pulse_amplitude: float,
    pulse_kind: str,
    drive_2d_path: Optional[str],
    state_init: str,
    net_seed: Optional[int],
    noise_std: float,
    seed: Optional[int],
) -> None:
    """Run simple rate-based neuron simulation on a network and write CSV.

    Supports multi-format network loading and time-varying input drives.
    """
    try:
        from . import neuro as nmod  # lazy import
        import numpy as np  # lazy inside command
        import pandas as pd  # lazy inside command

        # Load network (PHI JSON full/ratio, CSV edgelist, NPY/NPZ adjacency, simple JSON)
        bundle = nmod.load_network_any(model_path, state_init=state_init.lower(), seed=net_seed)

        # Determine input drive
        drive_value: float | np.ndarray

        if drive_2d_path is not None:
            # Load a 2D array (time x nodes)
            def _load_drive_2d(path: str) -> np.ndarray:
                ext = os.path.splitext(path)[1].lower()
                if ext == ".npy":
                    arr = np.load(path)
                elif ext == ".npz":
                    with np.load(path) as data:
                        arr = None
                        for k in ("arr", "drive", "x", "data"):
                            if k in data:
                                arr = data[k]
                                break
                        if arr is None:
                            arr = list(data.values())[0]
                elif ext == ".csv":
                    df2 = pd.read_csv(path)
                    num = df2.select_dtypes(include=[np.number])
                    if num.shape[1] == 0:
                        raise click.UsageError("2D drive CSV must contain numeric columns")
                    arr = num.to_numpy(dtype=np.float32)
                elif ext == ".json":
                    with open(path, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if isinstance(obj, dict):
                        arr_like = None
                        for k in ("drive", "array", "data", "x"):
                            if k in obj:
                                arr_like = obj[k]
                                break
                        if arr_like is None:
                            raise click.UsageError("2D drive JSON must be a list of lists or contain 'drive'/'array'/'data'/'x'")
                        arr = np.asarray(arr_like, dtype=np.float32)
                    elif isinstance(obj, list):
                        arr = np.asarray(obj, dtype=np.float32)
                    else:
                        raise click.UsageError("Unsupported JSON structure for 2D drive")
                else:
                    raise click.UsageError("Unsupported 2D drive file type: {}".format(ext))
                arr = np.asarray(arr)
                if arr.ndim != 2:
                    raise click.UsageError("2D drive array must be 2D (time x nodes)")
                return arr.astype(np.float32, copy=False)

            drive_value = _load_drive_2d(drive_2d_path)
        elif drive_signal is not None:
            sig = nmod.load_signal_any(drive_signal, target_length=steps, normalize=drive_signal_normalize)
            drive_value = (sig.astype(np.float32, copy=False) * float(drive_signal_scale))
        elif pulse:
            drive_value = nmod.make_pulse_signal(
                steps=steps,
                period=pulse_period,
                width=pulse_width,
                amplitude=pulse_amplitude,
                kind=pulse_kind.lower(),
            )
        else:
            drive_value = float(input_drive)

        # Run simulation
        traj = nmod.simulate_states(
            bundle,
            steps=steps,
            dt=dt,
            leak=leak,
            input_drive=drive_value,
            noise_std=noise_std,
            seed=seed,
        )

        # Save CSV
        t = np.arange(traj.shape[0], dtype=np.int32)
        cols = ["t"] + [f"x{i}" for i in range(traj.shape[1])]
        df = pd.DataFrame(np.column_stack([t, traj]), columns=cols)
        df.to_csv(output_path, index=False)
        click.echo(f"Wrote CSV: {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_neuro.command("preview")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input neuro JSON path (full or compressed)")
@click.option("--output", "image_path", type=click.Path(dir_okay=False), required=True, help="Output adjacency PNG path")
@click.option("--cmap", type=str, default="viridis", show_default=True, help="Matplotlib colormap name")
def neuro_preview(model_path: str, image_path: str, cmap: str) -> None:
    """Export an adjacency matrix image from a neuro model (full or compressed)."""
    try:
        from . import neuro as nmod  # lazy import
        bundle = nmod.load_model(model_path)
        nmod.save_adjacency_image(bundle, image_path, cmap=cmap)
        click.echo(f"Wrote image: {image_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_neuro.command("analyze")
@click.option("--a", "a_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full neuro JSON path A")
@click.option("--b", "b_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full neuro JSON path B")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output metrics CSV path")
def neuro_analyze(a_path: str, b_path: str, output_path: str) -> None:
    """Compute metrics between two full neuro bundles and write CSV."""
    try:
        from . import neuro as nmod  # lazy import
        mdf = nmod.metrics_from_paths(a_path, b_path)
        mdf.to_csv(output_path, index=False)
        click.echo(f"Wrote analysis: {output_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_neuro.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input full neuro JSON path")
@click.option("--recon-output", "recon_output", type=click.Path(dir_okay=False), required=True, help="Output full neuro JSON path (reconstructed)")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional path to save compressed model JSON")
@click.option("--ratio", type=int, default=4, show_default=True, help="Keep every Nth neuron")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method")
@click.option("--nodes", type=int, default=None, help="Target number of neurons for reconstruction (optional; defaults to input nodes)")
@click.option("--seed", type=int, default=None, help="Random seed for topology regeneration (optional)")
@click.option("--preview", "preview_path", type=click.Path(dir_okay=False), default=None, help="Optional adjacency PNG of reconstruction")
@click.option("--analyze", "--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (state MSE and degree-hist L1)")
def neuro_engine(input_path: str, recon_output: str, model_path: Optional[str], ratio: int, method: str, nodes: Optional[int], seed: Optional[int], preview_path: Optional[str], analyze_output: Optional[str]) -> None:
    """Compress + expand a neuro network; optionally save model, metrics, and adjacency preview."""
    try:
        from . import neuro as nmod  # lazy import
        full_in = nmod.load_model(input_path)
        cfg = nmod.NeuroConfig(strategy="ratio", ratio=ratio, method=method.lower())
        comp = nmod.compress_network(full_in, cfg)
        if model_path:
            nmod.save_model(comp, model_path)
            click.echo(f"Wrote model: {model_path}")
        target_nodes = int(nodes) if nodes is not None else int(full_in.get("nodes", 0))
        full_out = nmod.expand_network(comp, target_nodes=target_nodes, method=method.lower(), seed=seed)
        nmod.save_model(full_out, recon_output)
        click.echo(f"Wrote model: {recon_output}")
        if preview_path:
            nmod.save_adjacency_image(full_out, preview_path)
            click.echo(f"Wrote adjacency: {preview_path}")
        if analyze_output:
            import pandas as pd  # lazy inside command
            mdf = nmod.metrics_from_paths(input_path, recon_output)
            mdf.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


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
@click.option("--analyze", "--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (MSE/RMSE/PSNR over sampled frames)")
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
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output point cloud path (.ply/.npz/.npy)")
@click.option("--points", type=int, default=20000, show_default=True, help="Number of points to generate")
@click.option("--scale", type=float, default=1.0, show_default=True, help="Model scale")
@click.option("--seed", type=int, default=None, help="Random seed (optional)")
@click.option("--fractal", type=click.Choice(["sierpinski", "menger", "mandelbulb"], case_sensitive=False), default="sierpinski", show_default=True, help="Fractal type to generate")
@click.option("--level", type=int, default=3, show_default=True, help="Menger sponge level (for fractal=menger)")
@click.option("--power", type=int, default=8, show_default=True, help="Mandelbulb power (for fractal=mandelbulb)")
@click.option("--c", nargs=3, type=float, default=(0.2, 0.35, 0.0), show_default=True, help="Mandelbulb Julia constant c=(cx cy cz)")
@click.option("--bounds", type=float, default=1.5, show_default=True, help="Mandelbulb sampling bounds (for fractal=mandelbulb)")
@click.option("--max-iter", type=int, default=20, show_default=True, help="Mandelbulb max iterations (for fractal=mandelbulb)")
@click.option("--bail", type=float, default=4.0, show_default=True, help="Mandelbulb bailout radius (for fractal=mandelbulb)")
@click.option("--preview", type=click.Path(dir_okay=False), default=None, help="Optional 2D projection PNG path")
@click.option("--axis", type=click.Choice(["x", "y", "z"], case_sensitive=False), default="z", show_default=True, help="Projection axis for preview")
@click.option("--height", type=int, default=400, show_default=True, help="Preview image height")
@click.option("--plot3d", type=click.Path(dir_okay=False), default=None, help="Optional Matplotlib 3D scatter PNG path")
@click.option("--plot3d-show/--no-plot3d-show", default=False, show_default=True, help="Show interactive 3D plot (requires matplotlib)")
@click.option("--point-size", type=float, default=1.0, show_default=True, help="Matplotlib scatter point size")
@click.option("--elev", type=float, default=20.0, show_default=True, help="Matplotlib 3D elevation")
@click.option("--azim", type=float, default=30.0, show_default=True, help="Matplotlib 3D azimuth")
def three_generate(output_path: str, points: int, scale: float, seed: Optional[int], fractal: str, level: int, power: int, c: Tuple[float, float, float], bounds: float, max_iter: int, bail: float, preview: Optional[str], axis: str, height: int, plot3d: Optional[str], plot3d_show: bool, point_size: float, elev: float, azim: float) -> None:
    """Generate a 3D fractal point cloud and save (optionally preview/plot)."""
    try:
        from . import three as three_mod  # lazy import
        f = fractal.lower()
        if f == "sierpinski":
            pts = three_mod.generate_sierpinski_tetrahedron(n_points=points, scale=scale, seed=seed)
        elif f == "menger":
            pts = three_mod.generate_menger_sponge(n_points=points, level=level, scale=scale, seed=seed)
        elif f == "mandelbulb":
            pts = three_mod.generate_mandelbulb_julia(
                n_points=points,
                power=power,
                c=(float(c[0]), float(c[1]), float(c[2])),
                bounds=bounds,
                max_iter=max_iter,
                bail=bail,
                scale=scale,
                seed=seed,
            )
        else:
            raise click.UsageError(f"Unknown fractal: {fractal}")
        three_mod.save_point_cloud(pts, output_path)
        click.echo(f"Wrote point cloud: {output_path}")
        if preview:
            three_mod.save_projection_from_ply(output_path, preview, axis=axis.lower(), height=height)
            click.echo(f"Wrote preview: {preview}")
        if plot3d or plot3d_show:
            three_mod.plot_point_cloud_matplotlib(pts, save_path=plot3d, show=plot3d_show, size=point_size, elev=elev, azim=azim)
            if plot3d:
                click.echo(f"Wrote 3D plot: {plot3d}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_three.command("compress")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input point cloud path (.ply/.npz/.npy)")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), required=True, help="Output JSON model path")
@click.option("--ratio", type=int, default=4, show_default=True, help="Keep every Nth point")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method to record in model")
def three_compress(input_path: str, model_path: str, ratio: int, method: str) -> None:
    """Compress a point cloud by point decimation into a JSON model (educational)."""
    try:
        from . import three as three_mod
        pts = three_mod.load_point_cloud(input_path)
        cfg = three_mod.ThreeConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = three_mod.compress_point_cloud(pts, cfg)
        three_mod.save_model(bundle, model_path)
        click.echo(f"Wrote model: {model_path}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_three.command("expand")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input JSON model path")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output point cloud path (.ply/.npz/.npy)")
@click.option("--points", type=int, default=None, help="Target number of output points (optional)")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default=None, help="Override expansion method recorded in model")
@click.option("--preview", type=click.Path(dir_okay=False), default=None, help="Optional 2D projection PNG path")
@click.option("--axis", type=click.Choice(["x", "y", "z"], case_sensitive=False), default="z", show_default=True, help="Projection axis for preview")
@click.option("--height", type=int, default=400, show_default=True, help="Preview image height")
@click.option("--plot3d", type=click.Path(dir_okay=False), default=None, help="Optional Matplotlib 3D scatter PNG path")
@click.option("--plot3d-show/--no-plot3d-show", default=False, show_default=True, help="Show interactive 3D plot (requires matplotlib)")
@click.option("--point-size", type=float, default=1.0, show_default=True, help="Matplotlib scatter point size")
@click.option("--elev", type=float, default=20.0, show_default=True, help="Matplotlib 3D elevation")
@click.option("--azim", type=float, default=30.0, show_default=True, help="Matplotlib 3D azimuth")
def three_expand(model_path: str, output_path: str, points: Optional[int], method: Optional[str], preview: Optional[str], axis: str, height: int, plot3d: Optional[str], plot3d_show: bool, point_size: float, elev: float, azim: float) -> None:
    """Expand a 3D model back into a point cloud (PLY)."""
    try:
        from . import three as three_mod
        bundle = three_mod.load_model(model_path)
        recon = three_mod.expand_point_cloud(bundle, target_points=points, method=(method.lower() if method else None))
        three_mod.save_point_cloud(recon, output_path)
        click.echo(f"Wrote point cloud: {output_path}")
        if preview:
            three_mod.save_projection_from_ply(output_path, preview, axis=axis.lower(), height=height)
            click.echo(f"Wrote preview: {preview}")
        if plot3d or plot3d_show:
            three_mod.plot_point_cloud_matplotlib(recon, save_path=plot3d, show=plot3d_show, size=point_size, elev=elev, azim=azim)
            if plot3d:
                click.echo(f"Wrote 3D plot: {plot3d}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@fractal_three.command("engine")
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False), required=True, help="Input point cloud path (.ply/.npz/.npy)")
@click.option("--output", "output_path", type=click.Path(dir_okay=False), required=True, help="Output reconstructed point cloud path (.ply/.npz/.npy)")
@click.option("--model", "model_path", type=click.Path(dir_okay=False), default=None, help="Optional model JSON save path")
@click.option("--ratio", type=int, default=4, show_default=True, help="Keep every Nth point")
@click.option("--method", type=click.Choice(["interp", "nearest"], case_sensitive=False), default="interp", show_default=True, help="Expansion method")
@click.option("--points", type=int, default=None, help="Target number of output points (optional)")
@click.option("--compare", "compare_path", type=click.Path(dir_okay=False), default=None, help="Optional side-by-side projection compare image")
@click.option("--axis", type=click.Choice(["x", "y", "z"], case_sensitive=False), default="z", show_default=True, help="Projection axis for compare image")
@click.option("--analyze", "--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (approx. symmetric Chamfer distance)")
@click.option("--sample-points", type=int, default=2000, show_default=True, help="Number of points to sample for metrics")
@click.option("--nn-method", type=click.Choice(["auto", "kd", "sklearn", "brute"], case_sensitive=False), default="auto", show_default=True, help="Nearest-neighbor method for metrics")
@click.option("--plot3d", type=click.Path(dir_okay=False), default=None, help="Optional Matplotlib 3D scatter PNG path (of reconstruction)")
@click.option("--plot3d-show/--no-plot3d-show", default=False, show_default=True, help="Show interactive 3D plot (requires matplotlib)")
@click.option("--point-size", type=float, default=1.0, show_default=True, help="Matplotlib scatter point size")
@click.option("--elev", type=float, default=20.0, show_default=True, help="Matplotlib 3D elevation")
@click.option("--azim", type=float, default=30.0, show_default=True, help="Matplotlib 3D azimuth")
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
    nn_method: str,
    plot3d: Optional[str],
    plot3d_show: bool,
    point_size: float,
    elev: float,
    azim: float,
) -> None:
    """Compress + expand a point cloud; optionally save model, compare, and metrics."""
    try:
        from . import three as three_mod
        pts = three_mod.load_point_cloud(input_path)
        cfg = three_mod.ThreeConfig(strategy="ratio", ratio=ratio, method=method.lower())
        bundle = three_mod.compress_point_cloud(pts, cfg)
        if model_path:
            three_mod.save_model(bundle, model_path)
            click.echo(f"Wrote model: {model_path}")
        target = points if points is not None else int(bundle.get("orig_count", len(pts)))
        recon = three_mod.expand_point_cloud(bundle, target_points=target, method=method.lower())
        three_mod.save_point_cloud(recon, output_path)
        click.echo(f"Wrote point cloud: {output_path}")
        if compare_path:
            three_mod.save_compare_projection_image(input_path, output_path, compare_path, axis=axis.lower())
            click.echo(f"Wrote compare image: {compare_path}")
        if analyze_output:
            mdf = three_mod.metrics_from_paths(input_path, output_path, sample_points=sample_points, nn_method=nn_method)
            mdf.to_csv(analyze_output, index=False)
            click.echo(f"Wrote analysis: {analyze_output}")
        if plot3d or plot3d_show:
            three_mod.plot_point_cloud_matplotlib(recon, save_path=plot3d, show=plot3d_show, size=point_size, elev=elev, azim=azim)
            if plot3d:
                click.echo(f"Wrote 3D plot: {plot3d}")
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
@click.option("--analyze", "--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (MSE/RMSE/PSNR)")
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
@click.option("--analyze", "--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional metrics CSV (MSE/RMSE/PSNR)")
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
@click.option("--analyze", "--analyze-output", "analyze_output", type=click.Path(dir_okay=False), default=None, help="Optional path to save analysis metrics CSV")
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


@main.group(name="llm")
def llm_cmd() -> None:
    """LLM pipeline helpers (optional deps: transformers, datasets, peft).

    This group provides an interactive wizard to:
    - pick a base model by size and repo id
    - (optionally) download and cache tokenizer/weights locally
    - ingest a dataset (JSONL), perform a φ (61.8/38.2) train/val split if needed
    - tokenize to a ready-to-train format
    - suggest golden-heuristic hyperparameters
    - (optionally) kick off LoRA training
    """


@llm_cmd.command("wizard")
@click.option("--project", type=str, default=None, help="Project name used under models/, datasets/, runs/")
@click.option("--size", type=click.Choice(["125m", "350m", "1.3b", "1.4b", "2.7b", "6b"], case_sensitive=False), default=None, help="Model size hint")
@click.option("--base", type=str, default=None, help="Base model repo id, e.g. EleutherAI/gpt-neo-1.3B")
@click.option("--dataset", "datasets_in", type=str, multiple=True, help="Dataset path(s). File JSONL (golden-split) or dir with train/val.jsonl")
@click.option("--phi-mix/--no-phi-mix", default=True, show_default=True, help="When multiple datasets are provided, interleave with φ weights")
@click.option("--download-tokenizer/--no-download-tokenizer", default=True, show_default=True, help="Download tokenizer locally")
@click.option("--download-weights/--no-download-weights", default=False, show_default=True, help="Download base weights locally (large)")
@click.option("--tokenize/--no-tokenize", default=True, show_default=True, help="Tokenize dataset into ready/ directory")
@click.option("--train/--no-train", default=False, show_default=True, help="Start LoRA training automatically")
@click.option("--eval/--no-eval", default=False, show_default=True, help="Evaluate perplexity after training (or given adapter)")
@click.option("--adapter", type=str, default=None, help="Optional adapter dir to evaluate (defaults to runs/<project>/lora/final)")
@click.option("--max-length", type=int, default=1024, show_default=True, help="Tokenizer truncation length")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for golden split and mixing")
@click.option("--auto/--no-auto", default=False, show_default=True, help="Non-interactive mode; do not prompt, use flags and defaults")
# Hyperparam overrides (optional)
@click.option("--epochs", "epochs_override", type=int, default=None, help="Override suggested epochs")
@click.option("--lr", "lr_override", type=float, default=None, help="Override suggested learning rate")
@click.option("--warmup", "warmup_override", type=float, default=None, help="Override warmup ratio")
@click.option("--rank", "rank_override", type=int, default=None, help="Override LoRA rank")
@click.option("--alpha", "alpha_override", type=int, default=None, help="Override LoRA alpha")
@click.option("--dropout", "dropout_override", type=float, default=None, help="Override LoRA dropout")
@click.option("--per-device-batch", "per_device_batch_override", type=int, default=None, help="Override per-device batch size")
@click.option("--grad-accum", "grad_accum_override", type=int, default=None, help="Override gradient accumulation")
@click.option("--precision", "precision_override", type=click.Choice(["bf16", "fp16", "fp32"], case_sensitive=False), default=None, help="Precision override")
def llm_wizard(project: Optional[str], size: Optional[str], base: Optional[str], datasets_in: Tuple[str, ...], phi_mix: bool, download_tokenizer: bool, download_weights: bool, tokenize: bool, train: bool, eval: bool, adapter: Optional[str], max_length: int, seed: int, auto: bool, epochs_override: Optional[int], lr_override: Optional[float], warmup_override: Optional[float], rank_override: Optional[int], alpha_override: Optional[int], dropout_override: Optional[float], per_device_batch_override: Optional[int], grad_accum_override: Optional[int], precision_override: Optional[str]) -> None:
    """Interactive end-to-end setup for an LLM fine-tune run."""
    try:
        import pathlib
        import shutil
        import random

        # 1) Choose model size and repo
        model_candidates = {
            "125m": ["EleutherAI/gpt-neo-125M"],
            "350m": ["EleutherAI/gpt-neo-350M"],
            "1.3b": ["EleutherAI/gpt-neo-1.3B", "EleutherAI/pythia-1.4b"],
            "1.4b": ["EleutherAI/pythia-1.4b", "EleutherAI/gpt-neo-1.3B"],
            "2.7b": ["EleutherAI/gpt-neo-2.7B"],
            "6b": ["EleutherAI/gpt-j-6B"],
        }
        # Interactive selection unless base provided
        if base is None:
            size_lower = (size or (click.prompt(
                "Select model size",
                type=click.Choice(["125m", "350m", "1.3b", "1.4b", "2.7b", "6b"], case_sensitive=False),
                default="1.3b",
            ) if not auto else "1.3b")).lower()
            choices = model_candidates.get(size_lower, model_candidates["1.3b"])  # default fallback
            if auto:
                base = choices[0]
            else:
                base = click.prompt("Choose a base model", type=click.Choice(choices, case_sensitive=False), default=choices[0])
        base = str(base)

        # 2) Name the project
        if not project:
            default_project = base.split("/")[-1].lower().replace(".", "_").replace("-", "_")
            project = default_project if auto else click.prompt("Project name", default=default_project)

        # 3) Layout
        cwd = pathlib.Path(os.getcwd())
        models_dir = cwd / "models"
        proj_model_dir = models_dir / base.replace("/", "__")
        raw_dir = cwd / "datasets" / "raw" / project
        ready_dir = cwd / "datasets" / "ready" / project
        runs_dir = cwd / "runs" / project
        for d in [models_dir, proj_model_dir, raw_dir, ready_dir, runs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 4) Download artifacts (tokenizer + optional base weights)
        if download_tokenizer and (auto or click.confirm("Download tokenizer now?", default=True)):
            try:
                from transformers import AutoTokenizer
            except Exception:
                click.echo("Transformers not installed. Install: pip install transformers datasets peft", err=True)
            else:
                tok = AutoTokenizer.from_pretrained(base, use_fast=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                tok.save_pretrained(str(proj_model_dir / "tokenizer"))
                click.echo(f"Saved tokenizer to {proj_model_dir / 'tokenizer'}")

        if download_weights and (auto or click.confirm("Download FULL base weights now? (large)", default=False)):
            try:
                from transformers import AutoModelForCausalLM
            except Exception:
                click.echo("Transformers not installed. Install: pip install transformers", err=True)
            else:
                model = AutoModelForCausalLM.from_pretrained(base)
                model.save_pretrained(str(proj_model_dir / "base"))
                click.echo(f"Saved base weights to {proj_model_dir / 'base'}")

        # 5) Ingest dataset
        # 5) Ingest dataset(s)
        def _golden_split_lines(path: pathlib.Path, rnd: random.Random) -> Tuple[list[str], list[str]]:
            lines: list[str] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        lines.append(line.rstrip("\n"))
            if not lines:
                raise click.UsageError(f"Input JSONL is empty: {path}")
            rnd.shuffle(lines)
            split_idx = int(len(lines) * (1.0 / float(PHI)))  # ≈ 0.618
            return lines[:split_idx], lines[split_idx:]

        def _weighted_merge(list_of_lists: list[list[str]], weights: list[float], rnd: random.Random) -> list[str]:
            # Merge items from multiple lists approximately following given weights
            idxs = [0] * len(list_of_lists)
            result: list[str] = []
            active = [i for i, lst in enumerate(list_of_lists) if len(lst) > 0]
            while active:
                # Renormalize over remaining lists
                rem_weights = [weights[i] for i in active]
                s = sum(rem_weights)
                probs = [w / s for w in rem_weights]
                # Roulette selection
                r = rnd.random()
                cum = 0.0
                chosen_idx = active[-1]
                for j, pidx in enumerate(active):
                    cum += probs[j]
                    if r <= cum:
                        chosen_idx = pidx
                        break
                # Take next item if available
                ptr = idxs[chosen_idx]
                lst = list_of_lists[chosen_idx]
                if ptr < len(lst):
                    result.append(lst[ptr])
                    idxs[chosen_idx] = ptr + 1
                # Drop exhausted lists
                active = [i for i in active if idxs[i] < len(list_of_lists[i])]
            return result

        rnd = random.Random(int(seed))
        train_buckets: list[list[str]] = []
        val_buckets: list[list[str]] = []

        # Interactive single-path prompt if none provided
        ds_args = list(datasets_in)
        if not ds_args and not auto:
            ds_path = click.prompt(
                "Dataset path (JSONL file or directory with train/val.jsonl)",
                default="",
            ).strip()
            if ds_path:
                ds_args = [ds_path]

        for ds in ds_args:
            p = pathlib.Path(ds)
            if p.is_file():
                tr, va = _golden_split_lines(p, rnd)
                train_buckets.append(tr)
                val_buckets.append(va)
            elif p.is_dir():
                train_p = p / "train.jsonl"
                val_p = p / "val.jsonl"
                if not train_p.exists() and (p / "valid.jsonl").exists():
                    val_p = p / "valid.jsonl"
                if not train_p.exists() or not val_p.exists():
                    raise click.UsageError(f"Directory must contain train.jsonl and val.jsonl/valid.jsonl: {p}")
                # Read lines as-is
                def _read_lines(path: pathlib.Path) -> list[str]:
                    with path.open("r", encoding="utf-8") as f:
                        return [ln.rstrip("\n") for ln in f if ln.strip()]
                train_buckets.append(_read_lines(train_p))
                val_buckets.append(_read_lines(val_p))
            else:
                raise click.UsageError(f"Dataset path is neither file nor directory: {p}")

        # Write merged train/val if any provided
        if train_buckets or val_buckets:
            # Compute φ weights for buckets
            n = max(len(train_buckets), len(val_buckets))
            if n > 0:
                phi_weights = [ (INV_PHI ** (i+1)) for i in range(n) ]  # 0.618, 0.382, 0.236, ...
                # Normalize to 1
                s = sum(phi_weights)
                phi_weights = [w / s for w in phi_weights]
            else:
                phi_weights = []

            def _merge_or_concat(buckets: list[list[str]]) -> list[str]:
                if not buckets:
                    return []
                if phi_mix and len(buckets) > 1:
                    return _weighted_merge(buckets, phi_weights[:len(buckets)], rnd)
                # else simple concatenation with shuffle
                merged: list[str] = []
                for b in buckets:
                    merged.extend(b)
                rnd.shuffle(merged)
                return merged

            merged_train = _merge_or_concat(train_buckets)
            merged_val = _merge_or_concat(val_buckets)
            raw_dir.mkdir(parents=True, exist_ok=True)
            with (raw_dir / "train.jsonl").open("w", encoding="utf-8") as f:
                if merged_train:
                    f.write("\n".join(merged_train) + "\n")
            with (raw_dir / "val.jsonl").open("w", encoding="utf-8") as f:
                if merged_val:
                    f.write("\n".join(merged_val) + "\n")
            click.echo(f"Wrote merged train/val to {raw_dir}")
        else:
            click.echo("No dataset provided; skipping ingestion step")

        # 6) Tokenize to ready dataset
        if tokenize and (auto or click.confirm("Tokenize dataset now?", default=True)):
            try:
                from datasets import load_dataset
                from transformers import AutoTokenizer
            except Exception:
                click.echo("Install deps first: pip install transformers datasets", err=True)
            else:
                train_p = raw_dir / "train.jsonl"
                val_p = raw_dir / "val.jsonl"
                if not train_p.exists() or not val_p.exists():
                    raise click.UsageError("Expected train.jsonl and val.jsonl under datasets/raw/<project>/")
                tok = AutoTokenizer.from_pretrained(base, use_fast=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token

                raw = load_dataset("json", data_files={"train": str(train_p), "validation": str(val_p)})

                def build_text(example):
                    if "text" in example and example["text"]:
                        return {"text": example["text"]}
                    instruction = (example.get("instruction", "") or "").strip()
                    inp = (example.get("input", "") or "").strip()
                    out = (example.get("output", "") or "").strip()
                    text = f"Instruction: {instruction}\nInput: {inp}\nOutput: {out}\n"
                    return {"text": text}

                raw = raw.map(build_text)

                def tok_map(batch):
                    return tok(batch["text"], truncation=True, max_length=int(max_length))

                cols = raw["train"].column_names
                tokd = raw.map(tok_map, batched=True, remove_columns=cols)
                tokd.save_to_disk(str(ready_dir))
                click.echo(f"Saved tokenized dataset to {ready_dir}")

        # 7) Suggest hyperparameters (golden heuristics)
        vram_gb = 24 if auto else click.prompt("Approx GPU VRAM (GB)", type=int, default=24)
        # Simple heuristic knobs
        if vram_gb < 16:
            rank_val = 8
            ga_val = 16
            per_dev_val = 1
        elif vram_gb >= 48:
            rank_val = 32
            ga_val = 4
            per_dev_val = 1
        else:
            rank_val = 16
            ga_val = 8
            per_dev_val = 1
        suggestions = {
            "base": base,
            "dataset_ready": str(ready_dir),
            "output_dir": str(runs_dir / "lora"),
            "epochs": int(epochs_override) if epochs_override is not None else 3,
            "learning_rate": float(lr_override) if lr_override is not None else 2e-4,
            "warmup_ratio": float(warmup_override) if warmup_override is not None else 0.0618,  # φ^{-3}
            "weight_decay": 0.1,
            "per_device_batch": int(per_device_batch_override) if per_device_batch_override is not None else per_dev_val,
            "gradient_accumulation": int(grad_accum_override) if grad_accum_override is not None else ga_val,
            "lora_rank": int(rank_override) if rank_override is not None else rank_val,
            "lora_alpha": int(alpha_override) if alpha_override is not None else (2 * (int(rank_override) if rank_override is not None else rank_val)),
            "lora_dropout": float(dropout_override) if dropout_override is not None else 0.05,
            "eval_steps": 200,
            "save_steps": 200,
            "precision": (precision_override or "bf16"),
        }
        with (runs_dir / "suggested.json").open("w", encoding="utf-8") as f:
            json.dump(suggestions, f, indent=2)
        click.echo(f"Wrote suggestions to {runs_dir / 'suggested.json'}")

        # 8) Optional: start training now
        start_train = train if auto else (click.confirm("Start training now with these settings?", default=False))
        if start_train:
            try:
                from datasets import load_from_disk
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    Trainer,
                    TrainingArguments,
                    DataCollatorForLanguageModeling,
                )
                from peft import LoraConfig, get_peft_model, TaskType
            except Exception:
                click.echo("Install deps first: pip install transformers datasets peft", err=True)
            else:
                tok = AutoTokenizer.from_pretrained(base, use_fast=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                model = AutoModelForCausalLM.from_pretrained(base)
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=int(suggestions["lora_rank"]),
                    lora_alpha=int(suggestions["lora_alpha"]),
                    lora_dropout=float(suggestions["lora_dropout"]),
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "fc_in", "fc_out", "dense", "proj",
                    ],
                )
                model = get_peft_model(model, lora_cfg)

                ds = load_from_disk(str(ready_dir))
                collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
                targs = TrainingArguments(
                    output_dir=str(runs_dir / "lora"),
                    per_device_train_batch_size=int(suggestions["per_device_batch"]),
                    per_device_eval_batch_size=1,
                    gradient_accumulation_steps=int(suggestions["gradient_accumulation"]),
                    num_train_epochs=int(suggestions["epochs"]),
                    learning_rate=float(suggestions["learning_rate"]),
                    lr_scheduler_type="cosine",
                    warmup_ratio=float(suggestions["warmup_ratio"]),
                    weight_decay=float(suggestions["weight_decay"]),
                    logging_steps=25,
                    evaluation_strategy="steps",
                    eval_steps=int(suggestions["eval_steps"]),
                    save_steps=int(suggestions["save_steps"]),
                    save_total_limit=3,
                    bf16=(suggestions["precision"].lower() == "bf16"),
                    fp16=(suggestions["precision"].lower() == "fp16"),
                    report_to=["none"],
                )
                trainer = Trainer(
                    model=model,
                    args=targs,
                    train_dataset=ds["train"],
                    eval_dataset=ds.get("validation"),
                    data_collator=collator,
                )
                trainer.train()
                out_dir = runs_dir / "lora" / "final"
                trainer.save_model(str(out_dir))
                click.echo(f"Saved LoRA adapter to {out_dir}")

        # 9) Optional: evaluation
        do_eval = eval if auto else (click.confirm("Run evaluation (perplexity) now?", default=False))
        if do_eval:
            try:
                import math
                from datasets import load_from_disk
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    Trainer,
                    TrainingArguments,
                    DataCollatorForLanguageModeling,
                )
                from peft import PeftModel
            except Exception:
                click.echo("Install deps first: pip install transformers datasets peft", err=True)
            else:
                tok = AutoTokenizer.from_pretrained(base, use_fast=True)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                base_model = AutoModelForCausalLM.from_pretrained(base)
                adapter_dir = adapter or str(runs_dir / "lora" / "final")
                model = PeftModel.from_pretrained(base_model, adapter_dir)
                collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
                ds = load_from_disk(str(ready_dir))
                args = TrainingArguments(output_dir=str(runs_dir / "eval"), per_device_eval_batch_size=1)
                trainer = Trainer(model=model, args=args, eval_dataset=ds["validation"], data_collator=collator)
                metrics = trainer.evaluate()
                ppl = math.exp(metrics.get("eval_loss", float("nan")))
                click.echo(json.dumps({"perplexity": ppl, **metrics}, indent=2))

        click.echo("LLM wizard completed.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@main.group("neuro")
def neuro_cmd() -> None:
    """Neural tools including BCI simulation."""


@neuro_cmd.command("bci-sim")
@click.option("--steps", type=int, default=500, show_default=True, help="Number of closed-loop steps")
@click.option("--fs", type=float, default=256.0, show_default=True, help="Sampling rate (Hz)")
@click.option("--window-sec", type=float, default=1.0, show_default=True, help="Window duration per step (sec)")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed")
@click.option("--process-noise", type=float, default=0.02, show_default=True, help="Latent process noise")
@click.option("--drift", type=float, default=0.001, show_default=True, help="Latent slow drift magnitude")
@click.option("--snr-scale", type=float, default=0.6, show_default=True, help="How strongly latent modulates the gamma envelope")
@click.option("--noise-std", type=float, default=1.0, show_default=True, help="Observation noise standard deviation")
@click.option("--theta-hz", type=float, default=6.0, show_default=True, help="Theta oscillation frequency (Hz)")
@click.option("--gamma-hz", type=float, default=40.0, show_default=True, help="Gamma oscillation frequency (Hz)")
@click.option("--ctrl-effect", type=float, default=0.05, show_default=True, help="Control influence on latent")
@click.option("--base-lr", type=float, default=0.05, show_default=True, help="Decoder base learning rate")
@click.option("--base-gain", type=float, default=0.5, show_default=True, help="Controller base gain")
@click.option("--scheduler", type=click.Choice(["constant", "cosine", "cosine_phi", "linear", "step"], case_sensitive=False), default="cosine_phi", show_default=True, help="Scheduler type")
@click.option("--const-v", type=float, default=1.0, show_default=True, help="Constant scheduler value (if scheduler=constant)")
@click.option("--cos-period", type=int, default=200, show_default=True, help="Cosine period (steps)")
@click.option("--cos-min", type=float, default=0.2, show_default=True, help="Cosine min value")
@click.option("--cos-max", type=float, default=1.0, show_default=True, help="Cosine max value")
@click.option("--phi-T0", "phi_T0", type=int, default=200, show_default=True, help="Initial period for φ-restarts")
@click.option("--phi", "phi_val", type=float, default=1.618, show_default=True, help="Golden ratio factor for period growth")
@click.option("--phi-min", type=float, default=0.2, show_default=True, help="Min value for φ-restarts scheduler")
@click.option("--phi-max", type=float, default=1.0, show_default=True, help="Max value for φ-restarts scheduler")
@click.option("--lin-start", type=float, default=1.0, show_default=True, help="Linear start value (if scheduler=linear)")
@click.option("--lin-end", type=float, default=0.2, show_default=True, help="Linear end value (if scheduler=linear)")
@click.option("--lin-duration", type=int, default=500, show_default=True, help="Linear duration in steps (if scheduler=linear)")
@click.option("--step-initial", type=float, default=1.0, show_default=True, help="Initial value (if scheduler=step)")
@click.option("--step-gamma", type=float, default=0.5, show_default=True, help="Multiplicative decay factor per period (if scheduler=step)")
@click.option("--step-period", type=int, default=200, show_default=True, help="Period in steps for step schedule (if scheduler=step)")
@click.option("--save-features/--no-save-features", default=False, show_default=True, help="Save per-step feature vectors to 'bci_features.csv'")
@click.option("--save-windows/--no-save-windows", default=False, show_default=True, help="Save raw signal windows to 'bci_windows.npz'")
@click.option("--save-config/--no-save-config", default=True, show_default=True, help="Write 'bci_config.json' with metadata and feature names")
@click.option("--out-dir", type=click.Path(dir_okay=True, file_okay=False), default=None, help="Optional output directory to save logs")
def neuro_bci_sim(steps: int, fs: float, window_sec: float, seed: int, process_noise: float, drift: float, snr_scale: float, noise_std: float, ctrl_effect: float, theta_hz: float, gamma_hz: float, base_lr: float, base_gain: float, scheduler: str, const_v: float, cos_period: int, cos_min: float, cos_max: float, phi_T0: int, phi_val: float, phi_min: float, phi_max: float, lin_start: float, lin_end: float, lin_duration: int, step_initial: float, step_gamma: float, step_period: int, save_features: bool, save_windows: bool, save_config: bool, out_dir: Optional[str]) -> None:
    """Run a closed-loop BCI simulation and print summary metrics."""
    try:
        from .neuro import bci as bci_mod
    except Exception as e:
        click.echo(f"Error importing neuro.bci: {e}", err=True)
        sys.exit(1)

    # Build scheduler
    sch_type = scheduler.lower()
    if sch_type == "constant":
        sch = bci_mod.ConstantScheduler(v=float(const_v))
    elif sch_type == "cosine":
        sch = bci_mod.CosineScheduler(period=int(cos_period), min_v=float(cos_min), max_v=float(cos_max))
    elif sch_type == "linear":
        sch = bci_mod.LinearScheduler(start_v=float(lin_start), end_v=float(lin_end), duration=int(lin_duration))
    elif sch_type == "step":
        sch = bci_mod.StepScheduler(initial=float(step_initial), gamma=float(step_gamma), period=int(step_period))
    else:
        sch = bci_mod.CosineWithPhiRestarts(T0=int(phi_T0), phi=float(phi_val), min_v=float(phi_min), max_v=float(phi_max))

    # Config
    cfg = bci_mod.BCIConfig(
        fs=float(fs),
        window_sec=float(window_sec),
        steps=int(steps),
        seed=int(seed),
        process_noise=float(process_noise),
        drift=float(drift),
        ctrl_effect=float(ctrl_effect),
        base_lr=float(base_lr),
        base_gain=float(base_gain),
        noise_std=float(noise_std),
        snr_scale=float(snr_scale),
        theta_hz=float(theta_hz),
        gamma_hz=float(gamma_hz),
    )

    logs = bci_mod.simulate(
        cfg,
        scheduler=sch,
        out_dir=out_dir,
        save_features=bool(save_features),
        save_windows=bool(save_windows),
        save_config=bool(save_config),
    )
    summary = {k: float(v[0]) for k, v in logs.items() if k in ("mse", "mae", "ttc")}
    out = {"summary": summary, "out_dir": out_dir}
    # If out_dir provided, include which artifacts likely exist
    if out_dir:
        out["artifacts"] = {
            "timeseries_csv": "bci_timeseries.csv",
            "features_csv": "bci_features.csv" if save_features else None,
            "windows_npz": "bci_windows.npz" if save_windows else None,
            "config_json": "bci_config.json" if save_config else None,
            "summary_json": "bci_summary.json",
        }
    click.echo(json.dumps(out, indent=2))


@neuro_cmd.command("bci-sweep")
@click.option("--steps", type=int, default=300, show_default=True)
@click.option("--fs", type=float, default=256.0, show_default=True)
@click.option("--window-sec", type=float, default=1.0, show_default=True)
@click.option("--seeds", type=int, multiple=True, default=[1, 2], show_default=True)
@click.option("--process-noise", type=float, multiple=True, default=[0.02], show_default=True)
@click.option("--drift", type=float, multiple=True, default=[0.001], show_default=True)
@click.option("--snr-scale", type=float, multiple=True, default=[0.6], show_default=True)
@click.option("--noise-std", type=float, multiple=True, default=[1.0], show_default=True)
@click.option("--theta-hz", type=float, default=6.0, show_default=True)
@click.option("--gamma-hz", type=float, default=40.0, show_default=True)
@click.option("--base-lr", type=float, default=0.05, show_default=True)
@click.option("--base-gain", type=float, default=0.5, show_default=True)
@click.option("--ctrl-effect", type=float, default=0.05, show_default=True)
@click.option("--schedulers", type=click.Choice(["constant", "cosine", "cosine_phi", "linear", "step"], case_sensitive=False), multiple=True, default=["cosine_phi"], show_default=True)
@click.option("--const-v", type=float, default=1.0, show_default=True)
@click.option("--cos-period", type=int, default=200, show_default=True)
@click.option("--cos-min", type=float, default=0.2, show_default=True)
@click.option("--cos-max", type=float, default=1.0, show_default=True)
@click.option("--phi-T0", "phi_T0", type=int, default=200, show_default=True)
@click.option("--phi", "phi_val", type=float, default=1.618, show_default=True)
@click.option("--phi-min", type=float, default=0.2, show_default=True)
@click.option("--phi-max", type=float, default=1.0, show_default=True)
@click.option("--lin-start", type=float, default=1.0, show_default=True)
@click.option("--lin-end", type=float, default=0.2, show_default=True)
@click.option("--lin-duration", type=int, default=500, show_default=True)
@click.option("--step-initial", type=float, default=1.0, show_default=True)
@click.option("--step-gamma", type=float, default=0.5, show_default=True)
@click.option("--step-period", type=int, default=200, show_default=True)
@click.option("--max-run-sec", type=float, default=0.0, show_default=True, help="Max seconds per run; 0 disables timeout")
@click.option("--save-features/--no-save-features", default=True, show_default=True)
@click.option("--save-windows/--no-save-windows", default=False, show_default=True)
@click.option("--save-config/--no-save-config", default=True, show_default=True)
@click.option("--out-root", type=click.Path(dir_okay=True, file_okay=False), required=True, help="Root directory to write runs and manifest.csv")
def neuro_bci_sweep(steps: int, fs: float, window_sec: float, seeds: Tuple[int, ...], process_noise: Tuple[float, ...], drift: Tuple[float, ...], snr_scale: Tuple[float, ...], noise_std: Tuple[float, ...], theta_hz: float, gamma_hz: float, base_lr: float, base_gain: float, ctrl_effect: float, schedulers: Tuple[str, ...], const_v: float, cos_period: int, cos_min: float, cos_max: float, phi_T0: int, phi_val: float, phi_min: float, phi_max: float, lin_start: float, lin_end: float, lin_duration: int, step_initial: float, step_gamma: float, step_period: int, max_run_sec: float, save_features: bool, save_windows: bool, save_config: bool, out_root: str) -> None:
    """Run a grid of BCI simulations and write a manifest of results."""
    try:
        import pandas as pd
        from itertools import product
        from .neuro import bci as bci_mod
        import numpy as np
    except Exception as e:
        click.echo(f"Error importing deps: {e}", err=True)
        sys.exit(1)

    os.makedirs(out_root, exist_ok=True)
    rows = []
    run_idx = 0

    # Build scheduler factory
    def make_scheduler(name: str):
        n = name.lower()
        if n == "constant":
            return bci_mod.ConstantScheduler(v=float(const_v))
        if n == "cosine":
            return bci_mod.CosineScheduler(period=int(cos_period), min_v=float(cos_min), max_v=float(cos_max))
        if n == "linear":
            return bci_mod.LinearScheduler(start_v=float(lin_start), end_v=float(lin_end), duration=int(lin_duration))
        if n == "step":
            return bci_mod.StepScheduler(initial=float(step_initial), gamma=float(step_gamma), period=int(step_period))
        return bci_mod.CosineWithPhiRestarts(T0=int(phi_T0), phi=float(phi_val), min_v=float(phi_min), max_v=float(phi_max))

    for seed, pn, dr, snr, ns, sch_name in product(seeds, process_noise, drift, snr_scale, noise_std, schedulers):
        run_idx += 1
        run_dir = os.path.join(out_root, f"run_{run_idx:04d}")
        os.makedirs(run_dir, exist_ok=True)

        cfg = bci_mod.BCIConfig(
            fs=float(fs), window_sec=float(window_sec), steps=int(steps), seed=int(seed),
            process_noise=float(pn), drift=float(dr), ctrl_effect=float(ctrl_effect), base_lr=float(base_lr), base_gain=float(base_gain),
            noise_std=float(ns), snr_scale=float(snr), theta_hz=float(theta_hz), gamma_hz=float(gamma_hz),
        )
        sch = make_scheduler(sch_name)

        # Cooperative timeout per run via on_step
        run_start = time.time()
        click.echo(
            f"[bci-sweep] start run {run_idx}: seed={seed}, pn={pn}, dr={dr}, snr={snr}, ns={ns}, scheduler={sch_name}",
            err=True,
        )
        def _on_step(t: int) -> None:
            mrs = float(max_run_sec or 0.0)
            if mrs > 0.0 and (time.time() - run_start) > mrs:
                raise bci_mod.SimulationInterrupt("timeout")

        try:
            logs = bci_mod.simulate(
                cfg, scheduler=sch, out_dir=run_dir,
                save_features=bool(save_features), save_windows=bool(save_windows), save_config=bool(save_config),
                on_step=_on_step,
            )
            summary = {k: float(v[0]) for k, v in logs.items() if k in ("mse", "mae", "ttc")}
            row = {
                "run": run_idx,
                "run_dir": run_dir,
                "seed": seed,
                "process_noise": pn,
                "drift": dr,
                "snr_scale": snr,
                "noise_std": ns,
                "scheduler": sch_name,
                **summary,
                "status": "ok",
            }
            rows.append(row)
            elapsed = time.time() - run_start
            click.echo(
                f"[bci-sweep] done run {run_idx} in {elapsed:.2f}s: {summary}",
                err=True,
            )
        except bci_mod.SimulationInterrupt as si:
            # Timeout: record row with NaN metrics and continue
            summary = {"mse": np.nan, "mae": np.nan, "ttc": np.nan}
            row = {
                "run": run_idx,
                "run_dir": run_dir,
                "seed": seed,
                "process_noise": pn,
                "drift": dr,
                "snr_scale": snr,
                "noise_std": ns,
                "scheduler": sch_name,
                **summary,
                "status": "timeout",
            }
            rows.append(row)
            elapsed = time.time() - run_start
            click.echo(
                f"[bci-sweep] timeout run {run_idx} after {elapsed:.2f}s (limit={max_run_sec}s)",
                err=True,
            )

    manifest_path = os.path.join(out_root, "manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    click.echo(json.dumps({"runs": len(rows), "manifest": manifest_path, "out_root": out_root}, indent=2))


@neuro_cmd.command("bci-train")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False), required=True, help="Directory with bci_* artifacts from bci-sim")
@click.option("--mode", type=click.Choice(["features", "windows"], case_sensitive=False), default="features", show_default=True)
@click.option("--model", type=click.Choice(["ridge", "lasso", "linear"], case_sensitive=False), default="ridge", show_default=True)
@click.option("--alpha", type=float, default=1.0, show_default=True, help="Regularization strength for ridge/lasso")
@click.option("--test-size", type=float, default=0.2, show_default=True)
@click.option("--random-state", type=int, default=0, show_default=True)
@click.option("--save-preds/--no-save-preds", default=True, show_default=True)
def neuro_bci_train(data_dir: str, mode: str, model: str, alpha: float, test_size: float, random_state: int, save_preds: bool) -> None:
    """Train a simple regression baseline on one BCI run directory and report metrics."""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import Ridge, Lasso, LinearRegression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from .neuro import load_bci_dataset as load_ds
    except Exception as e:
        click.echo(f"Error importing deps: {e}", err=True)
        sys.exit(1)

    use_features = mode.lower() == "features"
    use_windows = mode.lower() == "windows"
    ds = load_ds(data_dir, use_features=use_features, use_windows=use_windows)
    X = ds["X_feat"] if use_features else ds["X_win"]
    y = ds["y"]
    if X is None:
        click.echo("Error: selected mode has no available X in data_dir", err=True)
        sys.exit(2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), shuffle=False
    )

    if model.lower() == "ridge":
        clf = Ridge(alpha=float(alpha))
    elif model.lower() == "lasso":
        clf = Lasso(alpha=float(alpha))
    else:
        clf = LinearRegression()

    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    metrics = {
        "mse_train": float(mean_squared_error(y_train, y_pred_train)),
        "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "mse_test": float(mean_squared_error(y_test, y_pred_test)),
        "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "mode": mode.lower(),
        "model": model.lower(),
        "alpha": float(alpha),
    }

    # Save metrics
    metrics_path = os.path.join(data_dir, "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Optional predictions
    preds_path = None
    if save_preds:
        ts_path = os.path.join(data_dir, "bci_timeseries.csv")
        t = None
        if os.path.exists(ts_path):
            t = pd.read_csv(ts_path)["t"].to_numpy()
        else:
            t = np.arange(ds["meta"].get("T", X.shape[0]))

        # Map splits back to the full index if needed; here we just save two blocks
        dfp = pd.DataFrame({
            "t": np.concatenate([t[:len(y_train)], t[len(y_train):len(y_train)+len(y_test)]]),
            "split": ["train"] * len(y_train) + ["test"] * len(y_test),
            "y_true": np.concatenate([y_train, y_test]),
            "y_pred": np.concatenate([y_pred_train, y_pred_test]),
        })
        preds_path = os.path.join(data_dir, "train_predictions.csv")
        dfp.to_csv(preds_path, index=False)

    click.echo(json.dumps({"metrics": metrics, "metrics_path": metrics_path, "preds_path": preds_path}, indent=2))


@neuro_cmd.command("bci-eval")
@click.option("--manifest", type=click.Path(exists=True, dir_okay=False), required=True, help="Path to manifest.csv from bci-sweep")
@click.option("--compute-signal-metrics/--no-compute-signal-metrics", default=True, show_default=True)
@click.option("--save-plots/--no-save-plots", default=False, show_default=True)
@click.option("--out-path", type=click.Path(dir_okay=False), default=None, help="Optional path to save group summary CSV (default: manifest_dir/summary.csv)")
def neuro_bci_eval(manifest: str, compute_signal_metrics: bool, save_plots: bool, out_path: Optional[str]) -> None:
    """Aggregate sweep results, optionally compute signal metrics, and summarize."""
    try:
        import pandas as pd
        import numpy as np
        from .signals import compute_metrics as sig_metrics
    except Exception as e:
        click.echo(f"Error importing deps: {e}", err=True)
        sys.exit(1)

    man_path = os.path.abspath(manifest)
    man_dir = os.path.dirname(man_path)
    df = pd.read_csv(man_path)

    # Optionally enrich with signal metrics per run
    enrich_cols = [
        "psd_slope", "pac_tg_mi", "higuchi_fd", "lzc",
        "bp_delta", "bp_theta", "bp_alpha", "bp_beta", "bp_gamma",
    ]
    if compute_signal_metrics:
        new_vals = {c: [] for c in enrich_cols}
        for _, row in df.iterrows():
            run_dir = str(row["run_dir"]) if "run_dir" in row else None
            if not run_dir or not os.path.isdir(run_dir):
                for c in enrich_cols:
                    new_vals[c].append(np.nan)
                continue
            cfg_path = os.path.join(run_dir, "bci_config.json")
            npz_path = os.path.join(run_dir, "bci_windows.npz")
            fs = None
            try:
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        fs = float(cfg.get("fs", 256.0))
                if os.path.exists(npz_path) and fs is not None:
                    npz = np.load(npz_path)
                    X = npz.get("X")
                    if X is not None and X.size > 0:
                        x_concat = np.asarray(X).reshape(-1).astype(float)
                        sm = sig_metrics(x_concat, fs)
                        new_vals["psd_slope"].append(float(sm.psd_slope))
                        new_vals["pac_tg_mi"].append(float(sm.pac_tg_mi))
                        new_vals["higuchi_fd"].append(float(sm.higuchi_fd))
                        new_vals["lzc"].append(float(sm.lzc))
                        for b in ("delta", "theta", "alpha", "beta", "gamma"):
                            new_vals[f"bp_{b}"].append(float(sm.bandpowers.get(b, np.nan)))
                        continue
            except Exception:
                pass
            # Fallback if unavailable or on error
            for c in enrich_cols:
                new_vals[c].append(np.nan)

        for c in enrich_cols:
            df[c] = new_vals[c]

        # Save enriched manifest next to original
        man_enriched = os.path.join(man_dir, "manifest_enriched.csv")
        df.to_csv(man_enriched, index=False)
    else:
        man_enriched = man_path

    # Group summary
    group_keys = [k for k in ["scheduler", "snr_scale", "drift", "process_noise", "noise_std"] if k in df.columns]
    metrics_cols = [c for c in ["mse", "mae", "ttc", *enrich_cols] if c in df.columns]
    if not metrics_cols:
        click.echo("Error: no metrics columns found in manifest", err=True)
        sys.exit(2)

    def _agg_spec(cols):
        spec = {}
        for c in cols:
            spec[c] = ["mean", "std", "count"] if c in ("mse", "mae", "ttc") else ["mean", "std"]
        return spec

    summary = df.groupby(group_keys, dropna=False).agg(_agg_spec(metrics_cols)) if group_keys else df.agg(_agg_spec(metrics_cols))
    # Flatten columns
    summary.columns = [f"{m}_{s}" for (m, s) in summary.columns]
    summary = summary.reset_index()

    # Write outputs
    if out_path is None:
        out_path = os.path.join(man_dir, "summary.csv")
    summary.to_csv(out_path, index=False)

    # Optional plots
    plots = {}
    if save_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless
            import matplotlib.pyplot as plt
            plots_dir = os.path.join(man_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            if "scheduler" in df.columns and "mse" in df.columns:
                fig, ax = plt.subplots(figsize=(5, 3))
                df.boxplot(column="mse", by="scheduler", ax=ax, grid=False)
                ax.set_title("MSE by Scheduler")
                ax.set_xlabel("Scheduler")
                ax.set_ylabel("MSE")
                plt.suptitle("")
                p = os.path.join(plots_dir, "mse_by_scheduler.png")
                fig.tight_layout()
                fig.savefig(p, dpi=150)
                plt.close(fig)
                plots["mse_by_scheduler"] = p

            if "snr_scale" in df.columns and "mse" in df.columns:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.scatter(df["snr_scale"], df["mse"], s=12, alpha=0.7)
                ax.set_title("MSE vs SNR scale")
                ax.set_xlabel("snr_scale")
                ax.set_ylabel("MSE")
                fig.tight_layout()
                p = os.path.join(plots_dir, "mse_vs_snr.png")
                fig.savefig(p, dpi=150)
                plt.close(fig)
                plots["mse_vs_snr"] = p
        except Exception as e:
            click.echo(f"Plotting error (ignored): {e}", err=True)

    click.echo(json.dumps({
        "manifest": man_path,
        "manifest_enriched": man_enriched,
        "summary_csv": out_path,
        "plots": plots if plots else None,
    }, indent=2))
if __name__ == "__main__":
    main()
