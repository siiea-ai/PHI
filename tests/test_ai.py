from pathlib import Path

import importlib.util
import numpy as np
import pytest

from phi import ai as ai_mod


def test_generate_and_save_load_roundtrip(tmp_path: Path):
    full = ai_mod.generate_full_model(
        input_dim=16,
        output_dim=4,
        depth=3,
        base_width=32,
        mode="phi",
        seed=123,
    )
    assert full["type"] == "phi-ai-full"
    assert int(full["input_dim"]) == 16
    assert int(full["output_dim"]) == 4
    assert len(full["hidden"]) == 3
    assert int(full["param_count"]) > 0

    p = tmp_path / "ai_full.json"
    ai_mod.save_model(full, str(p))
    loaded = ai_mod.load_model(str(p))
    assert loaded["type"] == "phi-ai-full"
    assert loaded["hidden"] == full["hidden"]


def test_compress_expand_shapes_and_metrics(tmp_path: Path):
    # Generate a small full model
    full = ai_mod.generate_full_model(
        input_dim=12,
        output_dim=3,
        depth=2,
        base_width=24,
        mode="phi",
        seed=0,
    )

    # Compress with ratio=2
    cfg = ai_mod.AIConfig(strategy="ratio", ratio=2, method="interp")
    comp = ai_mod.compress_model(full, cfg)
    assert comp["type"] == "phi-ai-model"
    assert comp["strategy"] == "ratio"
    assert int(comp["ratio"]) == 2
    assert len(comp["ds_hidden"]) == len(full["hidden"])  # one per hidden layer

    # Expand back to original hidden widths
    recon = ai_mod.expand_model(comp, target_hidden=list(full["hidden"]), method="interp", seed=1)
    assert recon["type"] == "phi-ai-full"
    assert recon["hidden"] == full["hidden"]

    # Save and compute metrics
    orig_p = tmp_path / "ai_full.json"
    recon_p = tmp_path / "ai_recon.json"
    ai_mod.save_model(full, str(orig_p))
    ai_mod.save_model(recon, str(recon_p))

    mdf = ai_mod.metrics_from_paths(str(orig_p), str(recon_p))
    assert hasattr(mdf, "to_csv")
    assert set(["layers_equal", "mse_total"]) <= set(mdf.columns)
    assert len(mdf) == 1


def test_export_keras_if_tf_available(tmp_path: Path):
    # If TensorFlow is available, export should succeed and write a file
    if importlib.util.find_spec("tensorflow") is None:
        pytest.skip("TensorFlow not installed")
    import tensorflow as tf  # noqa: F401

    full = ai_mod.generate_full_model(
        input_dim=8,
        output_dim=2,
        depth=2,
        base_width=16,
        mode="phi",
        seed=0,
    )
    out = tmp_path / "model.h5"
    ai_mod.export_keras(full, str(out))
    assert out.exists()


def test_export_keras_skipped_when_tf_missing():
    # If TensorFlow is NOT available, ensure raising a helpful RuntimeError
    if importlib.util.find_spec("tensorflow") is not None:
        pytest.skip("TensorFlow installed; covered by export success test")
    full = ai_mod.generate_full_model(8, 2, depth=1, base_width=8, mode="phi", seed=0)
    with pytest.raises(RuntimeError):
        ai_mod.export_keras(full, "_should_not_exist.h5")
