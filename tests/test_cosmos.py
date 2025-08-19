from pathlib import Path

import importlib.util
import numpy as np
import pytest

from phi import cosmos as cmod


def test_generate_save_load_and_preview(tmp_path: Path):
    # Generate small field
    full = cmod.generate_full_field(width=64, height=48, octaves=3, seed=123)
    assert full["type"] == "cosmos_field_full"
    assert int(full["width"]) == 64
    assert int(full["height"]) == 48

    # Save/load roundtrip
    p = tmp_path / "cosmos_full.json"
    cmod.save_model(full, str(p))
    loaded = cmod.load_model(str(p))
    assert loaded["type"] == "cosmos_field_full"

    # Preview image (robust to missing matplotlib)
    out = tmp_path / "preview.png"
    try:
        import matplotlib.pyplot as _  # noqa: F401
    except Exception:
        with pytest.raises(Exception):
            cmod.save_image_from_model(full, str(out))
        return

    cmod.save_image_from_model(full, str(out))
    assert out.exists()


def test_compress_expand_shapes(tmp_path: Path):
    full = cmod.generate_full_field(width=80, height=50, octaves=2, seed=0)

    cfg = cmod.CosmosConfig(strategy="ratio", ratio=4, method="interp")
    comp = cmod.compress_field(full, cfg)
    assert comp["type"] == "cosmos_field_ratio"
    assert int(comp["ratio"]) == 4
    ds_w, ds_h = int(comp["ds_size"][0]), int(comp["ds_size"][1])
    assert ds_w == max(1, int(full["width"]) // 4)
    assert ds_h == max(1, int(full["height"]) // 4)

    # Expand to original size
    recon = cmod.expand_field(comp, target_size=(int(full["width"]), int(full["height"])), method="interp")
    assert recon["type"] == "cosmos_field_full"
    assert int(recon["width"]) == int(full["width"]) and int(recon["height"]) == int(full["height"]) 

    # Save recon for metrics test below
    p_full = tmp_path / "full.json"
    p_recon = tmp_path / "recon.json"
    cmod.save_model(full, str(p_full))
    cmod.save_model(recon, str(p_recon))

    mdf = cmod.metrics_from_paths(str(p_full), str(p_recon))
    assert hasattr(mdf, "to_csv")
    expected_cols = {"mse", "rmse", "psnr_db", "spec_l1", "spec_corr", "width", "height"}
    assert expected_cols.issubset(set(mdf.columns))


def test_preview_from_compressed(tmp_path: Path):
    full = cmod.generate_full_field(width=60, height=40, octaves=2, seed=7)
    comp = cmod.compress_field(full, cmod.CosmosConfig(strategy="ratio", ratio=3))
    img_path = tmp_path / "comp_preview.png"

    try:
        import matplotlib.pyplot as _  # noqa: F401
    except Exception:
        with pytest.raises(Exception):
            cmod.save_image_from_model(comp, str(img_path))
        return

    cmod.save_image_from_model(comp, str(img_path))
    assert img_path.exists()
