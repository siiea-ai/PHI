from pathlib import Path

import numpy as np
import pytest

from phi import multiverse as mmod


def test_generate_save_load_and_preview(tmp_path: Path):
    # Generate a small multiverse stack
    full = mmod.generate_full_stack(width=64, height=48, layers=5, octaves=3, seed=42)
    assert full["type"] == "multiverse_full"
    assert int(full["width"]) == 64
    assert int(full["height"]) == 48
    assert int(full["layers"]) == 5

    # Save/load roundtrip
    p = tmp_path / "multiverse_full.json"
    mmod.save_model(full, str(p))
    loaded = mmod.load_model(str(p))
    assert loaded["type"] == "multiverse_full"

    # Preview mosaic (robust to missing matplotlib)
    out = tmp_path / "preview.png"
    try:
        import matplotlib.pyplot as _  # noqa: F401
    except Exception:
        with pytest.raises(Exception):
            mmod.save_mosaic_from_model(full, str(out))
        return

    mmod.save_mosaic_from_model(full, str(out))
    assert out.exists()



def test_compress_expand_shapes_and_metrics(tmp_path: Path):
    full = mmod.generate_full_stack(width=80, height=50, layers=4, octaves=2, seed=0)

    cfg = mmod.MultiverseConfig(strategy="ratio", spatial_ratio=4, layer_ratio=2, method="interp")
    comp = mmod.compress_stack(full, cfg)
    assert comp["type"] == "multiverse_ratio"
    assert comp["strategy"] == "ratio"
    assert int(comp["spatial_ratio"]) == 4
    assert int(comp["layer_ratio"]) == 2

    ds_w, ds_h, ds_L = int(comp["ds_size"][0]), int(comp["ds_size"][1]), int(comp["ds_size"][2])
    assert ds_w == max(1, int(full["width"]) // 4)
    assert ds_h == max(1, int(full["height"]) // 4)
    assert ds_L == max(1, int(full["layers"]) // 2)

    # Expand back to original size
    recon = mmod.expand_stack(
        comp,
        target_size=(int(full["width"]), int(full["height"]), int(full["layers"])) ,
        method="interp",
    )
    assert recon["type"] == "multiverse_full"
    assert int(recon["width"]) == int(full["width"]) \
        and int(recon["height"]) == int(full["height"]) \
        and int(recon["layers"]) == int(full["layers"]) 

    # Save recon for metrics test below
    p_full = tmp_path / "full.json"
    p_recon = tmp_path / "recon.json"
    mmod.save_model(full, str(p_full))
    mmod.save_model(recon, str(p_recon))

    mdf = mmod.metrics_from_paths(str(p_full), str(p_recon))
    assert hasattr(mdf, "to_csv")
    expected_cols = {"mse", "rmse", "psnr_db", "spec_l1", "spec_corr", "width", "height", "layers"}
    assert expected_cols.issubset(set(mdf.columns))



def test_preview_from_compressed(tmp_path: Path):
    full = mmod.generate_full_stack(width=60, height=40, layers=3, octaves=2, seed=7)
    comp = mmod.compress_stack(full, mmod.MultiverseConfig(strategy="ratio", spatial_ratio=3, layer_ratio=1))
    img_path = tmp_path / "comp_preview.png"

    try:
        import matplotlib.pyplot as _  # noqa: F401
    except Exception:
        with pytest.raises(Exception):
            mmod.save_mosaic_from_model(comp, str(img_path))
        return

    mmod.save_mosaic_from_model(comp, str(img_path))
    assert img_path.exists()
