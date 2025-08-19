from pathlib import Path

import numpy as np
import pytest

from phi import omniverse as omod


def test_generate_save_load_and_preview(tmp_path: Path):
    # Generate a small omniverse grid
    full = omod.generate_full_grid(width=32, height=24, layers=3, universes=2, octaves=2, seed=7)
    assert full["type"] == "omniverse_full"
    assert int(full["width"]) == 32
    assert int(full["height"]) == 24
    assert int(full["layers"]) == 3
    assert int(full["universes"]) == 2

    # Save/load roundtrip
    p = tmp_path / "omni_full.json"
    omod.save_model(full, str(p))
    loaded = omod.load_model(str(p))
    assert loaded["type"] == "omniverse_full"

    # Preview mosaic (robust to missing matplotlib)
    out = tmp_path / "preview.png"
    try:
        import matplotlib.pyplot as _  # noqa: F401
    except Exception:
        with pytest.raises(Exception):
            omod.save_mosaic_from_model(full, str(out))
        return

    omod.save_mosaic_from_model(full, str(out))
    assert out.exists()


def test_compress_expand_shapes_and_metrics(tmp_path: Path):
    full = omod.generate_full_grid(width=40, height=28, layers=4, universes=3, octaves=2, seed=0)

    cfg = omod.OmniverseConfig(strategy="ratio", spatial_ratio=4, layer_ratio=2, universe_ratio=2, method="interp")
    comp = omod.compress_grid(full, cfg)
    assert comp["type"] == "omniverse_ratio"
    assert comp["strategy"] == "ratio"
    assert int(comp["spatial_ratio"]) == 4
    assert int(comp["layer_ratio"]) == 2
    assert int(comp["universe_ratio"]) == 2

    ds_w, ds_h, ds_L, ds_U = map(int, comp["ds_size"])  # [w,h,L,U]
    assert ds_w == max(1, int(full["width"]) // 4)
    assert ds_h == max(1, int(full["height"]) // 4)
    assert ds_L == max(1, int(full["layers"]) // 2)
    assert ds_U == max(1, int(full["universes"]) // 2)

    # Expand back to original size
    recon = omod.expand_grid(
        comp,
        target_size=(int(full["width"]), int(full["height"]), int(full["layers"]), int(full["universes"])) ,
        method="interp",
    )
    assert recon["type"] == "omniverse_full"
    assert int(recon["width"]) == int(full["width"]) \
        and int(recon["height"]) == int(full["height"]) \
        and int(recon["layers"]) == int(full["layers"]) \
        and int(recon["universes"]) == int(full["universes"]) 

    # Save recon for metrics test below
    p_full = tmp_path / "full.json"
    p_recon = tmp_path / "recon.json"
    omod.save_model(full, str(p_full))
    omod.save_model(recon, str(p_recon))

    mdf = omod.metrics_from_paths(str(p_full), str(p_recon))
    assert hasattr(mdf, "to_csv")
    expected_cols = {"mse", "rmse", "psnr_db", "spec_l1", "spec_corr", "width", "height", "layers", "universes"}
    assert expected_cols.issubset(set(mdf.columns))


def test_preview_from_compressed(tmp_path: Path):
    full = omod.generate_full_grid(width=36, height=24, layers=3, universes=2, octaves=2, seed=11)
    comp = omod.compress_grid(full, omod.OmniverseConfig(strategy="ratio", spatial_ratio=3, layer_ratio=1, universe_ratio=1))
    img_path = tmp_path / "comp_preview.png"

    try:
        import matplotlib.pyplot as _  # noqa: F401
    except Exception:
        with pytest.raises(Exception):
            omod.save_mosaic_from_model(comp, str(img_path))
        return

    omod.save_mosaic_from_model(comp, str(img_path))
    assert img_path.exists()
