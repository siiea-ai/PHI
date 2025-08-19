from pathlib import Path
import importlib.util

import pytest
from click.testing import CliRunner

from phi.cli import main as cli_main
from phi import multiverse as mmod


def test_cli_generate_compress_expand_engine_and_preview(tmp_path: Path):
    runner = CliRunner()

    full_p = tmp_path / "full.json"
    comp_p = tmp_path / "model.json"
    recon_p = tmp_path / "recon.json"
    metrics_p = tmp_path / "metrics.csv"
    mosaic_p = tmp_path / "mosaic.png"

    # generate (full)
    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "multiverse",
            "generate",
            "--output",
            str(full_p),
            "--width",
            "32",
            "--height",
            "24",
            "--layers",
            "3",
            "--octaves",
            "2",
            "--seed",
            "5",
        ],
    )
    assert res.exit_code == 0, res.output
    assert full_p.exists()

    # compress
    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "multiverse",
            "compress",
            "--input",
            str(full_p),
            "--model",
            str(comp_p),
            "--spatial-ratio",
            "2",
            "--layer-ratio",
            "1",
            "--method",
            "interp",
        ],
    )
    assert res.exit_code == 0, res.output
    assert comp_p.exists()
    comp = mmod.load_model(str(comp_p))
    assert comp["type"] == "multiverse_ratio"

    # expand
    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "multiverse",
            "expand",
            "--model",
            str(comp_p),
            "--output",
            str(recon_p),
            "--width",
            "32",
            "--height",
            "24",
            "--layers",
            "3",
            "--method",
            "interp",
        ],
    )
    assert res.exit_code == 0, res.output
    assert recon_p.exists()
    recon = mmod.load_model(str(recon_p))
    assert recon["type"] == "multiverse_full"

    # engine (reconstruct from full via compress->expand) with metrics
    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "multiverse",
            "engine",
            "--input",
            str(full_p),
            "--recon-output",
            str(recon_p),
            "--model",
            str(comp_p),
            "--spatial-ratio",
            "2",
            "--layer-ratio",
            "1",
            "--method",
            "interp",
            "--analyze",
            str(metrics_p),
        ],
    )
    assert res.exit_code == 0, res.output
    assert metrics_p.exists()

    # preview (matplotlib optional)
    if importlib.util.find_spec("matplotlib") is None:
        res = runner.invoke(
            cli_main,
            [
                "fractal",
                "multiverse",
                "preview",
                "--model",
                str(full_p),
                "--output",
                str(mosaic_p),
            ],
        )
        assert res.exit_code != 0  # expecting failure if matplotlib missing
        return

    # matplotlib available: preview should succeed
    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "multiverse",
            "preview",
            "--model",
            str(full_p),
            "--output",
            str(mosaic_p),
        ],
    )
    assert res.exit_code == 0, res.output
    assert mosaic_p.exists()
