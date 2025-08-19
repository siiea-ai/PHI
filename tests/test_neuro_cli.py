from pathlib import Path
import importlib.util

from click.testing import CliRunner

from phi.cli import main as cli_main
from phi import neuro as nmod


def test_neuro_cli_flow(tmp_path: Path):
    runner = CliRunner()

    full_p = tmp_path / "full.json"
    comp_p = tmp_path / "model.json"
    recon_p = tmp_path / "recon.json"
    metrics_p = tmp_path / "metrics.csv"
    preview_p = tmp_path / "adj.png"
    sim_p = tmp_path / "traj.csv"

    # generate full
    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "neuro",
            "generate",
            "--output",
            str(full_p),
            "--nodes",
            "32",
            "--model",
            "ws",
            "--ws-k",
            "6",
            "--ws-p",
            "0.1",
            "--seed",
            "5",
            "--state-init",
            "random",
        ],
    )
    assert res.exit_code == 0, res.output
    assert full_p.exists()
    full = nmod.load_model(str(full_p))
    assert full["type"] == "neuro_network_full"

    # compress
    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "neuro",
            "compress",
            "--input",
            str(full_p),
            "--model",
            str(comp_p),
            "--ratio",
            "4",
            "--method",
            "interp",
        ],
    )
    assert res.exit_code == 0, res.output
    assert comp_p.exists()
    comp = nmod.load_model(str(comp_p))
    assert comp["type"] == "neuro_network_ratio"

    # expand back to full with preview
    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "neuro",
            "expand",
            "--model",
            str(comp_p),
            "--output",
            str(recon_p),
            "--nodes",
            "32",
            "--method",
            "interp",
            "--seed",
            "7",
            "--preview",
            str(preview_p),
        ],
    )
    assert res.exit_code == 0, res.output
    assert recon_p.exists()
    recon = nmod.load_model(str(recon_p))
    assert recon["type"] == "neuro_network_full"

    # simulate on reconstructed full
    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "neuro",
            "simulate",
            "--model",
            str(recon_p),
            "--output",
            str(sim_p),
            "--steps",
            "10",
            "--dt",
            "0.1",
            "--leak",
            "0.1",
            "--input-drive",
            "0.05",
            "--noise-std",
            "0.0",
            "--seed",
            "42",
        ],
    )
    assert res.exit_code == 0, res.output
    assert sim_p.exists()

    # analyze two full bundles
    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "neuro",
            "analyze",
            "--a",
            str(full_p),
            "--b",
            str(recon_p),
            "--output",
            str(metrics_p),
        ],
    )
    assert res.exit_code == 0, res.output
    assert metrics_p.exists()

    # engine with --analyze alias
    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "neuro",
            "engine",
            "--input",
            str(full_p),
            "--recon-output",
            str(recon_p),
            "--model",
            str(comp_p),
            "--ratio",
            "4",
            "--method",
            "interp",
            "--nodes",
            "32",
            "--seed",
            "11",
            "--analyze",
            str(metrics_p),
        ],
    )
    assert res.exit_code == 0, res.output
    assert recon_p.exists() and metrics_p.exists()

    # preview (matplotlib optional): use full bundle
    if importlib.util.find_spec("matplotlib") is None:
        res = runner.invoke(
            cli_main,
            [
                "fractal",
                "neuro",
                "preview",
                "--model",
                str(full_p),
                "--output",
                str(preview_p),
            ],
        )
        assert res.exit_code != 0
        return

    res = runner.invoke(
        cli_main,
        [
            "fractal",
            "neuro",
            "preview",
            "--model",
            str(full_p),
            "--output",
            str(preview_p),
        ],
    )
    assert res.exit_code == 0, res.output
    assert preview_p.exists()
