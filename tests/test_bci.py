import json
import os

import numpy as np
import pandas as pd
from click.testing import CliRunner

from phi.neuro.bci import (
    BCIConfig,
    ConstantScheduler,
    CosineScheduler,
    CosineWithPhiRestarts,
    LinearScheduler,
    StepScheduler,
    simulate,
)
from phi.cli import main as cli_main


def test_simulate_shapes_and_metrics(tmp_path):
    cfg = BCIConfig(steps=120, fs=128.0, window_sec=0.5, seed=7)
    sch = ConstantScheduler(v=1.0)
    logs = simulate(cfg, scheduler=sch, out_dir=str(tmp_path))

    assert set(["y", "y_hat", "err", "sched"]) <= set(logs.keys())
    T = cfg.steps
    for k in ("y", "y_hat", "err", "sched"):
        assert logs[k].shape == (T,)
    # summary metrics
    assert np.isfinite(logs["mse"][0])
    assert np.isfinite(logs["mae"][0])
    assert 0 <= logs["ttc"][0] <= T


def test_cli_neuro_bci_sim(tmp_path):
    runner = CliRunner()
    res = runner.invoke(
        cli_main,
        [
            "neuro",
            "bci-sim",
            "--steps",
            "60",
            "--fs",
            "128",
            "--window-sec",
            "0.5",
            "--scheduler",
            "cosine_phi",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert "summary" in data
    s = data["summary"]
    assert set(["mse", "mae", "ttc"]) <= set(s.keys())


def test_simulate_linear_and_step(tmp_path):
    cfg = BCIConfig(steps=80, fs=128.0, window_sec=0.25, seed=11)

    # Linear scheduler run
    lin = LinearScheduler(start_v=1.0, end_v=0.2, duration=80)
    logs_lin = simulate(cfg, scheduler=lin, out_dir=str(tmp_path / "lin"))
    assert logs_lin["y"].shape == (cfg.steps,)
    assert np.isfinite(logs_lin["mse"][0])

    # Step scheduler run
    step = StepScheduler(initial=1.0, gamma=0.5, period=10)
    logs_step = simulate(cfg, scheduler=step, out_dir=str(tmp_path / "step"))
    assert logs_step["y"].shape == (cfg.steps,)
    assert np.isfinite(logs_step["mse"][0])


def test_cli_neuro_bci_sim_linear(tmp_path):
    runner = CliRunner()
    res = runner.invoke(
        cli_main,
        [
            "neuro",
            "bci-sim",
            "--steps",
            "60",
            "--fs",
            "128",
            "--window-sec",
            "0.5",
            "--scheduler",
            "linear",
            "--lin-start",
            "1.0",
            "--lin-end",
            "0.2",
            "--lin-duration",
            "40",
            "--out-dir",
            str(tmp_path / "sim_linear"),
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert "summary" in data
    s = data["summary"]
    assert set(["mse", "mae", "ttc"]) <= set(s.keys())


def test_cli_neuro_bci_sim_step(tmp_path):
    runner = CliRunner()
    res = runner.invoke(
        cli_main,
        [
            "neuro",
            "bci-sim",
            "--steps",
            "60",
            "--fs",
            "128",
            "--window-sec",
            "0.5",
            "--scheduler",
            "step",
            "--step-initial",
            "1.0",
            "--step-gamma",
            "0.7",
            "--step-period",
            "15",
            "--out-dir",
            str(tmp_path / "sim_step"),
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert "summary" in data
    s = data["summary"]
    assert set(["mse", "mae", "ttc"]) <= set(s.keys())


def test_cli_neuro_bci_sweep_linear_step(tmp_path):
    runner = CliRunner()
    out_root = tmp_path / "sweep"
    res = runner.invoke(
        cli_main,
        [
            "neuro",
            "bci-sweep",
            "--steps",
            "30",
            "--out-root",
            str(out_root),
            "--schedulers",
            "linear",
            "--schedulers",
            "step",
            "--lin-duration",
            "20",
            "--step-period",
            "10",
            "--no-save-features",
            "--no-save-windows",
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data.get("runs") == 4  # 2 seeds x 2 schedulers with defaults
    man_path = data.get("manifest")
    assert man_path and os.path.exists(man_path)
    df = pd.read_csv(man_path)
    assert df.shape[0] == 4
    assert set(["linear", "step"]) <= set(df["scheduler"].unique().tolist())


def test_cli_bci_config_serialization(tmp_path):
    runner = CliRunner()

    # Linear scheduler: ensure config JSON records scheduler type and params
    out_lin = tmp_path / "cfg_linear"
    res_lin = runner.invoke(
        cli_main,
        [
            "neuro",
            "bci-sim",
            "--steps",
            "40",
            "--fs",
            "128",
            "--window-sec",
            "0.5",
            "--scheduler",
            "linear",
            "--lin-start",
            "1.0",
            "--lin-end",
            "0.2",
            "--lin-duration",
            "40",
            "--out-dir",
            str(out_lin),
            "--save-config",
        ],
    )
    assert res_lin.exit_code == 0, res_lin.output
    cfg_path = os.path.join(str(out_lin), "bci_config.json")
    assert os.path.exists(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sch = cfg.get("scheduler", {})
    assert sch.get("type") == "linear"
    params = sch.get("params", {})
    assert params.get("start_v") == 1.0
    assert params.get("end_v") == 0.2
    assert params.get("duration") == 40

    # Step scheduler: ensure config JSON records scheduler type and params
    out_step = tmp_path / "cfg_step"
    res_step = runner.invoke(
        cli_main,
        [
            "neuro",
            "bci-sim",
            "--steps",
            "30",
            "--fs",
            "128",
            "--window-sec",
            "0.5",
            "--scheduler",
            "step",
            "--step-initial",
            "1.0",
            "--step-gamma",
            "0.7",
            "--step-period",
            "15",
            "--out-dir",
            str(out_step),
            "--save-config",
        ],
    )
    assert res_step.exit_code == 0, res_step.output
    cfg2_path = os.path.join(str(out_step), "bci_config.json")
    assert os.path.exists(cfg2_path)
    with open(cfg2_path, "r", encoding="utf-8") as f:
        cfg2 = json.load(f)
    sch2 = cfg2.get("scheduler", {})
    assert sch2.get("type") == "step"
    p2 = sch2.get("params", {})
    assert p2.get("initial") == 1.0
    assert p2.get("gamma") == 0.7
    assert p2.get("period") == 15
