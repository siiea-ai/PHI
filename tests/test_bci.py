import json

import numpy as np
from click.testing import CliRunner

from phi.neuro.bci import (
    BCIConfig,
    ConstantScheduler,
    CosineScheduler,
    CosineWithPhiRestarts,
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
