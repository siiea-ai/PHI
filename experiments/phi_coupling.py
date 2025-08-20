from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import List

import click
import numpy as np

from phi.neuro.bci import (
    BCIConfig,
    ConstantScheduler,
    CosineScheduler,
    CosineWithPhiRestarts,
    simulate,
)


@click.command()
@click.option("--steps", type=int, default=1000, show_default=True, help="Closed-loop steps per run")
@click.option("--seeds", type=str, default="42,43,44,45,46", show_default=True, help="Comma-separated seeds")
@click.option("--schedulers", type=str, default="constant,cosine,cosine_phi", show_default=True, help="Schedulers to compare")
@click.option("--out-csv", type=click.Path(dir_okay=False), default="out/bci_compare.csv", show_default=True, help="Output CSV path")
@click.option("--save-logs/--no-save-logs", default=False, show_default=True, help="Save per-run logs under out/bci_runs/")
@click.option("--fs", type=float, default=256.0, show_default=True, help="Sampling rate (Hz)")
@click.option("--window-sec", type=float, default=1.0, show_default=True, help="Window duration per step (sec)")
@click.option("--base-gain", type=float, default=0.5, show_default=True, help="Controller base gain")
@click.option("--base-lr", type=float, default=0.05, show_default=True, help="Decoder base learning rate")
@click.option("--verbose/--quiet", default=True, show_default=True, help="Print progress")
def main(steps: int, seeds: str, schedulers: str, out_csv: str, save_logs: bool, fs: float, window_sec: float, base_gain: float, base_lr: float, verbose: bool) -> None:
    """Benchmark φ-based scheduler vs baselines for closed-loop BCI control.

    Produces a CSV with summary metrics (MSE, MAE, TTC) per seed and scheduler.
    """
    seeds_list: List[int] = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    sched_list = [s.strip().lower() for s in schedulers.split(",") if s.strip()]

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    runs_root = "out/bci_runs"
    if save_logs:
        os.makedirs(runs_root, exist_ok=True)

    import pandas as pd

    rows = []
    for sched_name in sched_list:
        for sd in seeds_list:
            if sched_name == "constant":
                sch = ConstantScheduler(v=1.0)
            elif sched_name == "cosine":
                sch = CosineScheduler(period=200, min_v=0.2, max_v=1.0)
            elif sched_name in ("cosine_phi", "phi", "phi_restarts"):
                sch = CosineWithPhiRestarts(T0=200, phi=(1 + 5 ** 0.5) / 2, min_v=0.2, max_v=1.0)
            else:
                raise click.UsageError(f"Unknown scheduler: {sched_name}")

            cfg = BCIConfig(
                fs=fs,
                window_sec=window_sec,
                steps=steps,
                seed=sd,
                base_gain=base_gain,
                base_lr=base_lr,
            )
            out_dir = os.path.join(runs_root, f"{sched_name}_seed{sd}") if save_logs else None
            logs = simulate(cfg, scheduler=sch, out_dir=out_dir)
            summary = {k: float(v[0]) for k, v in logs.items() if k in ("mse", "mae", "ttc")}
            row = {"scheduler": sched_name, "seed": sd, **summary}
            rows.append(row)
            if verbose:
                click.echo(json.dumps(row))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    if verbose:
        click.echo(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
