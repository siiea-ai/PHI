#!/usr/bin/env python3
"""
Programmatic neuro pipeline example using phi.neuro:
- Generate a full neuro graph (WS or BA)
- Compress to ratio model
- Expand back to full network
- Simulate neuron states over time
- Analyze reconstruction fidelity (state MSE, degree-hist L1)

Run from repo root (recommended):
  .venv/bin/python examples/neuro_programmatic.py --outdir examples/neuro_out
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path if run from a subfolder
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from phi import neuro as nmod  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Programmatic neuro: generate -> compress -> expand -> simulate -> analyze")
    p.add_argument("--outdir", type=str, default=os.path.join("examples", "neuro_out"), help="Output directory for artifacts")

    # Generation params
    p.add_argument("--nodes", type=int, default=500, help="Number of neurons (nodes)")
    p.add_argument("--model", type=str, choices=["ws", "ba"], default="ws", help="Graph model")
    p.add_argument("--ws-k", type=int, default=10, help="WS: k-nearest neighbors")
    p.add_argument("--ws-p", type=float, default=0.1, help="WS: rewiring probability")
    p.add_argument("--ba-m", type=int, default=3, help="BA: edges per new node")
    p.add_argument("--seed", type=int, default=42, help="Random seed for generation and expansion")
    p.add_argument("--state-init", type=str, choices=["random", "zeros"], default="random", help="Initial neuron state init")

    # Compression/Expansion
    p.add_argument("--ratio", type=int, default=4, help="Keep every Nth neuron")
    p.add_argument("--method", type=str, choices=["interp", "nearest"], default="interp", help="State upsample method on expand")
    p.add_argument("--target-nodes", type=int, default=None, help="Target nodes for reconstruction (default: --nodes)")

    # Simulation params
    p.add_argument("--steps", type=int, default=200, help="Simulation steps")
    p.add_argument("--dt", type=float, default=0.05, help="Time step size")
    p.add_argument("--leak", type=float, default=0.1, help="Leak coefficient")
    p.add_argument("--input-drive", type=float, default=0.05, help="External input drive added to all neurons")
    p.add_argument("--noise-std", type=float, default=0.01, help="Gaussian noise std added to input drive")
    p.add_argument("--noise-seed", type=int, default=None, help="Random seed for noise (default: --seed)")

    # Output controls
    p.add_argument("--no-preview", action="store_true", help="Skip writing adjacency preview images")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    # File paths
    full_path = os.path.join(args.outdir, "neuro_full.json")
    model_path = os.path.join(args.outdir, "neuro_model.json")
    recon_path = os.path.join(args.outdir, "neuro_recon.json")
    adj_full = os.path.join(args.outdir, "neuro_adj.png")
    adj_recon = os.path.join(args.outdir, "neuro_recon_adj.png")
    states_csv = os.path.join(args.outdir, "neuro_states.csv")
    metrics_csv = os.path.join(args.outdir, "neuro_metrics.csv")

    print("[1/5] Generating full neuro network ...")
    full = nmod.generate_full_network(
        nodes=args.nodes,
        model=args.model,
        ws_k=args.ws_k,
        ws_p=args.ws_p,
        ba_m=args.ba_m,
        seed=args.seed,
        state_init=args.state_init,
    )
    nmod.save_model(full, full_path)
    if not args.no_preview:
        nmod.save_adjacency_image(full, adj_full)
    print(f"  -> full: {full_path}")
    if not args.no_preview:
        print(f"  -> preview: {adj_full}")

    print("[2/5] Compressing to ratio model ...")
    cfg = nmod.NeuroConfig(strategy="ratio", ratio=args.ratio, method=args.method, edge_method="regen")
    comp = nmod.compress_network(full, cfg)
    nmod.save_model(comp, model_path)
    print(f"  -> model: {model_path}")

    print("[3/5] Expanding back to full network ...")
    target_nodes = int(args.target_nodes) if args.target_nodes is not None else int(args.nodes)
    full_recon = nmod.expand_network(comp, target_nodes=target_nodes, method=args.method, seed=args.seed)
    nmod.save_model(full_recon, recon_path)
    if not args.no_preview:
        nmod.save_adjacency_image(full_recon, adj_recon)
    print(f"  -> recon: {recon_path}")
    if not args.no_preview:
        print(f"  -> preview: {adj_recon}")

    print("[4/5] Simulating neuron states ...")
    noise_seed = args.noise_seed if args.noise_seed is not None else args.seed
    traj = nmod.simulate_states(
        full_recon,
        steps=args.steps,
        dt=args.dt,
        leak=args.leak,
        input_drive=args.input_drive,
        noise_std=args.noise_std,
        seed=noise_seed,
    )
    t = np.arange(traj.shape[0], dtype=np.int32)
    cols = ["t"] + [f"x{i}" for i in range(traj.shape[1])]
    df = pd.DataFrame(np.column_stack([t, traj]), columns=cols)
    df.to_csv(states_csv, index=False)
    print(f"  -> states: {states_csv}")

    print("[5/5] Analyzing reconstruction fidelity ...")
    mdf = nmod.metrics_from_paths(full_path, recon_path)
    mdf.to_csv(metrics_csv, index=False)
    print(f"  -> metrics: {metrics_csv}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
