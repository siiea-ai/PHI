import argparse
import os
from pathlib import Path

import numpy as np

# Heavy optional deps are expected to be installed per project prefs
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional
    matplotlib = None
    plt = None

from phi import neuro as nm


def compute_metrics(X: np.ndarray) -> dict:
    """Compute simple complexity/coordination metrics from states (T, N)."""
    T, N = X.shape
    if T < 2 or N < 1:
        return {
            "synchrony": float("nan"),
            "metastability": float("nan"),
            "pca1_ratio": float("nan"),
            "participation_ratio": float("nan"),
            "spectral_entropy": float("nan"),
        }

    # Remove per-node mean (demean across time)
    Xc = X - X.mean(axis=0, keepdims=True)

    # Synchrony: mean |corr| off-diagonals
    if N >= 2:
        with np.errstate(invalid="ignore"):
            C = np.corrcoef(X.T)
        # replace NaNs from constant columns
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        iu = np.triu_indices_from(C, k=1)
        synchrony = float(np.mean(np.abs(C[iu]))) if iu[0].size > 0 else 0.0
    else:
        synchrony = 0.0

    # Metastability: std over time of spatial std
    sigma_t = X.std(axis=1)
    metastability = float(sigma_t.std())

    # PCA dominance: variance ratio of PC1
    try:
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        total_var = float(np.sum(s**2))
        pca1_ratio = float((s[0] ** 2) / total_var) if total_var > 0 else 0.0
    except np.linalg.LinAlgError:
        pca1_ratio = float("nan")

    # Participation ratio (PR) from covariance eigenvalues
    cov = (Xc.T @ Xc) / max(1, (T - 1))
    try:
        eig = np.linalg.eigvalsh(cov)
        eig = np.clip(np.real(eig), a_min=0.0, a_max=None)
        pr = float((eig.sum() ** 2) / np.sum(eig ** 2)) if np.sum(eig ** 2) > 0 else 0.0
    except np.linalg.LinAlgError:
        pr = float("nan")

    # Spectral entropy of global mean signal
    g = X.mean(axis=1)
    g = g - float(g.mean())
    psd = np.abs(np.fft.rfft(g)) ** 2
    K = psd.size
    if K > 1:
        p = psd / (psd.sum() + 1e-12)
        spectral_entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(K))
    else:
        spectral_entropy = 0.0

    return {
        "synchrony": float(synchrony),
        "metastability": float(metastability),
        "pca1_ratio": float(pca1_ratio),
        "participation_ratio": float(pr),
        "spectral_entropy": float(spectral_entropy),
    }


def plot_heatmaps(df: pd.DataFrame, out_prefix: Path, metrics: list[str]) -> None:
    if plt is None:
        print("[plot] matplotlib not available; skipping plots")
        return
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Ensure deterministic order on axes
    leaks = np.sort(df["leak"].unique())
    noises = np.sort(df["noise"].unique())

    for m in metrics:
        pv = df.pivot(index="leak", columns="noise", values=m)
        pv = pv.reindex(index=leaks, columns=noises)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pv.values, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(noises)))
        ax.set_xticklabels([f"{v:.3f}" for v in noises], rotation=45, ha="right")
        ax.set_yticks(range(len(leaks)))
        ax.set_yticklabels([f"{v:.3f}" for v in leaks])
        ax.set_xlabel("noise_std")
        ax.set_ylabel("leak")
        ax.set_title(m)
        fig.colorbar(im, ax=ax, shrink=0.9)
        out_path = out_prefix.parent / f"{out_prefix.name}_{m}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] saved {out_path}")


def linspace_from_args(lo: float, hi: float, count: int) -> np.ndarray:
    count = max(1, int(count))
    return np.linspace(float(lo), float(hi), count)


def main() -> None:
    ap = argparse.ArgumentParser(description="Neuro dynamics parameter sweep with metrics and optional heatmaps")
    ap.add_argument("--nodes", type=int, default=256)
    ap.add_argument("--model", choices=["ws", "ba"], default="ws")
    ap.add_argument("--ws-k", type=int, default=12)
    ap.add_argument("--ws-p", type=float, default=0.1)
    ap.add_argument("--ba-m", type=int, default=3)

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--burn-in", type=int, default=500, help="number of initial steps to discard as transient")
    ap.add_argument("--input-drive", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--leak-grid", nargs=3, type=float, metavar=("LO", "HI", "COUNT"), default=[0.05, 0.25, 9])
    ap.add_argument("--noise-grid", nargs=3, type=float, metavar=("LO", "HI", "COUNT"), default=[0.0, 0.15, 9])

    ap.add_argument("--out-prefix", type=str, default="neuro_sweep", help="output prefix (CSV: <prefix>.csv; plots: <prefix>_<metric>.png)")
    ap.add_argument("--plots", action="store_true")

    args = ap.parse_args()

    # Build network once per run (fixed topology for sweep)
    if args.model == "ws":
        full = nm.generate_full_network(
            nodes=args.nodes, model="ws", ws_k=args.ws_k, ws_p=args.ws_p, seed=args.seed, state_init="random"
        )
        model_params = {"ws_k": int(args.ws_k), "ws_p": float(args.ws_p)}
    else:
        full = nm.generate_full_network(
            nodes=args.nodes, model="ba", ba_m=args.ba_m, seed=args.seed, state_init="random"
        )
        model_params = {"ba_m": int(args.ba_m)}

    leaks = linspace_from_args(*args.leak_grid)
    noises = linspace_from_args(*args.noise_grid)

    rows: list[dict] = []
    for lv in leaks:
        for nv in noises:
            S = nm.simulate_states(
                full,
                steps=int(args.steps),
                dt=float(args.dt),
                leak=float(lv),
                input_drive=float(args.input_drive),
                noise_std=float(nv),
                seed=args.seed,
            )
            burn = max(0, min(int(args.burn_in), S.shape[0] - 1))
            X = S[burn:]
            m = compute_metrics(X)
            row = {
                "model": args.model,
                "nodes": int(args.nodes),
                "steps": int(args.steps),
                "dt": float(args.dt),
                "burn_in": int(burn),
                "input_drive": float(args.input_drive),
                "seed": int(args.seed),
                "leak": float(lv),
                "noise": float(nv),
                **model_params,
                **m,
            }
            rows.append(row)
            print({k: row[k] for k in ("leak", "noise", "synchrony", "metastability", "pca1_ratio", "participation_ratio", "spectral_entropy")})

    df = pd.DataFrame(rows)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}")

    if args.plots:
        plot_metrics = [
            "synchrony",
            "metastability",
            "pca1_ratio",
            "participation_ratio",
            "spectral_entropy",
        ]
        plot_heatmaps(df, out_prefix, plot_metrics)


if __name__ == "__main__":
    main()
