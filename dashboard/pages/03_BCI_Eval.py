import os
import sys
import pathlib
from typing import Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phi.signals import compute_metrics as sig_metrics  # noqa: E402

st.set_page_config(page_title="BCI Eval", page_icon="🧠", layout="wide")
st.title("Neuro BCI • Sweep Evaluation")

with st.sidebar:
    st.header("Inputs")
    manifest = st.text_input("manifest.csv path", value="")
    compute_signal_metrics = st.checkbox("compute_signal_metrics", value=True)
    save_plots = st.checkbox("save_plots", value=False)
    out_path = st.text_input("summary out_path (optional)", value="")
    run_btn = st.button("Run Evaluation", type="primary")


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

if run_btn:
    if not manifest or not os.path.exists(manifest):
        st.error("Provide a valid manifest.csv path")
        st.stop()

    man_path = os.path.abspath(manifest)
    man_dir = os.path.dirname(man_path)

    df = pd.read_csv(man_path)

    # Optional enrichment
    enrich_cols = [
        "psd_slope", "pac_tg_mi", "higuchi_fd", "lzc",
        "bp_delta", "bp_theta", "bp_alpha", "bp_beta", "bp_gamma",
    ]

    if compute_signal_metrics:
        new_vals: Dict[str, list] = {c: [] for c in enrich_cols}
        for _, row in df.iterrows():
            run_dir = str(row["run_dir"]) if "run_dir" in df.columns else None
            if not run_dir or not os.path.isdir(run_dir):
                for c in enrich_cols:
                    new_vals[c].append(np.nan)
                continue
            cfg_path = os.path.join(run_dir, "bci_config.json")
            npz_path = os.path.join(run_dir, "bci_windows.npz")
            fs: Optional[float] = None
            try:
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        import json
                        cfg = json.load(f)
                        fs = _safe_float(cfg.get("fs", 256.0))
                if os.path.exists(npz_path) and fs is not None:
                    npz = np.load(npz_path)
                    X = npz.get("X")
                    if X is not None and X.size > 0:
                        x_concat = np.asarray(X).reshape(-1).astype(float)
                        sm = sig_metrics(x_concat, float(fs))
                        new_vals["psd_slope"].append(float(sm.psd_slope))
                        new_vals["pac_tg_mi"].append(float(sm.pac_tg_mi))
                        new_vals["higuchi_fd"].append(float(sm.higuchi_fd))
                        new_vals["lzc"].append(float(sm.lzc))
                        for b in ("delta", "theta", "alpha", "beta", "gamma"):
                            new_vals[f"bp_{b}"].append(float(sm.bandpowers.get(b, np.nan)))
                        continue
            except Exception:
                pass
            for c in enrich_cols:
                new_vals[c].append(np.nan)

        for c in enrich_cols:
            df[c] = new_vals[c]
        man_enriched = os.path.join(man_dir, "manifest_enriched.csv")
        df.to_csv(man_enriched, index=False)
    else:
        man_enriched = man_path

    group_keys = [k for k in ["scheduler", "snr_scale", "drift", "process_noise", "noise_std"] if k in df.columns]
    metrics_cols = [c for c in ["mse", "mae", "ttc", *enrich_cols] if c in df.columns]
    if not metrics_cols:
        st.error("No metrics columns found in manifest")
        st.stop()

    def _agg_spec(cols):
        spec = {}
        for c in cols:
            spec[c] = ["mean", "std", "count"] if c in ("mse", "mae", "ttc") else ["mean", "std"]
        return spec

    summary = df.groupby(group_keys, dropna=False).agg(_agg_spec(metrics_cols)) if group_keys else df.agg(_agg_spec(metrics_cols))
    summary.columns = [f"{m}_{s}" for (m, s) in summary.columns]
    summary = summary.reset_index()

    if not out_path:
        out_path = os.path.join(man_dir, "summary.csv")
    summary.to_csv(out_path, index=False)

    st.success("Evaluation complete")
    st.write(f"Summary CSV: {out_path}")
    st.dataframe(summary.head(1000))

    # Downloads for summary and enriched manifest
    try:
        with open(out_path, "rb") as f:
            st.download_button("Download summary.csv", data=f.read(), file_name="summary.csv", mime="text/csv")
    except Exception:
        pass
    if compute_signal_metrics and os.path.exists(man_enriched):
        try:
            with open(man_enriched, "rb") as f:
                st.download_button("Download manifest_enriched.csv", data=f.read(), file_name="manifest_enriched.csv", mime="text/csv")
        except Exception:
            pass

    # Optional plots: mirror CLI defaults
    if save_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plots_dir = os.path.join(man_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        plot_paths = {}

        if "scheduler" in df.columns and "mse" in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3))
            df.boxplot(column="mse", by="scheduler", ax=ax, grid=False)
            ax.set_title("MSE by Scheduler")
            ax.set_xlabel("Scheduler")
            ax.set_ylabel("MSE")
            plt.suptitle("")
            p = os.path.join(plots_dir, "mse_by_scheduler.png")
            fig.tight_layout()
            fig.savefig(p, dpi=150)
            st.pyplot(fig)
            plt.close(fig)
            plot_paths["mse_by_scheduler"] = p

        if "snr_scale" in df.columns and "mse" in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.scatter(df["snr_scale"], df["mse"], s=12, alpha=0.7)
            ax.set_title("MSE vs SNR scale")
            ax.set_xlabel("snr_scale")
            ax.set_ylabel("MSE")
            fig.tight_layout()
            p = os.path.join(plots_dir, "mse_vs_snr.png")
            fig.savefig(p, dpi=150)
            st.pyplot(fig)
            plt.close(fig)
            plot_paths["mse_vs_snr"] = p

        st.subheader("Saved plot files")
        st.json(plot_paths)
        # Offer downloads for plots
        if plot_paths:
            c_plots = st.columns(min(3, len(plot_paths)))
            idx = 0
            for name, path in plot_paths.items():
                try:
                    with open(path, "rb") as f:
                        c_plots[idx % len(c_plots)].download_button(
                            f"Download {name}.png", data=f.read(), file_name=f"{name}.png", mime="image/png"
                        )
                except Exception:
                    pass
                idx += 1
