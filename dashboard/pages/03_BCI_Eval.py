import os
import sys
import pathlib
import threading
import uuid
import json
from typing import Optional, Dict, Any
import time

import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phi.signals import compute_metrics as sig_metrics  # noqa: E402

st.set_page_config(page_title="BCI Eval", page_icon="🧠", layout="wide")
st.title("Neuro BCI • Sweep Evaluation")
st.markdown(
    """
    How to use:
    - Provide a sweep `manifest.csv` produced by the BCI Sweep page (or CLI).
    - Optionally enable signal metrics enrichment and saving plots.
    - Click "Start Evaluation" to run in the background. Progress and logs update live.
    - On completion, download the summary and enriched manifest.
    """
)

# ---------------------------- Background Job ---------------------------- #
class EvalJob:
    def __init__(self, params: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.status = "running"  # running|completed|cancelled|error
        self.progress = 0.0
        self.error: Optional[str] = None
        self.results: Dict[str, Any] = {}
        self.start_time = time.time()
        self.stage: str = "initializing"
        self.logs: list[str] = []
        self._cancel_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(params,), daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_evt.set()

    def _log(self, msg: str) -> None:
        try:
            ts = time.strftime("%H:%M:%S")
            self.logs.append(f"[{ts}] {msg}")
        except Exception:
            self.logs.append(str(msg))

    def _run(self, params: Dict[str, Any]) -> None:
        try:
            man_path = os.path.abspath(str(params.get("manifest", "")))
            compute_signal_metrics = bool(params.get("compute_signal_metrics", True))
            save_plots = bool(params.get("save_plots", False))
            out_path = params.get("out_path")
            out_path = str(out_path) if out_path else None

            if not man_path or not os.path.exists(man_path):
                self.error = "Provide a valid manifest.csv path"
                self.status = "error"
                return

            self.stage = "loading manifest"
            self._log(f"Loading manifest: {man_path}")
            man_dir = os.path.dirname(man_path)
            df = pd.read_csv(man_path)
            self.progress = 0.1

            # Optional enrichment
            enrich_cols = [
                "psd_slope", "pac_tg_mi", "higuchi_fd", "lzc",
                "bp_delta", "bp_theta", "bp_alpha", "bp_beta", "bp_gamma",
            ]

            if compute_signal_metrics:
                self.stage = "computing signal metrics"
                self._log(f"Computing signal metrics for {len(df)} rows")
                new_vals: Dict[str, list] = {c: [] for c in enrich_cols}
                N = int(len(df)) if len(df) > 0 else 1
                step = 0.6 / float(N)
                for _, row in df.iterrows():
                    if self._cancel_evt.is_set():
                        self.status = "cancelled"
                        return
                    run_dir = str(row["run_dir"]) if "run_dir" in df.columns else None
                    if not run_dir or not os.path.isdir(run_dir):
                        for c in enrich_cols:
                            new_vals[c].append(np.nan)
                        self.progress = min(0.1 + step, 0.9)
                        continue
                    cfg_path = os.path.join(run_dir, "bci_config.json")
                    npz_path = os.path.join(run_dir, "bci_windows.npz")
                    fs: Optional[float] = None
                    try:
                        if os.path.exists(cfg_path):
                            with open(cfg_path, "r", encoding="utf-8") as f:
                                cfg = json.load(f)
                                fs = float(cfg.get("fs", 256.0))
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
                                self.progress = min(self.progress + step, 0.9)
                                continue
                    except Exception:
                        pass
                    # Fallback if unavailable or on error
                    for c in enrich_cols:
                        new_vals[c].append(np.nan)
                    self.progress = min(self.progress + step, 0.9)

                for c in enrich_cols:
                    df[c] = new_vals[c]
                man_enriched = os.path.join(man_dir, "manifest_enriched.csv")
                self.stage = "saving enriched manifest"
                self._log(f"Saving enriched manifest to {man_enriched}")
                df.to_csv(man_enriched, index=False)
            else:
                man_enriched = man_path

            self.progress = max(self.progress, 0.9)

            # Group summary
            self.stage = "grouping summary"
            group_keys = [k for k in ["scheduler", "snr_scale", "drift", "process_noise", "noise_std"] if k in df.columns]
            metrics_cols = [c for c in ["mse", "mae", "ttc", *([
                "psd_slope", "pac_tg_mi", "higuchi_fd", "lzc",
                "bp_delta", "bp_theta", "bp_alpha", "bp_beta", "bp_gamma",
            ] if compute_signal_metrics else [])] if c in df.columns]
            if not metrics_cols:
                self.error = "No metrics columns found in manifest"
                self.status = "error"
                return

            def _agg_spec(cols):
                spec: Dict[str, Any] = {}
                for c in cols:
                    spec[c] = ["mean", "std", "count"] if c in ("mse", "mae", "ttc") else ["mean", "std"]
                return spec

            summary = df.groupby(group_keys, dropna=False).agg(_agg_spec(metrics_cols)) if group_keys else df.agg(_agg_spec(metrics_cols))
            summary.columns = [f"{m}_{s}" for (m, s) in summary.columns]
            summary = summary.reset_index()

            if out_path is None:
                out_path = os.path.join(man_dir, "summary.csv")
            self.stage = "saving summary"
            self._log(f"Saving summary CSV to {out_path}")
            summary.to_csv(out_path, index=False)

            # Optional plots
            plots: Dict[str, str] = {}
            if save_plots:
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    plots_dir = os.path.join(man_dir, "plots")
                    os.makedirs(plots_dir, exist_ok=True)

                    self.stage = "saving plots"
                    self._log(f"Saving plots into {plots_dir}")
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
                        plt.close(fig)
                        plots["mse_by_scheduler"] = p

                    if "snr_scale" in df.columns and "mse" in df.columns:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.scatter(df["snr_scale"], df["mse"], s=12, alpha=0.7)
                        ax.set_title("MSE vs SNR scale")
                        ax.set_xlabel("snr_scale")
                        ax.set_ylabel("MSE")
                        fig.tight_layout()
                        p = os.path.join(plots_dir, "mse_vs_snr.png")
                        fig.savefig(p, dpi=150)
                        plt.close(fig)
                        plots["mse_vs_snr"] = p
                except Exception:
                    pass

            self.results = {
                "manifest": man_path,
                "manifest_enriched": man_enriched,
                "summary_csv": out_path,
                "plots": plots if plots else None,
            }
            self.progress = 1.0
            self.stage = "completed"
            self._log("Evaluation completed")
            self.status = "completed"
        except Exception as e:
            self.error = str(e)
            self.status = "error"


_JOBS: Dict[str, EvalJob] = {}
_JOBS_LOCK = threading.Lock()

def _register_job(job: "EvalJob") -> None:
    with _JOBS_LOCK:
        _JOBS[job.id] = job

def _get_job(job_id: Optional[str]) -> Optional["EvalJob"]:
    if not job_id:
        return None
    with _JOBS_LOCK:
        return _JOBS.get(job_id)

def _clear_job(job_id: Optional[str]) -> None:
    if not job_id:
        return
    with _JOBS_LOCK:
        _JOBS.pop(job_id, None)

with st.sidebar:
    st.header("Inputs")
    manifest = st.text_input(
        "manifest.csv path",
        value=st.session_state.get("eval_manifest_default", ""),
        help="Path to manifest.csv from bci-sweep",
    )
    compute_signal_metrics = st.checkbox(
        "compute_signal_metrics",
        value=True,
        help="Compute signal metrics per run (PSD slope, PAC, Higuchi FD, LZC, bandpowers)",
    )
    save_plots = st.checkbox(
        "save_plots",
        value=False,
        help="Save plot images under the manifest directory (plots/)",
    )
    out_path = st.text_input(
        "summary out_path (optional)",
        value="",
        help="Optional path to save group summary CSV (default: manifest_dir/summary.csv)",
    )
    # CLI preview
    def _cli_preview() -> str:
        parts = [
            f"{sys.executable} -m phi.cli neuro bci-eval",
            f"--manifest {manifest}" if manifest else "",
        ]
        parts.append("--compute-signal-metrics" if compute_signal_metrics else "--no-compute-signal-metrics")
        parts.append("--save-plots" if save_plots else "--no-save-plots")
        if out_path:
            parts.append(f"--out-path {out_path}")
        return " ".join([p for p in parts if p])
    st.caption("Equivalent CLI command")
    st.code(_cli_preview(), language="bash")
    start_btn = st.button("Start Evaluation (background)", type="primary")


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

params = {
    "manifest": manifest,
    "compute_signal_metrics": bool(compute_signal_metrics),
    "save_plots": bool(save_plots),
    "out_path": out_path if out_path else None,
}
if start_btn and "eval_job_id" not in st.session_state:
    job = EvalJob(params=params)
    _register_job(job)
    st.session_state["eval_job_id"] = job.id

# Job status / controls
job = _get_job(st.session_state.get("eval_job_id"))
if job is not None:
    # Fallback guard: if thread finished but status wasn't updated due to a stale rerun
    try:
        if job.status == "running" and hasattr(job, "_thread") and not job._thread.is_alive():
            if getattr(job, "error", None):
                job.status = "error"
            elif getattr(job, "_cancel_evt", None) and job._cancel_evt.is_set():
                job.status = "cancelled"
            else:
                job.status = "completed"
                job.stage = getattr(job, "stage", "completed") or "completed"
    except Exception:
        pass
    st.subheader("Evaluation Job Status")
    st.write({"job_id": job.id, "status": job.status, "progress": job.progress})
    st.progress(job.progress if isinstance(job.progress, float) else 0.0)
    # Stage and timing info
    def _fmt_secs(s: float) -> str:
        try:
            s = int(s)
            if s < 60:
                return f"{s}s"
            m, sec = divmod(s, 60)
            if m < 60:
                return f"{m}m{sec:02d}s"
            h, m = divmod(m, 60)
            return f"{h}h{m:02d}m{sec:02d}s"
        except Exception:
            return "-"

    now_ts = time.time()
    elapsed = now_ts - getattr(job, "start_time", now_ts)
    eta = None
    if isinstance(job.progress, float) and job.progress > 0 and job.status == "running":
        try:
            eta = (elapsed / job.progress) - elapsed
        except Exception:
            eta = None
    st.caption(f"Stage: {getattr(job, 'stage', '-')}")
    st.caption(f"Elapsed: {_fmt_secs(elapsed)}" + (f" • ETA: {_fmt_secs(eta)}" if eta else ""))
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Refresh status"):
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
    if c2.button("Cancel evaluation", disabled=(job.status != "running")):
        job.cancel()
    # Auto-refresh toggle
    auto_refresh = c4.checkbox("Auto-refresh", value=True, key="eval_auto_refresh")
    if job.status in ("completed", "cancelled", "error"):
        if job.error:
            st.error(job.error)
        res = job.results or {}
        out_path_eff = res.get("summary_csv")
        man_enriched = res.get("manifest_enriched")
        man_path = res.get("manifest")
        plots = res.get("plots")
        # Make the manifest sticky as default for convenience
        if man_path and os.path.exists(man_path):
            st.session_state["eval_manifest_default"] = man_path

        # Show summary preview if available
        if out_path_eff and os.path.exists(out_path_eff):
            try:
                df_summary = pd.read_csv(out_path_eff)
                st.success("Evaluation complete" if job.status == "completed" else f"Job {job.status}")
                st.write(f"Summary CSV: {out_path_eff}")
                st.dataframe(df_summary.head(1000))
                with open(out_path_eff, "rb") as f:
                    st.download_button("Download summary.csv", data=f.read(), file_name="summary.csv", mime="text/csv")
            except Exception:
                pass
        # Enriched manifest download
        if compute_signal_metrics and man_enriched and os.path.exists(man_enriched):
            try:
                with open(man_enriched, "rb") as f:
                    st.download_button("Download manifest_enriched.csv", data=f.read(), file_name="manifest_enriched.csv", mime="text/csv")
            except Exception:
                pass
        # Plots downloads
        if plots:
            st.subheader("Saved plot files")
            st.json(plots)
            c_plots = st.columns(min(3, len(plots)))
            idx = 0
            for name, path in plots.items():
                try:
                    if path and os.path.exists(path):
                        with open(path, "rb") as f:
                            c_plots[idx % len(c_plots)].download_button(
                                f"Download {name}.png", data=f.read(), file_name=f"{name}.png", mime="image/png"
                            )
                except Exception:
                    pass
                idx += 1
    with st.expander("Logs", expanded=False):
        try:
            if getattr(job, "logs", None):
                st.code("\n".join(job.logs[-200:]))
            else:
                st.write("No logs yet.")
        except Exception:
            pass
    if c3.button("Clear job"):
        _clear_job(job.id)
        st.session_state.pop("eval_job_id", None)
    # Lightweight auto-refresh while running
    if auto_refresh and job.status == "running":
        try:
            time.sleep(1.0)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
        except Exception:
            pass
else:
    st.info("Provide manifest and options in the sidebar and start evaluation.")
