import os
import sys
import time
import pathlib
from itertools import product
import threading
import uuid
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import streamlit as st

# Ensure project root on path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phi.neuro import bci as bci_mod  # noqa: E402

st.set_page_config(page_title="BCI Sweep", page_icon="🧠", layout="wide")
st.title("Neuro BCI • Parameter Sweep")
st.markdown(
    """
    How to use:
    - Configure base and grid parameters in the sidebar.
    - Click "Start Sweep (background)" to launch. Use Refresh or Auto-refresh for live updates.
    - On completion, a `manifest.csv` is written; the Eval page will prefill this path automatically.
    - Use the CLI preview to copy the equivalent command.
    """
)

# ---------------------------- Background Jobs ---------------------------- #

@st.cache_resource
def _jobs_state() -> Dict[str, Any]:
    """Cached registry to persist jobs across Streamlit reruns."""
    return {"lock": threading.Lock(), "jobs": {}}

class SweepJob:
    def __init__(self, combos: List[Tuple[int, float, float, float, float, str]], params: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.status = "running"  # running|completed|cancelled|error
        self.progress = 0.0
        self.total = max(1, len(combos))
        self.current = 0
        self.rows: List[Dict[str, Any]] = []
        self.manifest_path: Optional[str] = None
        self.error: Optional[str] = None
        self.out_root: str = str(params["out_root"])  # ensure str
        self.start_time = time.time()
        self.last_beat = self.start_time
        self.stage: str = "initializing"
        self.logs: List[str] = []
        self._cancel_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(combos, params), daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_evt.set()

    def _log(self, msg: str) -> None:
        try:
            ts = time.strftime("%H:%M:%S")
            self.logs.append(f"[{ts}] {msg}")
        except Exception:
            self.logs.append(str(msg))

    def _run(self, combos: List[Tuple[int, float, float, float, float, str]], params: Dict[str, Any]) -> None:
        try:
            os.makedirs(self.out_root, exist_ok=True)
            self.stage = "preparing output"
            self._log(f"Sweep start: {self.total} run(s), out_root={self.out_root}")
            self.last_beat = time.time()

            def make_scheduler(name: str):
                n = name.lower()
                if n == "constant":
                    return bci_mod.ConstantScheduler(v=float(params["const_v"]))
                if n == "cosine":
                    return bci_mod.CosineScheduler(
                        period=int(params["cos_period"]), min_v=float(params["cos_min"]), max_v=float(params["cos_max"]) 
                    )
                if n == "linear":
                    return bci_mod.LinearScheduler(
                        start_v=float(params["lin_start"]), end_v=float(params["lin_end"]), duration=int(params["lin_duration"]) 
                    )
                if n == "step":
                    return bci_mod.StepScheduler(
                        initial=float(params["step_initial"]), gamma=float(params["step_gamma"]), period=int(params["step_period"]) 
                    )
                return bci_mod.CosineWithPhiRestarts(
                    T0=int(params["phi_T0"]), phi=float(params["phi_val"]), min_v=float(params["phi_min"]), max_v=float(params["phi_max"]) 
                )

            for idx, (seed_v, pn, dr, snr, ns, sch_name) in enumerate(combos, start=1):
                if self._cancel_evt.is_set():
                    self.status = "cancelled"
                    break

                run_dir = os.path.join(self.out_root, f"run_{idx:04d}")
                os.makedirs(run_dir, exist_ok=True)

                cfg = bci_mod.BCIConfig(
                    fs=float(params["fs"]), window_sec=float(params["window_sec"]), steps=int(params["steps"]), seed=int(seed_v),
                    process_noise=float(pn), drift=float(dr), ctrl_effect=float(params["ctrl_effect"]), base_lr=float(params["base_lr"]), base_gain=float(params["base_gain"]),
                    noise_std=float(ns), snr_scale=float(snr), theta_hz=float(params["theta_hz"]), gamma_hz=float(params["gamma_hz"]),
                )
                sch = make_scheduler(str(sch_name))
                self.stage = f"running {idx}/{self.total}"
                run_start = time.time()
                self._log(
                    f"Run {idx}/{self.total} start: seed={seed_v}, pn={pn}, dr={dr}, snr={snr}, ns={ns}, scheduler={sch_name}"
                )
                self.last_beat = time.time()
                def _on_step(t: int) -> None:
                    # heartbeat
                    self.last_beat = time.time()
                    # cooperative cancel
                    if self._cancel_evt.is_set():
                        raise bci_mod.SimulationInterrupt("cancelled")
                    # per-run timeout
                    mrs = float(params.get("max_run_sec", 0.0) or 0.0)
                    if mrs > 0.0 and (time.time() - run_start) > mrs:
                        raise bci_mod.SimulationInterrupt("timeout")

                try:
                    logs = bci_mod.simulate(
                        cfg, scheduler=sch, out_dir=run_dir,
                        save_features=bool(params["save_features"]), save_windows=bool(params["save_windows"]), save_config=bool(params["save_config"]),
                        on_step=_on_step,
                    )
                    self.last_beat = time.time()
                    summary = {k: float(v[0]) for k, v in logs.items() if k in ("mse", "mae", "ttc")}
                    row = {
                        "run": idx, "run_dir": run_dir, "seed": seed_v,
                        "process_noise": pn, "drift": dr, "snr_scale": snr, "noise_std": ns,
                        "scheduler": sch_name, **summary, "status": "ok",
                    }
                    self.rows.append(row)
                    elapsed = time.time() - run_start
                    self._log(f"Run {idx} done in {elapsed:.2f}s: {summary}")
                    self.current = idx
                    self.progress = idx / float(self.total)
                except bci_mod.SimulationInterrupt as si:
                    # Distinguish timeout vs user cancel
                    elapsed = time.time() - run_start
                    mrs = float(params.get('max_run_sec', 0.0) or 0.0)
                    if self._cancel_evt.is_set() or (str(si).lower().strip() == "cancelled"):
                        self._log(f"Run {idx} cancelled by user after {elapsed:.2f}s")
                        self.status = "cancelled"
                        break
                    else:
                        self._log(f"Run {idx} timeout at {elapsed:.2f}s (limit={mrs}s); recording NaN metrics and continuing")
                        summary = {"mse": float('nan'), "mae": float('nan'), "ttc": float('nan')}
                        row = {
                            "run": idx, "run_dir": run_dir, "seed": seed_v,
                            "process_noise": pn, "drift": dr, "snr_scale": snr, "noise_std": ns,
                            "scheduler": sch_name, **summary, "status": "timeout",
                        }
                        self.rows.append(row)
                        self.current = idx
                        self.progress = idx / float(self.total)

            # Write manifest if not cancelled
            if self.status != "cancelled":
                self.stage = "writing manifest"
                df_rows = pd.DataFrame(self.rows)
                self.manifest_path = os.path.join(self.out_root, "manifest.csv")
                self._log(f"Writing manifest.csv with {len(self.rows)} rows to {self.manifest_path}")
                df_rows.to_csv(self.manifest_path, index=False)
                self.last_beat = time.time()
                self.stage = "completed"
                job_elapsed = time.time() - self.start_time
                self._log(f"Sweep completed in {job_elapsed:.2f}s")
                self.status = "completed"
        except Exception as e:
            self.error = str(e)
            self.status = "error"

def _register_job(job: SweepJob) -> None:
    state = _jobs_state()
    with state["lock"]:
        state["jobs"][job.id] = job

def _get_job(job_id: Optional[str]) -> Optional[SweepJob]:
    if not job_id:
        return None
    state = _jobs_state()
    with state["lock"]:
        return state["jobs"].get(job_id)

def _clear_job(job_id: Optional[str]) -> None:
    if not job_id:
        return
    state = _jobs_state()
    with state["lock"]:
        state["jobs"].pop(job_id, None)

with st.sidebar:
    st.header("Base Settings")
    steps = st.number_input("steps", min_value=10, max_value=100000, value=300, step=10, help="Number of closed-loop steps")
    fs = st.number_input("fs (Hz)", min_value=16.0, max_value=4096.0, value=256.0, step=16.0, help="Sampling rate (Hz)")
    window_sec = st.number_input("window_sec", min_value=0.05, max_value=5.0, value=1.0, step=0.05, help="Window duration per step (sec)")
    base_lr = st.number_input("base_lr", min_value=0.0001, max_value=1.0, value=0.05, step=0.005, format="%0.4f", help="Decoder base learning rate")
    base_gain = st.number_input("base_gain", min_value=0.0, max_value=5.0, value=0.5, step=0.1, help="Controller base gain")
    ctrl_effect = st.number_input("ctrl_effect", min_value=0.0, max_value=5.0, value=0.05, step=0.01, format="%0.4f", help="Control effect on latent (environment sensitivity)")
    theta_hz = st.number_input("theta_hz", min_value=0.1, max_value=20.0, value=6.0, step=0.1, help="Theta oscillation frequency (Hz)")
    gamma_hz = st.number_input("gamma_hz", min_value=10.0, max_value=200.0, value=40.0, step=1.0, help="Gamma oscillation frequency (Hz)")

    st.header("Grid Parameters")
    seeds = st.text_input("seeds (comma)", value="1,2", help="Comma-separated seed integers")
    process_noise = st.text_input("process_noise (comma)", value="0.02", help="Comma-separated latent process noise values")
    drift = st.text_input("drift (comma)", value="0.001", help="Comma-separated latent slow drift magnitudes")
    snr_scale = st.text_input("snr_scale (comma)", value="0.6", help="Comma-separated SNR scale values")
    noise_std = st.text_input("noise_std (comma)", value="1.0", help="Comma-separated observation noise std values")

    schedulers = st.multiselect("schedulers", options=["cosine_phi", "cosine", "constant", "linear", "step"], default=["cosine_phi"], help="Scheduler types to include in sweep") 

    st.header("Scheduler Params")
    const_v = st.number_input("const_v", min_value=0.0, max_value=5.0, value=1.0, step=0.1, help="Constant scheduler value (if scheduler=constant)")
    cos_period = st.number_input("cos_period", min_value=1, max_value=100000, value=200, step=1, help="Cosine period (steps)")
    cos_min = st.number_input("cos_min", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="Cosine min value")
    cos_max = st.number_input("cos_max", min_value=0.0, max_value=5.0, value=1.0, step=0.05, help="Cosine max value")
    phi_T0 = st.number_input("phi_T0", min_value=1, max_value=100000, value=200, step=1, help="Initial period for φ-restarts")
    phi_val = st.number_input("phi", min_value=1.0, max_value=3.0, value=1.618, step=0.001, format="%0.3f", help="Golden ratio factor for period growth")
    phi_min = st.number_input("phi_min", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="Min value for φ-restarts scheduler")
    phi_max = st.number_input("phi_max", min_value=0.0, max_value=5.0, value=1.0, step=0.05, help="Max value for φ-restarts scheduler")
    lin_start = st.number_input("lin_start", min_value=0.0, max_value=5.0, value=1.0, step=0.1, help="Linear start value (if scheduler=linear)")
    lin_end = st.number_input("lin_end", min_value=0.0, max_value=5.0, value=0.2, step=0.05, help="Linear end value (if scheduler=linear)")
    lin_duration = st.number_input("lin_duration", min_value=1, max_value=100000, value=500, step=1, help="Linear duration in steps")
    step_initial = st.number_input("step_initial", min_value=0.0, max_value=5.0, value=1.0, step=0.1, help="Initial value (if scheduler=step)")
    step_gamma = st.number_input("step_gamma", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Multiplicative decay factor per period")
    step_period = st.number_input("step_period", min_value=1, max_value=100000, value=200, step=1, help="Period in steps for step schedule")

    st.header("Outputs")
    default_out_root = ROOT / "out" / f"bci_sweep_{int(time.time())}"
    out_root = st.text_input("out_root", value=str(default_out_root), help="Root directory to write runs and manifest.csv")
    save_features = st.checkbox("save_features", value=True, help="Save per-step feature vectors to 'bci_features.csv'")
    save_windows = st.checkbox("save_windows", value=False, help="Save raw signal windows to 'bci_windows.npz'")
    save_config = st.checkbox("save_config", value=True, help="Write 'bci_config.json' with metadata and feature names")
    st.header("Job Controls")
    max_run_sec = st.number_input("max_run_sec (0 = off)", min_value=0.0, max_value=36000.0, value=0.0, step=1.0, help="Maximum allowed seconds per run before timing out")

    # CLI preview (reflects current sidebar values)
    def _cli_preview() -> str:
        parts: List[str] = [
            f"{sys.executable} -m phi.cli neuro bci-sweep",
            f"--steps {int(steps)}",
            f"--fs {float(fs)}",
            f"--window-sec {float(window_sec)}",
            f"--base-lr {float(base_lr)}",
            f"--base-gain {float(base_gain)}",
            f"--ctrl-effect {float(ctrl_effect)}",
            f"--theta-hz {float(theta_hz)}",
            f"--gamma-hz {float(gamma_hz)}",
        ]
        try:
            seeds_v = [int(x) for x in seeds.split(",") if x.strip()]
            pn_v = [float(x) for x in process_noise.split(",") if x.strip()]
            dr_v = [float(x) for x in drift.split(",") if x.strip()]
            snr_v = [float(x) for x in snr_scale.split(",") if x.strip()]
            ns_v = [float(x) for x in noise_std.split(",") if x.strip()]
        except Exception:
            seeds_v, pn_v, dr_v, snr_v, ns_v = [], [], [], [], []
        for v in seeds_v:
            parts += [f"--seeds {int(v)}"]
        for v in pn_v:
            parts += [f"--process-noise {float(v)}"]
        for v in dr_v:
            parts += [f"--drift {float(v)}"]
        for v in snr_v:
            parts += [f"--snr-scale {float(v)}"]
        for v in ns_v:
            parts += [f"--noise-std {float(v)}"]
        for sch in schedulers:
            parts += [f"--schedulers {sch}"]
        # Scheduler params: include only those relevant to the selected schedulers
        sel = set([str(s).lower() for s in schedulers])
        if "constant" in sel:
            parts += [f"--const-v {float(const_v)}"]
        if "cosine" in sel:
            parts += [
                f"--cos-period {int(cos_period)}",
                f"--cos-min {float(cos_min)}",
                f"--cos-max {float(cos_max)}",
            ]
        if "cosine_phi" in sel:
            parts += [
                f"--phi-T0 {int(phi_T0)}",
                f"--phi {float(phi_val)}",
                f"--phi-min {float(phi_min)}",
                f"--phi-max {float(phi_max)}",
            ]
        if "linear" in sel:
            parts += [
                f"--lin-start {float(lin_start)}",
                f"--lin-end {float(lin_end)}",
                f"--lin-duration {int(lin_duration)}",
            ]
        if "step" in sel:
            parts += [
                f"--step-initial {float(step_initial)}",
                f"--step-gamma {float(step_gamma)}",
                f"--step-period {int(step_period)}",
            ]
        parts += ["--save-features" if save_features else "--no-save-features"]
        parts += ["--save-windows" if save_windows else "--no-save-windows"]
        parts += ["--save-config" if save_config else "--no-save-config"]
        parts += [f"--max-run-sec {float(max_run_sec)}"]
        if out_root:
            parts += [f"--out-root {out_root}"]
        return " ".join(parts)

    st.caption("Equivalent CLI command")
    st.code(_cli_preview(), language="bash")

    start_btn = st.button("Start Sweep (background)", type="primary")


def _parse_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]

def _parse_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def _make_scheduler(name: str):
    n = name.lower()
    if n == "constant":
        return bci_mod.ConstantScheduler(v=float(const_v))
    if n == "cosine":
        return bci_mod.CosineScheduler(period=int(cos_period), min_v=float(cos_min), max_v=float(cos_max))
    if n == "linear":
        return bci_mod.LinearScheduler(start_v=float(lin_start), end_v=float(lin_end), duration=int(lin_duration))
    if n == "step":
        return bci_mod.StepScheduler(initial=float(step_initial), gamma=float(step_gamma), period=int(step_period))
    return bci_mod.CosineWithPhiRestarts(T0=int(phi_T0), phi=float(phi_val), min_v=float(phi_min), max_v=float(phi_max))


seeds_v = _parse_ints(seeds)
pn_v = _parse_floats(process_noise)
dr_v = _parse_floats(drift)
snr_v = _parse_floats(snr_scale)
ns_v = _parse_floats(noise_std)
combos_preview = list(product(seeds_v, pn_v, dr_v, snr_v, ns_v, schedulers))
st.info(f"Planned total runs: {len(combos_preview)}")

# Start a new background job
if start_btn and "sweep_job_id" not in st.session_state:
    params = {
        "steps": int(steps),
        "fs": float(fs),
        "window_sec": float(window_sec),
        "base_lr": float(base_lr),
        "base_gain": float(base_gain),
        "ctrl_effect": float(ctrl_effect),
        "theta_hz": float(theta_hz),
        "gamma_hz": float(gamma_hz),
        "const_v": float(const_v),
        "cos_period": int(cos_period),
        "cos_min": float(cos_min),
        "cos_max": float(cos_max),
        "phi_T0": int(phi_T0),
        "phi_val": float(phi_val),
        "phi_min": float(phi_min),
        "phi_max": float(phi_max),
        "lin_start": float(lin_start),
        "lin_end": float(lin_end),
        "lin_duration": int(lin_duration),
        "step_initial": float(step_initial),
        "step_gamma": float(step_gamma),
        "step_period": int(step_period),
        "save_features": bool(save_features),
        "save_windows": bool(save_windows),
        "save_config": bool(save_config),
        "out_root": str(out_root),
        "max_run_sec": float(max_run_sec),
    }
    combos = combos_preview
    job = SweepJob(combos=combos, params=params)
    _register_job(job)
    st.session_state["sweep_job_id"] = job.id

# Job status / controls
job = _get_job(st.session_state.get("sweep_job_id"))
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
    st.subheader("Sweep Job Status")
    st.write({"job_id": job.id, "status": job.status, "progress": job.progress, "current": job.current, "total": job.total})
    st.progress(job.progress)
    # Stage and timing
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
    # Heartbeat-based stall diagnostics
    last_beat_age = now_ts - getattr(job, "last_beat", getattr(job, "start_time", now_ts))
    st.caption(f"Last update: {_fmt_secs(last_beat_age)} ago")
    if job.status == "running" and last_beat_age > 60:
        st.warning(f"No updates for {int(last_beat_age)}s — job may be stalled.")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Refresh status"):
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
    if c2.button("Cancel sweep", disabled=(job.status != "running")):
        job.cancel()
    # Auto-refresh toggle
    auto_refresh = c4.checkbox("Auto-refresh", value=True, key="sweep_auto_refresh")
    if job.status in ("completed", "cancelled", "error"):
        # Show manifest if available
        if job.manifest_path and os.path.exists(job.manifest_path):
            st.success("Sweep complete" if job.status == "completed" else f"Job {job.status}")
            # Make the manifest sticky as default for the Eval page
            try:
                st.session_state["eval_manifest_default"] = job.manifest_path
                st.caption("Default set for Eval page: open 'BCI Eval' to see it pre-filled.")
            except Exception:
                pass
            st.write(f"Manifest: {job.manifest_path}")
            try:
                df_rows = pd.read_csv(job.manifest_path)
                st.dataframe(df_rows.head(1000))
                st.download_button(
                    label="Download manifest.csv",
                    data=df_rows.to_csv(index=False).encode("utf-8"),
                    file_name="manifest.csv",
                    mime="text/csv",
                )
                # Retry timed-out runs helper
                timeout_count = 0
                try:
                    if "status" in df_rows.columns:
                        timeout_count = int((df_rows["status"] == "timeout").sum())
                except Exception:
                    timeout_count = 0
                if timeout_count > 0:
                    st.info(f"Timed-out runs: {timeout_count}")
                    if st.button("Retry timed-out runs", key="retry_timeouts_btn"):
                        try:
                            tdf = df_rows[df_rows["status"] == "timeout"]
                            combos = list(zip(
                                tdf["seed"].astype(int).tolist(),
                                tdf["process_noise"].astype(float).tolist(),
                                tdf["drift"].astype(float).tolist(),
                                tdf["snr_scale"].astype(float).tolist(),
                                tdf["noise_std"].astype(float).tolist(),
                                tdf["scheduler"].astype(str).tolist(),
                            ))
                            if len(combos) == 0:
                                st.info("No timed-out runs to retry.")
                            else:
                                new_out = str(job.out_root) + f"_retry_{int(time.time())}"
                                params2 = {
                                    "steps": int(steps),
                                    "fs": float(fs),
                                    "window_sec": float(window_sec),
                                    "base_lr": float(base_lr),
                                    "base_gain": float(base_gain),
                                    "ctrl_effect": float(ctrl_effect),
                                    "theta_hz": float(theta_hz),
                                    "gamma_hz": float(gamma_hz),
                                    "const_v": float(const_v),
                                    "cos_period": int(cos_period),
                                    "cos_min": float(cos_min),
                                    "cos_max": float(cos_max),
                                    "phi_T0": int(phi_T0),
                                    "phi_val": float(phi_val),
                                    "phi_min": float(phi_min),
                                    "phi_max": float(phi_max),
                                    "lin_start": float(lin_start),
                                    "lin_end": float(lin_end),
                                    "lin_duration": int(lin_duration),
                                    "step_initial": float(step_initial),
                                    "step_gamma": float(step_gamma),
                                    "step_period": int(step_period),
                                    "save_features": bool(save_features),
                                    "save_windows": bool(save_windows),
                                    "save_config": bool(save_config),
                                    "out_root": str(new_out),
                                    "max_run_sec": float(max_run_sec),
                                }
                                job2 = SweepJob(combos=[tuple(x) for x in combos], params=params2)
                                _register_job(job2)
                                st.session_state["sweep_job_id"] = job2.id
                                st.success(f"Retry job started with {len(combos)} timed-out run(s).")
                                try:
                                    st.rerun()
                                except Exception:
                                    st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Failed to start retry: {e}")
                else:
                    st.info("No timed-out runs to retry.")
            except Exception:
                pass
        if job.error:
            st.error(job.error)
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
        st.session_state.pop("sweep_job_id", None)
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
    st.info("Configure grid and start a sweep from the sidebar.")
