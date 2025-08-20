import os
import sys
import time
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phi.neuro import bci as bci_mod  # noqa: E402

st.set_page_config(page_title="BCI Sim", page_icon="🧠", layout="wide")
st.title("Neuro BCI • Simulation")
st.markdown(
    """
    How to use:
    - Configure parameters in the sidebar.
    - Click "Run Simulation" to generate artifacts under `out/bci_sim_.../`.
    - After completion, the Train page will prefill the run directory automatically.
    - Use the CLI preview to copy the equivalent command.
    """
)

with st.sidebar:
    st.header("Simulation Parameters")
    steps = st.number_input("steps", min_value=10, max_value=100000, value=500, step=10, help="Number of closed-loop steps")
    fs = st.number_input("fs (Hz)", min_value=16.0, max_value=4096.0, value=256.0, step=16.0, help="Sampling rate (Hz)")
    window_sec = st.number_input("window_sec", min_value=0.05, max_value=5.0, value=1.0, step=0.05, help="Window duration per step (sec)")
    seed = st.number_input("seed", min_value=0, max_value=1_000_000, value=42, step=1, help="Random seed")
    process_noise = st.number_input("process_noise", min_value=0.0, max_value=1.0, value=0.02, step=0.005, format="%0.4f", help="Latent process noise")
    drift = st.number_input("drift", min_value=0.0, max_value=0.1, value=0.001, step=0.0005, format="%0.4f", help="Latent slow drift magnitude")
    ctrl_effect = st.number_input("ctrl_effect", min_value=0.0, max_value=1.0, value=0.05, step=0.01, help="Control influence on latent")
    # Signal composition
    theta_hz = st.number_input("theta_hz", min_value=0.1, max_value=20.0, value=6.0, step=0.1, help="Theta oscillation frequency (Hz)")
    gamma_hz = st.number_input("gamma_hz", min_value=10.0, max_value=200.0, value=40.0, step=1.0, help="Gamma oscillation frequency (Hz)")
    snr_scale = st.number_input("snr_scale", min_value=0.0, max_value=5.0, value=0.6, step=0.05, help="How strongly latent modulates the gamma envelope")
    noise_std = st.number_input("noise_std", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Observation noise standard deviation")
    base_lr = st.number_input("base_lr", min_value=0.0001, max_value=1.0, value=0.05, step=0.005, format="%0.4f", help="Decoder base learning rate")
    base_gain = st.number_input("base_gain", min_value=0.0, max_value=5.0, value=0.5, step=0.1, help="Controller base gain")

    scheduler = st.selectbox("scheduler", options=["cosine_phi", "cosine", "constant", "linear", "step"], index=0, help="Scheduler type")
    if scheduler == "constant":
        const_v = st.number_input("const_v", min_value=0.0, max_value=5.0, value=1.0, step=0.1, help="Constant scheduler value (if scheduler=constant)")
        cos_period = 200; cos_min = 0.2; cos_max = 1.0; phi_T0 = 200; phi_val = 1.618; phi_min = 0.2; phi_max = 1.0; lin_start = 1.0; lin_end = 0.2; lin_duration = 500; step_initial = 1.0; step_gamma = 0.5; step_period = 200
    elif scheduler == "cosine":
        cos_period = st.number_input("cos_period", min_value=1, max_value=100000, value=200, step=1, help="Cosine period (steps)")
        cos_min = st.number_input("cos_min", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="Cosine min value")
        cos_max = st.number_input("cos_max", min_value=0.0, max_value=5.0, value=1.0, step=0.05, help="Cosine max value")
        const_v = 1.0; phi_T0 = 200; phi_val = 1.618; phi_min = 0.2; phi_max = 1.0; lin_start = 1.0; lin_end = 0.2; lin_duration = 500; step_initial = 1.0; step_gamma = 0.5; step_period = 200
    elif scheduler == "cosine_phi":
        phi_T0 = st.number_input("phi_T0", min_value=1, max_value=100000, value=200, step=1, help="Initial period for φ-restarts")
        phi_val = st.number_input("phi", min_value=1.0, max_value=3.0, value=1.618, step=0.001, format="%0.3f", help="Golden ratio factor for period growth")
        phi_min = st.number_input("phi_min", min_value=0.0, max_value=1.0, value=0.2, step=0.05, help="Min value for φ-restarts scheduler")
        phi_max = st.number_input("phi_max", min_value=0.0, max_value=5.0, value=1.0, step=0.05, help="Max value for φ-restarts scheduler")
        const_v = 1.0; cos_period = 200; cos_min = 0.2; cos_max = 1.0; lin_start = 1.0; lin_end = 0.2; lin_duration = 500; step_initial = 1.0; step_gamma = 0.5; step_period = 200
    elif scheduler == "linear":
        lin_start = st.number_input("lin_start", min_value=0.0, max_value=5.0, value=1.0, step=0.1, help="Linear start value (if scheduler=linear)")
        lin_end = st.number_input("lin_end", min_value=0.0, max_value=5.0, value=0.2, step=0.05, help="Linear end value (if scheduler=linear)")
        lin_duration = st.number_input("lin_duration", min_value=1, max_value=100000, value=500, step=1, help="Linear duration in steps")
        const_v = 1.0; cos_period = 200; cos_min = 0.2; cos_max = 1.0; phi_T0 = 200; phi_val = 1.618; phi_min = 0.2; phi_max = 1.0; step_initial = 1.0; step_gamma = 0.5; step_period = 200
    elif scheduler == "step":
        step_initial = st.number_input("step_initial", min_value=0.0, max_value=5.0, value=1.0, step=0.1, help="Initial value (if scheduler=step)")
        step_gamma = st.number_input("step_gamma", min_value=0.0, max_value=1.0, value=0.5, step=0.05, help="Multiplicative decay factor per period")
        step_period = st.number_input("step_period", min_value=1, max_value=100000, value=200, step=1, help="Period in steps for step schedule")
        const_v = 1.0; cos_period = 200; cos_min = 0.2; cos_max = 1.0; phi_T0 = 200; phi_val = 1.618; phi_min = 0.2; phi_max = 1.0; lin_start = 1.0; lin_end = 0.2; lin_duration = 500

    st.divider()
    st.header("Outputs")
    default_out_dir = ROOT / "out" / f"bci_sim_{int(time.time())}"
    out_dir = st.text_input("out_dir", value=str(default_out_dir), help="Optional output directory to save logs")
    save_features = st.checkbox("save_features", value=False, help="Save per-step feature vectors to 'bci_features.csv'")
    save_windows = st.checkbox("save_windows", value=False, help="Save raw signal windows to 'bci_windows.npz'")
    save_config = st.checkbox("save_config", value=True, help="Write 'bci_config.json' with metadata and feature names")

    # CLI preview
    def _cli_preview() -> str:
        parts = [
            f"{sys.executable} -m phi.cli neuro bci-sim",
            f"--steps {int(steps)}",
            f"--fs {float(fs)}",
            f"--window-sec {float(window_sec)}",
            f"--seed {int(seed)}",
            f"--process-noise {float(process_noise)}",
            f"--drift {float(drift)}",
            f"--ctrl-effect {float(ctrl_effect)}",
            f"--theta-hz {float(theta_hz)}",
            f"--gamma-hz {float(gamma_hz)}",
            f"--snr-scale {float(snr_scale)}",
            f"--noise-std {float(noise_std)}",
            f"--base-lr {float(base_lr)}",
            f"--base-gain {float(base_gain)}",
            f"--scheduler {scheduler}",
        ]

        if scheduler == "constant":
            parts += [f"--const-v {float(const_v)}"]
        elif scheduler == "cosine":
            parts += [f"--cos-period {int(cos_period)}", f"--cos-min {float(cos_min)}", f"--cos-max {float(cos_max)}"]
        elif scheduler == "cosine_phi":
            parts += [f"--phi-T0 {int(phi_T0)}", f"--phi {float(phi_val)}", f"--phi-min {float(phi_min)}", f"--phi-max {float(phi_max)}"]
        elif scheduler == "linear":
            parts += [
                f"--lin-start {float(lin_start)}",
                f"--lin-end {float(lin_end)}",
                f"--lin-duration {int(lin_duration)}",
            ]
        elif scheduler == "step":
            parts += [
                f"--step-initial {float(step_initial)}",
                f"--step-gamma {float(step_gamma)}",
                f"--step-period {int(step_period)}",
            ]

        if out_dir:
            parts += [f"--out-dir {out_dir}"]
        if save_features:
            parts += ["--save-features"]
        if save_windows:
            parts += ["--save-windows"]
        if not save_config:
            parts += ["--no-save-config"]
        return " ".join(parts)

    st.caption("Equivalent CLI command")
    st.code(_cli_preview(), language="bash")

    run_btn = st.button("Run Simulation", type="primary")

def _make_scheduler(name: str):
    n = name.lower()
    if n == "constant":
        return bci_mod.ConstantScheduler(v=float(const_v))
    if n == "cosine":
        return bci_mod.CosineScheduler(period=int(cos_period), min_v=float(cos_min), max_v=float(cos_max))
    if n == "cosine_phi":
        return bci_mod.CosineWithPhiRestarts(T0=int(phi_T0), phi=float(phi_val), min_v=float(phi_min), max_v=float(phi_max))
    if n == "linear":
        return bci_mod.LinearScheduler(start_v=float(lin_start), end_v=float(lin_end), duration=int(lin_duration))
    if n == "step":
        return bci_mod.StepScheduler(initial=float(step_initial), gamma=float(step_gamma), period=int(step_period))
    return bci_mod.CosineWithPhiRestarts(T0=int(phi_T0), phi=float(phi_val), min_v=float(phi_min), max_v=float(phi_max))

if run_btn:
    os.makedirs(out_dir, exist_ok=True)
    cfg = bci_mod.BCIConfig(
        fs=float(fs), window_sec=float(window_sec), steps=int(steps), seed=int(seed),
        process_noise=float(process_noise), drift=float(drift), ctrl_effect=float(ctrl_effect),
        base_lr=float(base_lr), base_gain=float(base_gain),
        theta_hz=float(theta_hz), gamma_hz=float(gamma_hz), snr_scale=float(snr_scale), noise_std=float(noise_std),
    )
    sch = _make_scheduler(scheduler)
    # Live progress via on_step heartbeat
    progress = st.progress(0.0)
    status_txt = st.empty()
    def _on_step(t: int):
        try:
            frac = min(1.0, (int(t) + 1) / float(max(1, int(steps))))
            progress.progress(frac)
            if (t % 10) == 0:
                status_txt.caption(f"Step {int(t)}/{int(steps)}")
        except Exception:
            pass
    with st.spinner("Running simulation..."):
        logs = bci_mod.simulate(
            cfg, scheduler=sch, out_dir=out_dir,
            save_features=bool(save_features), save_windows=bool(save_windows), save_config=bool(save_config),
            on_step=_on_step,
        )

    st.success("Simulation complete")
    # Make the run directory sticky as default for the Train page
    try:
        st.session_state["train_data_dir_default"] = out_dir
        st.caption("Default set for Train page: open 'BCI Train' to see it pre-filled.")
    except Exception:
        pass

    # Metrics
    summary = {k: float(v[0]) for k, v in logs.items() if k in ("mse", "mae", "ttc")}
    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{summary.get('mse', float('nan')):.4f}")
    col2.metric("MAE", f"{summary.get('mae', float('nan')):.4f}")
    col3.metric("TTC", f"{summary.get('ttc', float('nan')):.0f}")

    # Time series plot
    T = len(logs["y"]) if isinstance(logs.get("y"), np.ndarray) else 0
    df_plot = pd.DataFrame({
        "t": np.arange(T),
        "y": logs["y"],
        "y_true": logs.get("y_true", np.zeros(T)),
        "y_hat": logs["y_hat"],
        "err": logs["err"],
        "sched": logs["sched"],
    })
    st.line_chart(df_plot.set_index("t")[ ["y", "y_hat", "err", "sched"] ])

    st.subheader("Artifacts")
    st.json({
        "timeseries_csv": os.path.join(out_dir, "bci_timeseries.csv"),
        "features_csv": os.path.join(out_dir, "bci_features.csv") if save_features else None,
        "windows_npz": os.path.join(out_dir, "bci_windows.npz") if save_windows else None,
        "config_json": os.path.join(out_dir, "bci_config.json") if save_config else None,
        "summary_json": os.path.join(out_dir, "bci_summary.json"),
    })

    # Downloads and previews
    ts_path = os.path.join(out_dir, "bci_timeseries.csv")
    feat_path = os.path.join(out_dir, "bci_features.csv")
    win_path = os.path.join(out_dir, "bci_windows.npz")
    cfg_path = os.path.join(out_dir, "bci_config.json")
    sum_path = os.path.join(out_dir, "bci_summary.json")

    st.subheader("Downloads")
    cols = st.columns(3)
    if os.path.exists(ts_path):
        with open(ts_path, "rb") as f:
            cols[0].download_button("Download bci_timeseries.csv", data=f.read(), file_name="bci_timeseries.csv", mime="text/csv")
        try:
            st.dataframe(pd.read_csv(ts_path).head(200))
        except Exception:
            pass
    if save_features and os.path.exists(feat_path):
        with open(feat_path, "rb") as f:
            cols[1].download_button("Download bci_features.csv", data=f.read(), file_name="bci_features.csv", mime="text/csv")
    if save_windows and os.path.exists(win_path):
        with open(win_path, "rb") as f:
            cols[2].download_button("Download bci_windows.npz", data=f.read(), file_name="bci_windows.npz", mime="application/octet-stream")
    # Config and summary
    cols2 = st.columns(2)
    if save_config and os.path.exists(cfg_path):
        with open(cfg_path, "rb") as f:
            cols2[0].download_button("Download bci_config.json", data=f.read(), file_name="bci_config.json", mime="application/json")
    if os.path.exists(sum_path):
        with open(sum_path, "rb") as f:
            cols2[1].download_button("Download bci_summary.json", data=f.read(), file_name="bci_summary.json", mime="application/json")
else:
    st.info("Configure parameters in the sidebar and click Run Simulation.")
