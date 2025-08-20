import os
import sys
import time
import pathlib
from itertools import product

import pandas as pd
import streamlit as st

# Ensure project root on path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phi.neuro import bci as bci_mod  # noqa: E402

st.set_page_config(page_title="BCI Sweep", page_icon="🧠", layout="wide")
st.title("Neuro BCI • Parameter Sweep")

with st.sidebar:
    st.header("Base Settings")
    steps = st.number_input("steps", min_value=10, max_value=100000, value=300, step=10)
    fs = st.number_input("fs (Hz)", min_value=16.0, max_value=4096.0, value=256.0, step=16.0)
    window_sec = st.number_input("window_sec", min_value=0.05, max_value=5.0, value=1.0, step=0.05)
    base_lr = st.number_input("base_lr", min_value=0.0001, max_value=1.0, value=0.05, step=0.005, format="%0.4f")
    base_gain = st.number_input("base_gain", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

    st.header("Grid Parameters")
    seeds = st.text_input("seeds (comma)", value="1,2")
    process_noise = st.text_input("process_noise (comma)", value="0.02")
    drift = st.text_input("drift (comma)", value="0.001")
    snr_scale = st.text_input("snr_scale (comma)", value="0.6")
    noise_std = st.text_input("noise_std (comma)", value="1.0")

    schedulers = st.multiselect("schedulers", options=["cosine_phi", "cosine", "constant"], default=["cosine_phi"]) 

    st.header("Scheduler Params")
    const_v = st.number_input("const_v", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    cos_period = st.number_input("cos_period", min_value=1, max_value=100000, value=200, step=1)
    cos_min = st.number_input("cos_min", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    cos_max = st.number_input("cos_max", min_value=0.0, max_value=5.0, value=1.0, step=0.05)
    phi_T0 = st.number_input("phi_T0", min_value=1, max_value=100000, value=200, step=1)
    phi_val = st.number_input("phi", min_value=1.0, max_value=3.0, value=1.618, step=0.001, format="%0.3f")
    phi_min = st.number_input("phi_min", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    phi_max = st.number_input("phi_max", min_value=0.0, max_value=5.0, value=1.0, step=0.05)

    st.header("Outputs")
    default_out_root = ROOT / "out" / f"bci_sweep_{int(time.time())}"
    out_root = st.text_input("out_root", value=str(default_out_root))
    save_features = st.checkbox("save_features", value=True)
    save_windows = st.checkbox("save_windows", value=False)
    save_config = st.checkbox("save_config", value=True)

    run_btn = st.button("Run Sweep", type="primary")


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
    return bci_mod.CosineWithPhiRestarts(T0=int(phi_T0), phi=float(phi_val), min_v=float(phi_min), max_v=float(phi_max))


if run_btn:
    os.makedirs(out_root, exist_ok=True)

    seeds_v = _parse_ints(seeds)
    pn_v = _parse_floats(process_noise)
    dr_v = _parse_floats(drift)
    snr_v = _parse_floats(snr_scale)
    ns_v = _parse_floats(noise_std)

    rows = []
    combos = list(product(seeds_v, pn_v, dr_v, snr_v, ns_v, schedulers))
    st.info(f"Planned total runs: {len(combos)}")
    progress = st.progress(0.0, text="Starting sweep...")

    for idx, (seed_v, pn, dr, snr, ns, sch_name) in enumerate(combos, start=1):
        run_dir = os.path.join(out_root, f"run_{idx:04d}")
        os.makedirs(run_dir, exist_ok=True)

        cfg = bci_mod.BCIConfig(
            fs=float(fs), window_sec=float(window_sec), steps=int(steps), seed=int(seed_v),
            process_noise=float(pn), drift=float(dr), ctrl_effect=0.05, base_lr=float(base_lr), base_gain=float(base_gain),
            noise_std=float(ns), snr_scale=float(snr),
        )
        sch = _make_scheduler(sch_name)
        logs = bci_mod.simulate(
            cfg, scheduler=sch, out_dir=run_dir,
            save_features=bool(save_features), save_windows=bool(save_windows), save_config=bool(save_config),
        )
        summary = {k: float(v[0]) for k, v in logs.items() if k in ("mse", "mae", "ttc")}
        row = {
            "run": idx, "run_dir": run_dir, "seed": seed_v,
            "process_noise": pn, "drift": dr, "snr_scale": snr, "noise_std": ns,
            "scheduler": sch_name, **summary,
        }
        rows.append(row)

        progress.progress(idx / len(combos), text=f"Completed {idx}/{len(combos)}")

    manifest_path = os.path.join(out_root, "manifest.csv")
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(manifest_path, index=False)

    st.success("Sweep complete")
    st.write(f"Manifest: {manifest_path}")
    st.dataframe(df_rows.head(1000))

    # Offer manifest download
    csv_bytes = df_rows.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download manifest.csv",
        data=csv_bytes,
        file_name="manifest.csv",
        mime="text/csv",
    )
else:
    st.info("Configure grid and click Run Sweep.")
