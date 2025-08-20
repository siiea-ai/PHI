import os
import pathlib
import streamlit as st

st.set_page_config(
    page_title="PHI Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("PHI Dashboard")
st.subheader("Interactive UI for all CLI capabilities")

st.markdown(
    """
Welcome to the PHI Dashboard. This app provides a web UI for the PHI CLI functionality.

Core areas (see sidebar Pages):
- Neuro BCI: Sim, Sweep, Eval, Train
- CLI Explorer: run any `phi.cli` command with arguments

Quick start:
1) Use "BCI Sim" to generate a single run and artifacts under `out/bci_sim_.../`.
2) Or use "BCI Sweep" to generate many runs and a `manifest.csv` under `out/bci_sweep_.../`.
3) Use "BCI Train" on a single run directory (features/windows) to fit a baseline model.
4) Use "BCI Eval" on a sweep `manifest.csv` to compute summaries and optional plots.

Outputs are written under the same project `out/` directory by default.
    """
)

# Discover common output roots for quick access
def _discover_runs(base_dir: str = "out"):
    p = pathlib.Path(base_dir)
    if not p.exists():
        return []
    return [str(d) for d in p.glob("**/run_*" ) if d.is_dir()]

def _discover_sim_dirs(base_dir: str = "out"):
    p = pathlib.Path(base_dir)
    if not p.exists():
        return []
    # bci_sim_*
    return sorted([str(d) for d in p.glob("bci_sim_*") if d.is_dir()], key=lambda x: os.path.getmtime(x), reverse=True)

def _discover_sweep_manifests(base_dir: str = "out"):
    p = pathlib.Path(base_dir)
    if not p.exists():
        return []
    # manifest.csv under bci_sweep_*/
    items = []
    for d in p.glob("bci_sweep_*"):
        man = d / "manifest.csv"
        if man.exists():
            items.append(str(man))
    return sorted(items, key=lambda x: os.path.getmtime(x), reverse=True)

with st.expander("Recent BCI outputs (quick shortcuts)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Recent Sim outputs (use in Train page)")
        sims = _discover_sim_dirs("out")
        if sims:
            for i, d in enumerate(sims[:10]):
                st.write(d)
                if st.button("Use in Train page", key=f"use_train_{i}"):
                    st.session_state["train_data_dir_default"] = d
                    st.success("Default set for Train page: open 'BCI Train' to see it pre-filled.")
        else:
            st.info("No Sim outputs found yet. Run a simulation from the 'BCI Sim' page.")
    with c2:
        st.caption("Recent Sweep manifests (use in Eval page)")
        mans = _discover_sweep_manifests("out")
        if mans:
            for i, m in enumerate(mans[:10]):
                st.write(m)
                if st.button("Use in Eval page", key=f"use_eval_{i}"):
                    st.session_state["eval_manifest_default"] = m
                    st.success("Default set for Eval page: open 'BCI Eval' to see it pre-filled.")
        else:
            st.info("No Sweep manifests found yet. Start a sweep from the 'BCI Sweep' page.")
