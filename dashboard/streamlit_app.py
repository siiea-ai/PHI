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

Core areas available via sidebar Pages:
- Neuro BCI: Sim, Sweep, Eval, Train
- CLI Explorer: run any `phi.cli` command with arguments

To run locally:
```
streamlit run dashboard/streamlit_app.py
```
Outputs are written under the same project `out/` directory by default.
    """
)

# Discover common output roots for quick access
def _discover_runs(base_dir: str = "out"):
    p = pathlib.Path(base_dir)
    if not p.exists():
        return []
    return [str(d) for d in p.glob("**/run_*" ) if d.is_dir()]

with st.expander("Recent BCI run directories found under out/", expanded=False):
    runs = _discover_runs("out")
    if runs:
        st.write(runs[:50])
    else:
        st.info("No runs found yet. Use the BCI Sim or Sweep pages to create some.")
