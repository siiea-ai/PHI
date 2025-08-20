import os
import sys
import shlex
import subprocess
import pathlib
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]

st.set_page_config(page_title="CLI Explorer", page_icon="💻", layout="wide")
st.title("CLI Explorer • Run any phi.cli command")

st.markdown(
    """
This page allows running any `phi.cli` command as if from the terminal.

Examples:
- `ratio`
- `fractal ai generate --output out/ai.json --input-dim 8 --output-dim 4 --hidden 64,32`
- `neuro bci-sim --out-dir out/run_demo --save-features`

Outputs will show stdout/stderr and exit code.
    """
)

with st.form("cli_form"):
    cmd = st.text_input("Command (without `python -m phi.cli` prefix)", value="neuro bci-sim --steps 100 --out-dir out/cli_explorer_run --save-features")
    submitted = st.form_submit_button("Run Command", type="primary")

if submitted:
    full_cmd = f"{sys.executable} -m phi.cli {cmd}"
    st.write(f"Running: `{full_cmd}`")
    try:
        proc = subprocess.run(shlex.split(full_cmd), capture_output=True, text=True, cwd=str(ROOT))
        st.subheader("Stdout")
        st.code(proc.stdout or "", language="json")
        st.subheader("Stderr")
        st.code(proc.stderr or "", language="text")
        st.write({"returncode": proc.returncode})
        if proc.returncode != 0:
            st.error("Command failed. See stderr above.")
        else:
            st.success("Command completed successfully.")
    except Exception as e:
        st.exception(e)
