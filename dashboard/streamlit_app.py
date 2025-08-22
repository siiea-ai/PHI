import os
import pathlib
import streamlit as st

st.set_page_config(
    page_title="PHI AI Platform",
    page_icon="🌟",
    layout="wide"
)

st.title("🌟 PHI AI Platform - Complete Model Training & Chat System")
st.markdown("**Discover, Train, and Chat with AI Models using Golden Ratio Optimization**")

st.markdown("""
Welcome to the PHI AI Platform! A comprehensive system for discovering, training, and interacting 
with AI models using golden ratio (φ ≈ 1.618) optimization principles.

## 🚀 Complete AI Workflow

### 🔍 **Model Discovery & Management**
- **Model Browser**: Search and download models from HuggingFace Hub
- **Smart Organization**: Automatic categorization in `/models/raw/` and `/models/trained/`
- **Metadata Tracking**: Complete model information and version control

### 🎯 **PHI-Optimized Training**
- **Training Studio**: Advanced training with golden ratio optimization
- **Custom Save Options**: Flexible model saving with custom naming and locations
- **Real-time Monitoring**: Track training progress and PHI improvements

### 💬 **Interactive Chat Interface**
- **Model Testing**: Chat with your trained models
- **Advanced Settings**: Full control over generation parameters
- **Performance Analytics**: Response time and quality metrics

### 📊 **Legacy Modules**
- Audio, Image, Data processing with PHI algorithms
- BCI Simulation and Cosmos modeling
- Comprehensive analytics and visualization

Navigate using the sidebar to start your AI journey!
""")

# Quick stats and navigation
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Golden Ratio φ", "1.618033988749895")
    if st.button("🔍 Browse Models", use_container_width=True):
        st.switch_page("pages/09_Model_Browser.py")

with col2:
    st.metric("Inverse φ", "0.618033988749895")
    if st.button("🎯 Train Models", use_container_width=True):
        st.switch_page("pages/10_PHI_Training_Studio.py")

with col3:
    st.metric("φ²", "2.618033988749895")
    if st.button("💬 Chat Interface", use_container_width=True):
        st.switch_page("pages/11_Model_Chat.py")

with col4:
    st.metric("φ - 1", "0.618033988749895")
    if st.button("📊 Analytics", use_container_width=True):
        st.switch_page("pages/07_Analytics.py")

with st.expander("Help & How-To", expanded=False):
    st.markdown(
        """
        - __Install dependencies__: `pip install -r requirements.txt`
        - __Audio/Video support__: Install ffmpeg (macOS: `brew install ffmpeg`) for MP3/FLAC/OGG/M4A and some video formats.
        - __Data Hub quick start__:
          1) Open the "Data Hub" page from the sidebar.
          2) Upload an image, audio, video, CSV, or JSON file.
          3) Choose a module (Auto default), set parameters, and click Run.
          4) Browse results in the History tab with previews and downloads.
        - __BCI pages quick start__:
          - Sim: configure parameters, run 1 simulation → artifacts under `out/bci_sim_*`.
          - Sweep: run many sims; see `manifest.csv` under `out/bci_sweep_*`.
          - Train: select a Sim output dir to train a baseline model.
          - Eval: load a Sweep `manifest.csv` to compute summaries.
        - __Where outputs go__: All pages write under `out/` (Data Hub uses `out/datahub/...`).
        - __Troubleshooting__:
          - If audio/video fails to load, verify ffmpeg is installed and on PATH.
          - If imports fail, ensure you run Streamlit from the project root.
        """
    )

def _discover_runs(base_dir: str = "out"):
    p = pathlib.Path(base_dir)
    if not p.exists():
        return []
    return [str(d) for d in p.glob("**/run_*") if d.is_dir()]

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
