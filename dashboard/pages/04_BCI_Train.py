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

from phi.neuro.datasets import load_bci_dataset  # noqa: E402
from sklearn.linear_model import Ridge, Lasso, LinearRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # noqa: E402

st.set_page_config(page_title="BCI Train", page_icon="🧠", layout="wide")
st.title("Neuro BCI • Baseline Training")
st.markdown(
    """
    How to use:
    - Select a BCI run directory (from Sim or Sweep) containing saved features or windows.
    - Choose `mode` to match the artifacts you saved (`features` or `windows`).
    - Pick a baseline model and train/test split options.
    - Click "Start Training (background)" to run. Status and ETA update live; use Auto-refresh if needed.
    - When done, review metrics and optional predictions below.
    """
)

# ---------------------------- Background Job ---------------------------- #
class TrainJob:
    def __init__(self, params: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.status = "running"  # running|completed|cancelled|error
        self.progress = 0.0
        self.error: Optional[str] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.metrics_path: Optional[str] = None
        self.preds_path: Optional[str] = None
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
            data_dir = str(params["data_dir"])
            mode = str(params["mode"]).lower()
            model_name = str(params["model"]).lower()
            alpha = float(params["alpha"])
            test_size = float(params["test_size"])
            random_state = int(params["random_state"])
            save_preds = bool(params["save_preds"])

            if not data_dir or not os.path.isdir(data_dir):
                self.error = "Invalid data_dir"
                self.status = "error"
                return

            self.stage = "loading dataset"
            self._log(f"Loading dataset from {data_dir} (mode={mode})")
            self.progress = 0.05
            use_features = mode == "features"
            use_windows = mode == "windows"
            ds = load_bci_dataset(data_dir, use_features=use_features, use_windows=use_windows)
            X = ds["X_feat"] if use_features else ds["X_win"]
            y = ds["y"]
            if X is None:
                self.error = "Selected mode has no available X in data_dir"
                self.status = "error"
                return

            if self._cancel_evt.is_set():
                self.status = "cancelled"; return
            from sklearn.model_selection import train_test_split  # local import to avoid issues
            from sklearn.linear_model import Ridge, Lasso, LinearRegression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            self.stage = "splitting train/test"
            self._log("Splitting train/test")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=False
            )
            self.progress = 0.2
            if self._cancel_evt.is_set():
                self.status = "cancelled"; return

            if model_name == "ridge":
                clf = Ridge(alpha=alpha)
            elif model_name == "lasso":
                clf = Lasso(alpha=alpha)
            else:
                clf = LinearRegression()

            self.stage = "fitting model"
            self._log(f"Fitting {model_name} model")
            clf.fit(X_train, y_train)
            self.progress = 0.6
            if self._cancel_evt.is_set():
                self.status = "cancelled"; return

            self.stage = "predicting"
            self._log("Predicting on train/test")
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)

            self.stage = "computing metrics"
            self.metrics = {
                "mse_train": float(mean_squared_error(y_train, y_pred_train)),
                "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
                "r2_train": float(r2_score(y_train, y_pred_train)),
                "mse_test": float(mean_squared_error(y_test, y_pred_test)),
                "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
                "r2_test": float(r2_score(y_test, y_pred_test)),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "mode": mode,
                "model": model_name,
                "alpha": float(alpha),
            }

            # Save metrics
            self.stage = "saving metrics"
            self._log("Saving metrics to train_metrics.json")
            self.metrics_path = os.path.join(data_dir, "train_metrics.json")
            with open(self.metrics_path, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, indent=2)

            # Optional predictions
            self.preds_path = None
            if save_preds:
                self.stage = "saving predictions"
                self._log("Saving predictions to train_predictions.csv")
                ts_path = os.path.join(data_dir, "bci_timeseries.csv")
                if os.path.exists(ts_path):
                    t = pd.read_csv(ts_path)["t"].to_numpy()
                else:
                    t = np.arange(ds["meta"].get("T", X.shape[0]))
                dfp = pd.DataFrame({
                    "t": np.concatenate([t[:len(y_train)], t[len(y_train):len(y_train)+len(y_test)]]),
                    "split": ["train"] * len(y_train) + ["test"] * len(y_test),
                    "y_true": np.concatenate([y_train, y_test]),
                    "y_pred": np.concatenate([y_pred_train, y_pred_test]),
                })
                self.preds_path = os.path.join(data_dir, "train_predictions.csv")
                dfp.to_csv(self.preds_path, index=False)

            self.progress = 1.0
            self.stage = "completed"
            self._log("Training completed")
            self.status = "completed"
        except Exception as e:
            self.error = str(e)
            self.status = "error"


_JOBS: Dict[str, TrainJob] = {}
_JOBS_LOCK = threading.Lock()

def _register_job(job: "TrainJob") -> None:
    with _JOBS_LOCK:
        _JOBS[job.id] = job

def _get_job(job_id: Optional[str]) -> Optional["TrainJob"]:
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
    st.header("Dataset")
    data_dir = st.text_input("bci run directory", value=st.session_state.get("train_data_dir_default", ""), help="Directory with bci_* artifacts from bci-sim")
    mode = st.selectbox("mode", options=["features", "windows"], index=0, help="Select which artifacts to use: 'features' or 'windows'")

    st.header("Model")
    model = st.selectbox("model", options=["ridge", "lasso", "linear"], index=0, help="Baseline regression model")
    alpha = st.number_input("alpha (ridge/lasso)", min_value=0.0, max_value=1000.0, value=1.0, step=0.1, help="Regularization strength for ridge/lasso")

    st.header("Train/Test Split")
    test_size = st.number_input("test_size", min_value=0.05, max_value=0.95, value=0.2, step=0.05, help="Fraction of data for test split")
    random_state = st.number_input("random_state", min_value=0, max_value=1_000_000, value=0, step=1, help="Random state for deterministic split")

    st.header("Outputs")
    save_preds = st.checkbox("save_preds", value=True, help="Save predictions to 'train_predictions.csv'")
    # CLI preview
    def _cli_preview() -> str:
        parts = [
            f"{sys.executable} -m phi.cli neuro bci-train",
            f"--data-dir {data_dir}" if data_dir else "",
            f"--mode {mode}",
            f"--model {model}",
            f"--alpha {float(alpha)}",
            f"--test-size {float(test_size)}",
            f"--random-state {int(random_state)}",
        ]
        parts.append("--save-preds" if save_preds else "--no-save-preds")
        return " ".join([p for p in parts if p])
    st.caption("Equivalent CLI command")
    st.code(_cli_preview(), language="bash")
    start_btn = st.button("Start Training (background)", type="primary")


params = {
    "data_dir": data_dir,
    "mode": mode,
    "model": model,
    "alpha": float(alpha),
    "test_size": float(test_size),
    "random_state": int(random_state),
    "save_preds": bool(save_preds),
}
if start_btn and "train_job_id" not in st.session_state:
    job = TrainJob(params=params)
    _register_job(job)
    st.session_state["train_job_id"] = job.id

# Job status / controls
job = _get_job(st.session_state.get("train_job_id"))
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
    st.subheader("Training Job Status")
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
    if c2.button("Cancel training", disabled=(job.status != "running")):
        job.cancel()
    # Auto-refresh toggle
    auto_refresh = c4.checkbox("Auto-refresh", value=True, key="train_auto_refresh")
    if job.status in ("completed", "cancelled", "error"):
        if job.metrics_path and os.path.exists(job.metrics_path):
            st.success("Training complete" if job.status == "completed" else f"Job {job.status}")
            try:
                with open(job.metrics_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                c1m, c2m, c3m = st.columns(3)
                c1m.metric("MSE (test)", f"{m.get('mse_test', float('nan')):.4f}")
                c2m.metric("MAE (test)", f"{m.get('mae_test', float('nan')):.4f}")
                c3m.metric("R2 (test)", f"{m.get('r2_test', float('nan')):.3f}")
                st.json({"metrics_path": job.metrics_path, "preds_path": job.preds_path})
            except Exception:
                pass
        if job.preds_path and os.path.exists(job.preds_path):
            st.subheader("Predictions preview")
            try:
                st.dataframe(pd.read_csv(job.preds_path).head(1000))
            except Exception:
                pass
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
            st.session_state.pop("train_job_id", None)
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
    st.info("Configure parameters in the sidebar and start training.")
