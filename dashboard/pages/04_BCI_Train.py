import os
import sys
import pathlib

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

with st.sidebar:
    st.header("Dataset")
    data_dir = st.text_input("bci run directory", value="")
    mode = st.selectbox("mode", options=["features", "windows"], index=0)

    st.header("Model")
    model = st.selectbox("model", options=["ridge", "lasso", "linear"], index=0)
    alpha = st.number_input("alpha (ridge/lasso)", min_value=0.0, max_value=1000.0, value=1.0, step=0.1)

    st.header("Train/Test Split")
    test_size = st.number_input("test_size", min_value=0.05, max_value=0.95, value=0.2, step=0.05)
    random_state = st.number_input("random_state", min_value=0, max_value=1_000_000, value=0, step=1)

    st.header("Outputs")
    save_preds = st.checkbox("save_preds", value=True)

    run_btn = st.button("Run Training", type="primary")


if run_btn:
    if not data_dir or not os.path.isdir(data_dir):
        st.error("Provide a valid BCI run directory containing artifacts.")
        st.stop()

    use_features = mode.lower() == "features"
    use_windows = mode.lower() == "windows"
    ds = load_bci_dataset(data_dir, use_features=use_features, use_windows=use_windows)
    X = ds["X_feat"] if use_features else ds["X_win"]
    y = ds["y"]
    if X is None:
        st.error("Selected mode has no available X in data_dir")
        st.stop()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), shuffle=False
    )

    if model.lower() == "ridge":
        clf = Ridge(alpha=float(alpha))
    elif model.lower() == "lasso":
        clf = Lasso(alpha=float(alpha))
    else:
        clf = LinearRegression()

    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    metrics = {
        "mse_train": float(mean_squared_error(y_train, y_pred_train)),
        "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "mse_test": float(mean_squared_error(y_test, y_pred_test)),
        "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "mode": mode.lower(),
        "model": model.lower(),
        "alpha": float(alpha),
    }

    # Save metrics
    metrics_path = os.path.join(data_dir, "train_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, indent=2)

    # Optional predictions
    preds_path = None
    if save_preds:
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
        preds_path = os.path.join(data_dir, "train_predictions.csv")
        dfp.to_csv(preds_path, index=False)

    st.success("Training complete")
    st.json({"metrics_path": metrics_path, "preds_path": preds_path})

    # Display metrics and plot
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE (test)", f"{metrics['mse_test']:.4f}")
    c2.metric("MAE (test)", f"{metrics['mae_test']:.4f}")
    c3.metric("R2 (test)", f"{metrics['r2_test']:.3f}")

    st.subheader("Predictions preview")
    if save_preds and preds_path and os.path.exists(preds_path):
        st.dataframe(pd.read_csv(preds_path).head(1000))
    else:
        st.info("Enable save_preds to write and view predictions.")
