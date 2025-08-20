from __future__ import annotations

import base64
import io
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import PHI, INV_PHI, fibonacci_sequence


@dataclass
class AIConfig:
    strategy: str = "ratio"           # only 'ratio' supported (educational)
    ratio: int = 2                     # keep every Nth neuron
    method: str = "interp"             # 'interp' (linear blend) or 'nearest'
    act_hidden: str = "relu"
    act_output: str = "sigmoid"


# ---------- JSON bundle I/O with base64-encoded numpy weights ----------

def _arr_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype=np.float32), allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_arr(b64_str: str) -> np.ndarray:
    data = base64.b64decode(b64_str.encode("ascii"))
    buf = io.BytesIO(data)
    arr = np.load(buf, allow_pickle=False)
    return np.asarray(arr, dtype=np.float32)


def save_model(bundle: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


def load_model(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Architecture generation ----------

def generate_architecture(
    input_dim: int,
    output_dim: int,
    depth: int = 3,
    base_width: int = 64,
    mode: str = "phi",  # 'phi'|'fibonacci'|'fixed'
    min_width: int = 4,
) -> List[int]:
    if depth < 0:
        return []
    widths: List[int] = []
    if mode.lower() == "phi":
        for i in range(depth):
            w = int(round(base_width * (INV_PHI ** i)))
            widths.append(max(min_width, w))
    elif mode.lower() == "fibonacci":
        fibs = fibonacci_sequence(depth + 1)[-depth:] if depth > 0 else []
        mx = fibs[-1] if fibs else 1
        for f in fibs:
            w = int(round(base_width * (f / max(1, mx))))
            widths.append(max(min_width, w))
    else:  # fixed
        widths = [max(min_width, int(base_width)) for _ in range(depth)]
    return widths


# ---------- Weights and bundles ----------

def init_weights(input_dim: int, output_dim: int, hidden: List[int], seed: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    dims: List[int] = [int(input_dim)] + [int(w) for w in hidden] + [int(output_dim)]
    layers: List[Dict[str, np.ndarray]] = []
    for i in range(len(dims) - 1):
        in_d, out_d = int(dims[i]), int(dims[i + 1])
        # He/Xavier-like init
        W = rng.normal(0.0, 1.0 / max(1.0, np.sqrt(in_d)), size=(in_d, out_d)).astype(np.float32)
        b = np.zeros((out_d,), dtype=np.float32)
        layers.append({"W": W, "b": b})
    return layers


def bundle_from_weights(
    layers: List[Dict[str, np.ndarray]],
    input_dim: int,
    output_dim: int,
    hidden: List[int],
    act_hidden: str = "relu",
    act_output: str = "sigmoid",
) -> Dict:
    enc_layers = [{"W": _arr_to_b64(L["W"]), "b": _arr_to_b64(L["b"]) } for L in layers]
    return {
        "version": 1,
        "type": "phi-ai-full",
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "hidden": [int(x) for x in hidden],
        "act_hidden": str(act_hidden),
        "act_output": str(act_output),
        "layers": enc_layers,
        "param_count": int(sum(L["W"].size + L["b"].size for L in layers)),
    }


def layers_from_bundle(bundle: Dict) -> List[Dict[str, np.ndarray]]:
    return [{"W": _b64_to_arr(L["W"]), "b": _b64_to_arr(L["b"]) } for L in bundle.get("layers", [])]


# ---------- Compress/Expand (ratio strategy across hidden layers) ----------

def _keep_indices(n: int, ratio: int) -> np.ndarray:
    ratio = max(1, int(ratio))
    idx = np.arange(n)
    keep = idx[::ratio]
    if keep.size == 0:
        keep = np.array([0], dtype=int)
    return keep.astype(int)


def compress_weights(layers: List[Dict[str, np.ndarray]], ratio: int = 2) -> Tuple[List[Dict[str, np.ndarray]], List[int]]:
    """Decimate hidden-layer neurons by keeping every Nth output channel.
    Assumes layers[i]['W'] has shape (in_dim_i, out_dim_i).
    Returns (compressed_layers, compressed_hidden_widths).
    """
    comp: List[Dict[str, np.ndarray]] = []
    widths: List[int] = []
    L = len(layers)
    for i in range(L):
        W = layers[i]["W"]
        b = layers[i]["b"]
        in_d, out_d = W.shape
        if i < L - 1:  # hidden layer
            keep = _keep_indices(out_d, ratio)
            W2 = W[:, keep]
            b2 = b[keep]
            widths.append(int(W2.shape[1]))
            # Next layer must shrink its rows accordingly
            # We'll apply this when processing the next layer by selecting rows
            comp.append({"W": W2, "b": b2})
            # Modify next layer input by selecting rows now, so when we reach it, it already has reduced input dim
            Wn = layers[i + 1]["W"]
            bn = layers[i + 1]["b"]
            layers[i + 1] = {"W": Wn[keep, :], "b": bn}
        else:
            # output layer (do not decimate outputs)
            comp.append({"W": W, "b": b})
    return comp, widths


def expand_weights(
    comp_layers: List[Dict[str, np.ndarray]],
    target_hidden: List[int],
    method: str = "interp",
    seed: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    """Expand compressed hidden layers to target widths.
    Adjusts subsequent layer rows to match expanded outputs.
    """
    rng = np.random.default_rng(seed)
    L = len(comp_layers)
    # Copy to avoid in-place mutation side effects
    layers = [{"W": Ld["W"].copy(), "b": Ld["b"].copy()} for Ld in comp_layers]

    for i in range(L - 1):  # process hidden layers only; last is output
        W = layers[i]["W"]  # (in, out)
        b = layers[i]["b"]  # (out,)
        in_d, out_d = W.shape
        tgt = int(target_hidden[i]) if i < len(target_hidden) else out_d
        if tgt <= out_d:
            # Truncate if needed
            layers[i]["W"] = W[:, :tgt]
            layers[i]["b"] = b[:tgt]
            # Adjust next layer rows
            layers[i + 1]["W"] = layers[i + 1]["W"][:tgt, :]
            continue
        need = tgt - out_d
        if method.lower() == "nearest":
            idx = rng.integers(0, out_d, size=need)
            addW = W[:, idx]
            addb = b[idx]
            addNext = layers[i + 1]["W"][idx, :]  # rows to replicate
        elif method.lower() == "interp":
            i0 = rng.integers(0, out_d, size=need)
            i1 = rng.integers(0, out_d, size=need)
            a = rng.random(size=need, dtype=np.float32)
            addW = (1.0 - a)[None, :] * W[:, i0] + a[None, :] * W[:, i1]
            addb = (1.0 - a) * b[i0] + a * b[i1]
            addNext = (1.0 - a)[:, None] * layers[i + 1]["W"][i0, :] + a[:, None] * layers[i + 1]["W"][i1, :]
        else:
            raise ValueError(f"Unknown method: {method}")
        layers[i]["W"] = np.concatenate([W, addW], axis=1)
        layers[i]["b"] = np.concatenate([b, addb], axis=0)
        layers[i + 1]["W"] = np.concatenate([layers[i + 1]["W"], addNext], axis=0)
    return layers


def compress_model(full_bundle: Dict, cfg: Optional[AIConfig] = None) -> Dict:
    cfg = cfg or AIConfig()
    if cfg.strategy.lower() != "ratio":
        raise ValueError("Only 'ratio' strategy is supported for AI")
    layers = layers_from_bundle(full_bundle)
    comp_layers, comp_hidden = compress_weights(layers, ratio=cfg.ratio)
    enc_layers = [{"W": _arr_to_b64(L["W"]), "b": _arr_to_b64(L["b"]) } for L in comp_layers]
    return {
        "version": 1,
        "type": "phi-ai-model",  # compressed model bundle
        "strategy": "ratio",
        "ratio": int(cfg.ratio),
        "method": str(cfg.method),
        "input_dim": int(full_bundle["input_dim"]),
        "output_dim": int(full_bundle["output_dim"]),
        "orig_hidden": [int(x) for x in full_bundle.get("hidden", [])],
        "ds_hidden": [int(x) for x in comp_hidden],
        "act_hidden": str(full_bundle.get("act_hidden", "relu")),
        "act_output": str(full_bundle.get("act_output", "sigmoid")),
        "layers": enc_layers,
        "orig_param_count": int(full_bundle.get("param_count", 0)),
        "ds_param_count": int(sum(_b64_to_arr(L["W"]).size + _b64_to_arr(L["b"]).size for L in enc_layers)),
    }


def expand_model(bundle: Dict, target_hidden: Optional[List[int]] = None, method: Optional[str] = None, seed: Optional[int] = None) -> Dict:
    if bundle.get("type") != "phi-ai-model":
        raise ValueError("Not a compressed AI model bundle")
    method_use = (method or bundle.get("method", "interp")).lower()
    ds_layers = layers_from_bundle(bundle)
    target_hidden = target_hidden or [int(x) for x in bundle.get("orig_hidden", bundle.get("ds_hidden", []))]
    full_layers = expand_weights(ds_layers, target_hidden=target_hidden, method=method_use, seed=seed)
    hidden = target_hidden
    enc_layers = [{"W": _arr_to_b64(L["W"]), "b": _arr_to_b64(L["b"]) } for L in full_layers]
    return {
        "version": 1,
        "type": "phi-ai-full",
        "input_dim": int(bundle["input_dim"]),
        "output_dim": int(bundle["output_dim"]),
        "hidden": [int(x) for x in hidden],
        "act_hidden": str(bundle.get("act_hidden", "relu")),
        "act_output": str(bundle.get("act_output", "sigmoid")),
        "layers": enc_layers,
        "param_count": int(sum(_b64_to_arr(L["W"]).size + _b64_to_arr(L["b"]).size for L in enc_layers)),
    }


# ---------- Metrics ----------

def metrics_from_paths(orig_model_path: str, recon_model_path: str) -> "pd.DataFrame":
    import pandas as pd  # optional dep pattern
    ob = load_model(orig_model_path)
    rb = load_model(recon_model_path)
    ol = layers_from_bundle(ob)
    rl = layers_from_bundle(rb)
    if len(ol) != len(rl):
        return pd.DataFrame([{"layers_equal": False, "mse_total": np.nan}])
    mse_layers: List[float] = []
    for i, (L0, L1) in enumerate(zip(ol, rl)):
        if L0["W"].shape != L1["W"].shape or L0["b"].shape != L1["b"].shape:
            mse_layers.append(np.nan)
            continue
        dW = L0["W"] - L1["W"]
        db = L0["b"] - L1["b"]
        mse = float((np.mean(dW ** 2) + np.mean(db ** 2)) / 2.0)
        mse_layers.append(mse)
    row = {
        "layers_equal": all(np.isfinite(m) for m in mse_layers),
        "mse_total": float(np.nanmean(mse_layers)) if any(np.isfinite(m) for m in mse_layers) else np.nan,
        "param_orig": int(ob.get("param_count", 0)),
        "param_recon": int(rb.get("param_count", 0)),
    }
    # also include per-layer mse columns
    for i, m in enumerate(mse_layers):
        row[f"layer{i}_mse"] = m
    return pd.DataFrame([row])


# ---------- Optional Keras export ----------

def export_keras(bundle: Dict, output_path: str) -> None:
    try:
        from tensorflow.keras.models import Sequential  # type: ignore
        from tensorflow.keras.layers import Dense, Input  # type: ignore
    except Exception as e:
        raise RuntimeError("TensorFlow Keras is required for export; pip install tensorflow") from e

    layers = layers_from_bundle(bundle)
    input_dim = int(bundle["input_dim"]) if "input_dim" in bundle else int(layers[0]["W"].shape[0])
    hidden = [int(L["W"].shape[1]) for L in layers[:-1]]
    output_dim = int(layers[-1]["W"].shape[1])
    act_hidden = str(bundle.get("act_hidden", "relu"))
    act_output = str(bundle.get("act_output", "sigmoid"))

    # Build model using an explicit Input to avoid legacy input_shape warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*__array__ implementation.*copy keyword.*",
            category=DeprecationWarning,
        )
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        for h in hidden:
            model.add(Dense(h, activation=act_hidden))
        model.add(Dense(output_dim, activation=act_output))

    # Assign weights to Dense layers (skip the InputLayer)
    dense_layers = [lyr for lyr in model.layers if isinstance(lyr, Dense)]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*__array__ implementation.*copy keyword.*",
            category=DeprecationWarning,
        )
        for i, L in enumerate(layers[:len(dense_layers)]):
            W = np.asarray(L["W"], dtype=np.float32)
            b = np.asarray(L["b"], dtype=np.float32)
            dense_layers[i].set_weights([W, b])

    # Save in native Keras format by default; suppress internal backend warnings during save
    ext = os.path.splitext(output_path)[1].lower()

    def _save(path: str) -> None:
        with warnings.catch_warnings():
            # Suppress NumPy/Keras DeprecationWarning triggered inside Keras backend on NumPy 2.x
            warnings.filterwarnings(
                "ignore",
                message=".*__array__ implementation.*copy keyword.*",
                category=DeprecationWarning,
            )
            # If user explicitly wants legacy HDF5, also suppress that noisy UserWarning
            if path.lower().endswith((".h5", ".hdf5")):
                warnings.filterwarnings("ignore", message=".*HDF5.*legacy.*", category=UserWarning)
            model.save(path)

    if ext in (".keras", ""):
        final_path = output_path if ext == ".keras" else f"{output_path}.keras"
        _save(final_path)
    else:
        _save(output_path)


# ---------- Convenience helpers for CLI ----------

def generate_full_model(
    input_dim: int,
    output_dim: int,
    depth: int = 3,
    base_width: int = 64,
    mode: str = "phi",
    act_hidden: str = "relu",
    act_output: str = "sigmoid",
    seed: Optional[int] = None,
) -> Dict:
    hidden = generate_architecture(input_dim, output_dim, depth=depth, base_width=base_width, mode=mode)
    layers = init_weights(input_dim, output_dim, hidden, seed=seed)
    return bundle_from_weights(layers, input_dim, output_dim, hidden, act_hidden=act_hidden, act_output=act_output)
