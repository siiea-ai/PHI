from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import networkx as nx


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


@dataclass
class NeuroConfig:
    strategy: str = "ratio"      # only 'ratio' supported (educational)
    ratio: int = 4                # keep every Nth neuron
    method: str = "interp"        # 'interp' (linear) or 'nearest' for state upsample
    edge_method: str = "regen"    # 'regen' (regenerate with generator params)


# -----------------------------------------------------------------------------
# Helpers: base64 encode/decode numpy arrays as NPZ
# -----------------------------------------------------------------------------


def _arr_to_b64_npz(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.savez_compressed(buf, arr=arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_npz_to_arr(b64: str, dtype: Optional[np.dtype] = None) -> np.ndarray:
    raw = base64.b64decode(b64.encode("ascii"))
    buf = io.BytesIO(raw)
    with np.load(buf) as data:
        arr = data["arr"]
    if dtype is not None:
        return np.asarray(arr, dtype=dtype)
    return arr


def save_model(bundle: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


def load_model(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Graph utils
# -----------------------------------------------------------------------------


def _edges_from_graph(g: nx.Graph) -> np.ndarray:
    # undirected, unique edges (u < v)
    edges: List[Tuple[int, int]] = []
    for u, v in g.edges():
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.append((a, b))
    if not edges:
        return np.zeros((2, 0), dtype=np.int32)
    e = np.asarray(edges, dtype=np.int64)
    e = np.unique(e, axis=0)
    return e.T.astype(np.int32, copy=False)  # shape (2, E)


def _generate_ws(n: int, k: int, p: float, seed: Optional[int]) -> np.ndarray:
    if k < 0:
        k = 0
    k = min(k, max(0, n - 1))
    g = nx.watts_strogatz_graph(n=n, k=k, p=float(p), seed=seed)
    return _edges_from_graph(g)


def _generate_ba(n: int, m: int, seed: Optional[int]) -> np.ndarray:
    m = max(1, min(m, max(1, n - 1)))
    g = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    return _edges_from_graph(g)


def _edges_to_adj(edges: np.ndarray, n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float32)
    if edges.size == 0 or n <= 0:
        return a
    u = edges[0, :].astype(int)
    v = edges[1, :].astype(int)
    u = u[(u >= 0) & (u < n) & (v >= 0) & (v < n)]
    v = v[(u >= 0) & (u < n) & (v >= 0) & (v < n)]
    a[u, v] = 1.0
    a[v, u] = 1.0
    np.fill_diagonal(a, 0.0)
    return a


# -----------------------------------------------------------------------------
# Full generation
# -----------------------------------------------------------------------------


def generate_full_network(
    nodes: int,
    model: str = "ws",
    ws_k: int = 10,
    ws_p: float = 0.1,
    ba_m: int = 3,
    seed: Optional[int] = None,
    state_init: str = "random",
) -> Dict:
    if nodes < 1:
        raise ValueError("nodes must be >= 1")
    m = model.lower()
    if m == "ws":
        edges = _generate_ws(nodes, ws_k, ws_p, seed)
        params = {"ws_k": int(ws_k), "ws_p": float(ws_p)}
    elif m == "ba":
        edges = _generate_ba(nodes, ba_m, seed)
        params = {"ba_m": int(ba_m)}
    else:
        raise ValueError(f"Unknown model: {model}")

    if state_init.lower() == "random":
        rng = np.random.default_rng(seed)
        state = rng.random(nodes, dtype=np.float32)
    elif state_init.lower() == "zeros":
        state = np.zeros(nodes, dtype=np.float32)
    else:
        raise ValueError("state_init must be 'random' or 'zeros'")

    bundle = {
        "version": 1,
        "type": "neuro_network_full",
        "nodes": int(nodes),
        "edges": int(edges.shape[1]),
        "graph_model": m,
        "params": params,
        "seed": (int(seed) if seed is not None else None),
        "state_dtype": "float32",
        "state_npz_b64": _arr_to_b64_npz(state.astype(np.float32)),
        "edges_npz_b64": _arr_to_b64_npz(edges.astype(np.int32)),
    }
    return bundle


# -----------------------------------------------------------------------------
# Compress / Expand
# -----------------------------------------------------------------------------


def _filter_edges_for_nodes(edges: np.ndarray, keep: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.zeros((2, 0), dtype=np.int32)
    s = set(int(i) for i in keep.tolist())
    u = edges[0, :]
    v = edges[1, :]
    mask = np.fromiter(((int(uu) in s) and (int(vv) in s) for uu, vv in zip(u, v)), count=edges.shape[1], dtype=bool)
    if not np.any(mask):
        return np.zeros((2, 0), dtype=np.int32)
    u2 = mapping[u[mask]]
    v2 = mapping[v[mask]]
    du = np.minimum(u2, v2)
    dv = np.maximum(u2, v2)
    ds = np.stack([du, dv], axis=1)
    ds = ds[du != dv]
    if ds.size == 0:
        return np.zeros((2, 0), dtype=np.int32)
    ds = np.unique(ds, axis=0)
    return ds.T.astype(np.int32, copy=False)


def compress_network(full_bundle: Dict, config: Optional[NeuroConfig] = None) -> Dict:
    if full_bundle.get("type") != "neuro_network_full":
        raise ValueError("Not a full neuro network bundle")
    cfg = config or NeuroConfig()
    if cfg.strategy.lower() != "ratio":
        raise ValueError("Only 'ratio' strategy is supported for neuro")
    if cfg.ratio < 1:
        raise ValueError("ratio must be >= 1")

    n = int(full_bundle.get("nodes", 0))
    edges = _b64_npz_to_arr(full_bundle["edges_npz_b64"], dtype=np.int32)
    state = _b64_npz_to_arr(full_bundle["state_npz_b64"], dtype=np.float32)

    keep_idx = np.arange(0, n, cfg.ratio, dtype=np.int32)
    if keep_idx[-1] != (n - 1):
        keep_idx = np.unique(np.append(keep_idx, n - 1)).astype(np.int32)
    ds_nodes = int(keep_idx.shape[0])

    mapping = np.full(n, -1, dtype=np.int32)
    mapping[keep_idx] = np.arange(ds_nodes, dtype=np.int32)

    ds_edges = _filter_edges_for_nodes(edges, keep_idx, mapping)
    ds_state = state[keep_idx]

    bundle = {
        "version": 1,
        "type": "neuro_network_ratio",
        "strategy": "ratio",
        "ratio": int(cfg.ratio),
        "orig_nodes": int(n),
        "orig_edges": int(edges.shape[1]),
        "ds_nodes": int(ds_nodes),
        "ds_edges": int(ds_edges.shape[1]),
        "graph_model": full_bundle.get("graph_model"),
        "params": full_bundle.get("params", {}),
        "config": {"method": cfg.method, "edge_method": cfg.edge_method},
        "ds_indices": [int(i) for i in keep_idx.tolist()],
        "ds_state_dtype": "float32",
        "ds_state_npz_b64": _arr_to_b64_npz(ds_state.astype(np.float32)),
        "ds_edges_npz_b64": _arr_to_b64_npz(ds_edges.astype(np.int32)),
    }
    return bundle


def _resample_states(ds_values: np.ndarray, ds_idx: np.ndarray, target_n: int, orig_n: int, method: str) -> np.ndarray:
    if target_n < 1:
        return np.zeros((0,), dtype=np.float32)
    if ds_values.shape[0] == 1:
        return np.full((target_n,), float(ds_values[0]), dtype=np.float32)
    if orig_n > 1:
        src_x = (ds_idx.astype(np.float64) / float(orig_n - 1)) * float(target_n - 1)
    else:
        src_x = np.zeros_like(ds_idx, dtype=np.float64)
    tgt_x = np.arange(target_n, dtype=np.float64)
    meth = method.lower()
    try:
        from scipy.interpolate import interp1d  # type: ignore

        kind = "linear" if meth == "interp" else "nearest"
        f = interp1d(src_x, ds_values.astype(np.float64), kind=kind, fill_value=(float(ds_values[0]), float(ds_values[-1])), bounds_error=False)
        out = f(tgt_x)
        return np.asarray(out, dtype=np.float32)
    except Exception:
        if meth == "interp":
            out = np.interp(tgt_x, src_x, ds_values.astype(np.float64), left=float(ds_values[0]), right=float(ds_values[-1]))
            return np.asarray(out, dtype=np.float32)
        else:
            mids = (src_x[:-1] + src_x[1:]) * 0.5
            idx = np.searchsorted(mids, tgt_x, side="right")
            idx = np.clip(idx, 0, ds_values.shape[0] - 1)
            return ds_values[idx].astype(np.float32)


def _regen_edges(graph_model: str, params: Dict, n: int, seed: Optional[int]) -> np.ndarray:
    gm = (graph_model or "ws").lower()
    if gm == "ws":
        k = int(params.get("ws_k", 10))
        p = float(params.get("ws_p", 0.1))
        return _generate_ws(n, k, p, seed)
    elif gm == "ba":
        m = int(params.get("ba_m", 3))
        return _generate_ba(n, m, seed)
    else:
        k = int(params.get("k", 4))
        return _generate_ws(n, k, 0.0, seed)


def expand_network(bundle: Dict, target_nodes: Optional[int] = None, method: Optional[str] = None, seed: Optional[int] = None) -> Dict:
    btype = bundle.get("type")
    if btype == "neuro_network_full":
        n = int(bundle.get("nodes", 0))
        if target_nodes is None or target_nodes == n:
            return bundle
        gm = bundle.get("graph_model", "ws")
        params = bundle.get("params", {})
        state = _b64_npz_to_arr(bundle["state_npz_b64"], dtype=np.float32)
        new_n = int(target_nodes)
        edges = _regen_edges(gm, params, new_n, seed=(bundle.get("seed") if seed is None else seed))
        ds_idx = np.arange(n, dtype=np.int32)
        new_state = _resample_states(state, ds_idx, new_n, n, method or "interp")
        out = {
            "version": 1,
            "type": "neuro_network_full",
            "nodes": int(new_n),
            "edges": int(edges.shape[1]),
            "graph_model": gm,
            "params": params,
            "seed": bundle.get("seed"),
            "state_dtype": "float32",
            "state_npz_b64": _arr_to_b64_npz(new_state.astype(np.float32)),
            "edges_npz_b64": _arr_to_b64_npz(edges.astype(np.int32)),
        }
        return out

    if btype != "neuro_network_ratio":
        raise ValueError("Not a neuro ratio bundle")

    orig_n = int(bundle.get("orig_nodes", 0))
    new_n = int(target_nodes) if target_nodes is not None else orig_n
    cfg_method = (method or bundle.get("config", {}).get("method", "interp")).lower()

    ds_state = _b64_npz_to_arr(bundle["ds_state_npz_b64"], dtype=np.float32)
    ds_idx = np.asarray(bundle.get("ds_indices", list(range(ds_state.shape[0]))), dtype=np.int32)
    gm = bundle.get("graph_model", "ws")
    params = bundle.get("params", {})

    # Edges: regenerate using original generator (deterministic shape), favoring topology consistency
    edges = _regen_edges(gm, params, new_n, seed=seed)
    state = _resample_states(ds_state, ds_idx, new_n, orig_n, cfg_method)

    full = {
        "version": 1,
        "type": "neuro_network_full",
        "nodes": int(new_n),
        "edges": int(edges.shape[1]),
        "graph_model": gm,
        "params": params,
        "seed": bundle.get("seed"),
        "state_dtype": "float32",
        "state_npz_b64": _arr_to_b64_npz(state.astype(np.float32)),
        "edges_npz_b64": _arr_to_b64_npz(edges.astype(np.int32)),
    }
    return full


# -----------------------------------------------------------------------------
# Simulation (simple rate model)
# -----------------------------------------------------------------------------


def simulate_states(bundle: Dict, steps: int = 100, dt: float = 0.1, leak: float = 0.1, input_drive: float = 0.0, noise_std: float = 0.0, seed: Optional[int] = None) -> np.ndarray:
    if bundle.get("type") != "neuro_network_full":
        raise ValueError("simulate_states expects a full neuro bundle")
    n = int(bundle.get("nodes", 0))
    edges = _b64_npz_to_arr(bundle["edges_npz_b64"], dtype=np.int32)
    state0 = _b64_npz_to_arr(bundle["state_npz_b64"], dtype=np.float32)
    x = state0.astype(np.float32).copy()

    A = _edges_to_adj(edges, n)
    deg = np.maximum(A.sum(axis=1, keepdims=True), 1.0)
    W = (A / deg).astype(np.float32)  # simple normalized weights

    rng = np.random.default_rng(seed)
    out = np.zeros((steps + 1, n), dtype=np.float32)
    out[0] = x
    for t in range(1, steps + 1):
        y = np.tanh(x)  # activation
        drive = input_drive
        if noise_std > 0.0:
            drive = drive + rng.normal(0.0, noise_std, size=n).astype(np.float32)
        dx = (-leak * x + W @ y + drive).astype(np.float32)
        x = (x + dt * dx).astype(np.float32)
        out[t] = x
    return out


# -----------------------------------------------------------------------------
# Visualization & Metrics
# -----------------------------------------------------------------------------


def save_adjacency_image(bundle: Dict, output_path: str, cmap: str = "viridis") -> None:
    btype = bundle.get("type")
    if btype == "neuro_network_full":
        n = int(bundle.get("nodes", 0))
        edges = _b64_npz_to_arr(bundle["edges_npz_b64"], dtype=np.int32)
    elif btype == "neuro_network_ratio":
        n = int(bundle.get("ds_nodes", 0))
        edges = _b64_npz_to_arr(bundle["ds_edges_npz_b64"], dtype=np.int32)
    else:
        raise ValueError("Unknown bundle type")

    A = _edges_to_adj(edges, n)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    fig.set_size_inches(max(2, n / 100.0), max(2, n / 100.0))
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(A, cmap=cmap, origin="lower")
    buf = io.BytesIO()
    fig.canvas.print_png(buf)
    plt.close(fig)
    from PIL import Image

    img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    img.save(output_path)


def _state_mse(a: np.ndarray, b: np.ndarray) -> float:
    m = min(a.shape[0], b.shape[0])
    if m < 1:
        return float("nan")
    a = a[:m].astype(np.float32)
    b = b[:m].astype(np.float32)
    return float(np.mean((a - b) ** 2))


def _deg_hist_l1(A1: np.ndarray, A2: np.ndarray) -> float:
    d1 = A1.sum(axis=1)
    d2 = A2.sum(axis=1)
    mx = float(max(d1.max(initial=0.0), d2.max(initial=0.0), 1.0))
    h1, _ = np.histogram(d1 / mx, bins=32, range=(0.0, 1.0), density=True)
    h2, _ = np.histogram(d2 / mx, bins=32, range=(0.0, 1.0), density=True)
    return float(np.mean(np.abs(h1 - h2)))


def metrics_from_paths(a_path: str, b_path: str) -> pd.DataFrame:
    a = load_model(a_path)
    b = load_model(b_path)
    if a.get("type") != "neuro_network_full" or b.get("type") != "neuro_network_full":
        raise ValueError("metrics_from_paths expects full neuro bundles")
    n1 = int(a.get("nodes", 0))
    n2 = int(b.get("nodes", 0))
    e1 = int(a.get("edges", 0))
    e2 = int(b.get("edges", 0))
    A1 = _edges_to_adj(_b64_npz_to_arr(a["edges_npz_b64"], dtype=np.int32), n1)
    A2 = _edges_to_adj(_b64_npz_to_arr(b["edges_npz_b64"], dtype=np.int32), n2)
    s1 = _b64_npz_to_arr(a["state_npz_b64"], dtype=np.float32)
    s2 = _b64_npz_to_arr(b["state_npz_b64"], dtype=np.float32)

    # align states by resampling to max length
    if n1 != n2:
        if n1 > n2:
            idx2 = np.arange(n2, dtype=np.float32)
            s2 = np.interp(np.linspace(0, n2 - 1, n1), idx2, s2.astype(np.float32)).astype(np.float32)
        else:
            idx1 = np.arange(n1, dtype=np.float32)
            s1 = np.interp(np.linspace(0, n1 - 1, n2), idx1, s1.astype(np.float32)).astype(np.float32)
    mse_state = _state_mse(s1, s2)
    deg_l1 = _deg_hist_l1(A1, A2)

    return pd.DataFrame([
        {"mse_state": float(mse_state), "deg_l1": float(deg_l1), "nodes_a": int(n1), "nodes_b": int(n2), "edges_a": int(e1), "edges_b": int(e2)}
    ])
