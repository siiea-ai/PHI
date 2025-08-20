from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import networkx as nx
import imageio


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
    mask = (u >= 0) & (u < n) & (v >= 0) & (v < n)
    if np.any(mask):
        uu = u[mask]
        vv = v[mask]
        a[uu, vv] = 1.0
        a[vv, uu] = 1.0
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
# Robust input loaders (networks and signals)
# -----------------------------------------------------------------------------


def load_network_any(path: str, state_init: str = "random", seed: Optional[int] = None) -> Dict:
    """Load a neuro network bundle from various file types.

    Supported formats:
    - PHI JSON bundles: full or ratio (auto-expand to full)
    - CSV edgelist: two numeric columns (u,v) or columns named 'u','v'
    - NPY/NPZ adjacency matrix: 2D square array; non-zero entries become edges
    - JSON simple: dict with keys {'nodes', 'edges': [[u,v], ...]} or {'edges': ...} (nodes inferred)

    Returns a full neuro bundle with an initial state (random or zeros).
    """
    ext = os.path.splitext(path)[1].lower()
    rng = np.random.default_rng(seed)

    def _bundle_from_edges(edges: np.ndarray, nodes: int) -> Dict:
        edges = edges.astype(np.int32, copy=False)
        n = int(nodes)
        state = (rng.random(n, dtype=np.float32) if state_init.lower() == "random" else np.zeros(n, dtype=np.float32))
        return {
            "version": 1,
            "type": "neuro_network_full",
            "nodes": n,
            "edges": int(edges.shape[1]),
            "graph_model": "custom",
            "params": {},
            "seed": (int(seed) if seed is not None else None),
            "state_dtype": "float32",
            "state_npz_b64": _arr_to_b64_npz(state.astype(np.float32)),
            "edges_npz_b64": _arr_to_b64_npz(edges.astype(np.int32)),
        }

    if ext == ".json":
        obj = load_model(path)
        btype = obj.get("type")
        if btype == "neuro_network_full":
            return obj
        if btype == "neuro_network_ratio":
            # expand to original size by default
            return expand_network(obj, target_nodes=int(obj.get("orig_nodes", 0)))
        # simple JSON structure
        if "edges" in obj:
            edges_list = obj["edges"]
            e = np.asarray(edges_list, dtype=np.int64)
            if e.ndim != 2 or e.shape[1] != 2:
                raise ValueError("JSON 'edges' must be a list of [u,v]")
            n = int(obj.get("nodes", int(np.max(e) + 1 if e.size > 0 else 0)))
            return _bundle_from_edges(e.T, n)
        raise ValueError("Unknown JSON structure for neuro network")

    if ext == ".csv":
        df = pd.read_csv(path)
        cols = list(df.columns)
        if {"u", "v"}.issubset(set(cols)):
            u = df["u"].to_numpy()
            v = df["v"].to_numpy()
        elif len(cols) >= 2:
            u = df[cols[0]].to_numpy()
            v = df[cols[1]].to_numpy()
        else:
            raise ValueError("CSV must have at least two columns for edgelist (u,v)")
        e = np.vstack([u, v]).astype(np.int64, copy=False)
        n = int(np.max(e) + 1) if e.size > 0 else 0
        return _bundle_from_edges(e, n)

    if ext in (".npy", ".npz"):
        arr = None
        if ext == ".npy":
            arr = np.load(path)
        else:
            with np.load(path) as data:
                # try common keys first
                for k in ("arr", "adj", "A", "matrix"):
                    if k in data:
                        arr = data[k]
                        break
                if arr is None:
                    # take the first array
                    arr = list(data.values())[0]
        A = np.asarray(arr)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Adjacency array must be 2D square")
        n = int(A.shape[0])
        iu, iv = np.where(np.triu(A, k=1) != 0)
        e = np.stack([iu, iv], axis=0).astype(np.int64, copy=False)
        return _bundle_from_edges(e, n)

    raise ValueError(f"Unsupported network file type: {ext}")


def load_signal_any(path: str, target_length: Optional[int] = None, normalize: bool = True) -> np.ndarray:
    """Load a 1D signal from many file types and optionally resample to target_length.

    Supported formats:
    - Audio: .wav (native via imageio), or any audio via pydub if installed (delegated to phi.audio)
    - CSV: first numeric column (or mean across columns) -> 1D
    - NPY/NPZ: 1D array or flatten 2D/3D
    - JSON: list of numbers under root or under key 'signal'
    - Image: read and grayscale-mean per row flattened to 1D
    - Video: per-frame grayscale mean -> 1D
    """
    ext = os.path.splitext(path)[1].lower()

    def _resample_1d(x: np.ndarray, L: int) -> np.ndarray:
        if x.size == 0:
            return np.zeros((L,), dtype=np.float32)
        if L <= 0 or x.size == L:
            return x.astype(np.float32, copy=False)
        src = np.linspace(0.0, 1.0, num=x.size)
        tgt = np.linspace(0.0, 1.0, num=L)
        y = np.interp(tgt, src, x.astype(np.float64))
        return y.astype(np.float32)

    sig: Optional[np.ndarray] = None

    if ext in (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"):
        try:
            # Prefer phi.audio for broader codec support
            from . import audio as am  # type: ignore

            arr, sr = am.load_audio(path)
            # Collapse to mono
            if arr.ndim > 1:
                sig = np.mean(arr, axis=1)
            else:
                sig = arr
        except Exception:
            # Fallback: try imageio for WAV only
            if ext == ".wav":
                try:
                    arr = imageio.v3.imread(path)
                    sig = np.asarray(arr, dtype=np.float32).squeeze()
                except Exception as _:
                    pass
            if sig is None:
                raise

    elif ext == ".csv":
        df = pd.read_csv(path)
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] == 0:
            raise ValueError("CSV has no numeric columns to load as signal")
        if num_df.shape[1] == 1:
            sig = num_df.iloc[:, 0].to_numpy(dtype=np.float32, copy=False)
        else:
            sig = num_df.mean(axis=1).to_numpy(dtype=np.float32, copy=False)

    elif ext in (".npy", ".npz"):
        arr = None
        if ext == ".npy":
            arr = np.load(path)
        else:
            with np.load(path) as data:
                for k in ("arr", "signal", "x"):
                    if k in data:
                        arr = data[k]
                        break
                if arr is None:
                    arr = list(data.values())[0]
        arr = np.asarray(arr)
        sig = arr.reshape(-1).astype(np.float32, copy=False)

    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "signal" in obj:
            sig = np.asarray(obj["signal"], dtype=np.float32)
        elif isinstance(obj, list):
            sig = np.asarray(obj, dtype=np.float32)
        else:
            raise ValueError("JSON must be a list of numbers or have key 'signal'")

    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
        img = imageio.v3.imread(path)
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 3:  # HxWxC
            arr = np.mean(arr, axis=2)
        # reduce to 1D by averaging rows
        sig = arr.mean(axis=1)

    elif ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        means: List[float] = []
        try:
            reader = imageio.get_reader(path)
            for frame in reader:
                frame_arr = np.asarray(frame, dtype=np.float32)
                if frame_arr.ndim == 3:
                    frame_arr = np.mean(frame_arr, axis=2)
                means.append(float(frame_arr.mean()))
            reader.close()
        except Exception:
            # best-effort; if fails, leave sig None
            pass
        if means:
            sig = np.asarray(means, dtype=np.float32)

    if sig is None:
        raise ValueError(f"Unsupported signal file type or failed to load: {ext}")

    sig = np.asarray(sig, dtype=np.float32)
    if normalize and sig.size > 0:
        sig = sig - float(sig.mean())
        mx = float(np.max(np.abs(sig)))
        if mx > 0:
            sig = sig / mx

    if target_length is not None:
        sig = _resample_1d(sig, int(target_length))
    return sig.astype(np.float32, copy=False)


def make_pulse_signal(steps: int, period: int = 100, width: int = 10, amplitude: float = 1.0, kind: str = "rect") -> np.ndarray:
    """Create a simple periodic pulse signal of length `steps`.

    kind: 'rect' (rectangular), 'tri' (triangular), 'sine' (sinusoidal windows)
    """
    steps = max(0, int(steps))
    period = max(1, int(period))
    width = max(1, int(width))
    t = np.arange(steps, dtype=np.float32)
    phase = (t % period) / float(period)
    if kind == "tri":
        # triangle window centered on start of each period
        win = 1.0 - np.abs((phase * 2.0) - 1.0)
        sig = (win > (1.0 - (width / float(period)))) * win
    elif kind == "sine":
        sig = 0.5 * (1.0 + np.sin(2.0 * np.pi * phase))
        sig = (phase < (width / float(period))) * sig
    else:  # rect
        sig = (phase < (width / float(period))).astype(np.float32)
    return amplitude * sig.astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# Simulation (simple rate model)
# -----------------------------------------------------------------------------


def simulate_states(
    bundle: Dict,
    steps: int = 100,
    dt: float = 0.1,
    leak: float = 0.1,
    input_drive: float | np.ndarray = 0.0,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    if bundle.get("type") != "neuro_network_full":
        raise ValueError("simulate_states expects a full neuro bundle")
    n = int(bundle.get("nodes", 0))
    edges = _b64_npz_to_arr(bundle["edges_npz_b64"], dtype=np.int32)
    state0 = _b64_npz_to_arr(bundle["state_npz_b64"], dtype=np.float32)
    # Use float64 internally for better numerical stability
    x = state0.astype(np.float64, copy=True)

    A = _edges_to_adj(edges, n).astype(np.float64, copy=False)
    deg = A.sum(axis=1, keepdims=True, dtype=np.float64)
    deg[deg < 1.0] = 1.0  # avoid division by zero for isolated nodes
    W = A / deg  # normalized weights in float64
    # Replace any NaN/Inf just in case of unexpected values
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

    rng = np.random.default_rng(seed)
    out = np.zeros((steps + 1, n), dtype=np.float32)
    out[0] = x.astype(np.float32)
    # Preprocess input drive (supports scalar, 1D over time, or 2D [time, nodes])
    drive_scalar: Optional[float] = None
    drive_1d: Optional[np.ndarray] = None
    drive_2d: Optional[np.ndarray] = None
    if isinstance(input_drive, (int, float)):
        drive_scalar = float(input_drive)
    else:
        arr = np.asarray(input_drive)
        if arr.ndim == 1:
            # length should be >= steps; if not, resample via interp
            if arr.size != steps:
                src = np.linspace(0.0, 1.0, num=max(1, arr.size))
                tgt = np.linspace(0.0, 1.0, num=steps)
                arr = np.interp(tgt, src, arr.astype(np.float64)).astype(np.float32)
            drive_1d = arr.astype(np.float64, copy=False)
        elif arr.ndim == 2:
            T, N = arr.shape
            if N != n:
                # resample across node dimension to n (nearest)
                idx = np.linspace(0, N - 1, num=n)
                j = np.clip(np.round(idx).astype(int), 0, N - 1)
                arr = arr[:, j]
            if T != steps:
                # resample time axis to steps via linear interp per node
                src = np.linspace(0.0, 1.0, num=max(1, T))
                tgt = np.linspace(0.0, 1.0, num=steps)
                out = np.zeros((steps, n), dtype=np.float64)
                for k in range(n):
                    out[:, k] = np.interp(tgt, src, arr[:, k].astype(np.float64))
                arr = out
            drive_2d = arr.astype(np.float64, copy=False)
        else:
            raise ValueError("input_drive must be a scalar, 1D array, or 2D [time, nodes]")

    for t in range(1, steps + 1):
        # Guard against numerical warnings during matmul and activation
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):  # suppress benign runtime warnings
            y = np.tanh(np.clip(x, -50.0, 50.0))  # clip input to tanh for stability
            if drive_scalar is not None:
                drive: np.ndarray | float = float(drive_scalar)
            elif drive_1d is not None:
                drive = float(drive_1d[t - 1])
            elif drive_2d is not None:
                drive = drive_2d[t - 1]
            else:
                drive = 0.0
            if noise_std > 0.0:
                drive = drive + rng.normal(0.0, float(noise_std), size=n).astype(np.float64)
            dx = -float(leak) * x + (W @ y) + drive
            x = x + float(dt) * dx
        # Ensure state stays finite
        x = np.nan_to_num(x, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        np.clip(x, -1e6, 1e6, out=x)
        out[t] = x.astype(np.float32)
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
