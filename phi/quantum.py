from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class QuantumConfig:
    """Configuration for ratio-based quantum circuit compression.

    - ratio: keep every Nth qubit (0, N, 2N, ...)
    - method: expansion method hint ("interp" or "nearest"). Currently used as metadata.
    """

    strategy: str = "ratio"
    ratio: int = 2
    method: str = "nearest"


# -----------------------------------------------------------------------------
# Core ops and helpers
# -----------------------------------------------------------------------------

Op = Dict[str, Any]


def pattern_ops(num_qubits: int, depth: int = 1) -> List[Op]:
    """Builds a simple fractal-like pattern repeated `depth` times:
    1) H on all qubits
    2) barrier
    3) ring CX across qubits
    4) barrier
    5) H on all qubits

    The pattern is intentionally simple and educational.
    """
    ops: List[Op] = []
    for _ in range(max(1, depth)):
        for q in range(num_qubits):
            ops.append({"name": "h", "qubits": [q]})
        ops.append({"name": "barrier"})
        for q in range(num_qubits):
            ops.append({"name": "cx", "qubits": [q, (q + 1) % num_qubits]})
        ops.append({"name": "barrier"})
        for q in range(num_qubits):
            ops.append({"name": "h", "qubits": [q]})
    return ops


def generate_full_circuit(num_qubits: int, depth: int = 1, seed: Optional[int] = None) -> Dict[str, Any]:
    """Generate a full circuit bundle.

    Bundle schema:
    {
      "type": "quantum_circuit_full",
      "num_qubits": int,
      "depth": int,
      "ops": List[Op]
    }
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if depth <= 0:
        raise ValueError("depth must be positive")
    # 'seed' reserved for future stochastic patterns
    ops = pattern_ops(num_qubits, depth=depth)
    return {
        "type": "quantum_circuit_full",
        "num_qubits": int(num_qubits),
        "depth": int(depth),
        "ops": ops,
    }


# -----------------------------------------------------------------------------
# JSON I/O
# -----------------------------------------------------------------------------


def save_model(bundle: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f)


def load_model(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Compression / Expansion
# -----------------------------------------------------------------------------


def compress_circuit(full: Dict[str, Any], cfg: QuantumConfig) -> Dict[str, Any]:
    """Compress full circuit by keeping every Nth qubit and pruning gates accordingly.

    Output bundle:
    {
      "type": "quantum_circuit_ratio",
      "strategy": "ratio",
      "ratio": int,
      "method": str,
      "orig_qubits": int,
      "kept_qubits": List[int],
      "ops": List[Op]
    }
    """
    if full.get("type") != "quantum_circuit_full":
        raise ValueError("compress_circuit expects a full circuit bundle")
    n = int(full["num_qubits"])
    ratio = int(cfg.ratio)
    if ratio <= 1:
        # trivial compression: copy ops
        return {
            "type": "quantum_circuit_ratio",
            "strategy": cfg.strategy,
            "ratio": ratio,
            "method": cfg.method,
            "orig_qubits": n,
            "kept_qubits": list(range(n)),
            "ops": list(full["ops"]),
        }
    kept = list(range(0, n, ratio))
    kept_set = set(kept)
    index_map = {q: i for i, q in enumerate(kept)}  # original q -> new index

    pruned_ops: List[Op] = []
    for op in full["ops"]:
        name = op.get("name")
        if name == "barrier":
            pruned_ops.append({"name": "barrier"})
        elif name == "h":
            (q,) = op["qubits"]
            if q in kept_set:
                pruned_ops.append({"name": "h", "qubits": [index_map[q]]})
        elif name == "cx":
            q0, q1 = op["qubits"]
            if q0 in kept_set and q1 in kept_set:
                pruned_ops.append({"name": "cx", "qubits": [index_map[q0], index_map[q1]]})
        else:
            # Unknown op: drop for safety
            continue

    return {
        "type": "quantum_circuit_ratio",
        "strategy": cfg.strategy,
        "ratio": ratio,
        "method": cfg.method,
        "orig_qubits": n,
        "kept_qubits": kept,
        "ops": pruned_ops,
    }


def expand_circuit(bundle: Dict[str, Any], target_qubits: Optional[int] = None, method: Optional[str] = None) -> Dict[str, Any]:
    """Expand compressed circuit back to a full circuit.

    If target_qubits is None, expands back to orig_qubits (if available),
    otherwise regenerates the canonical pattern for the target size and recorded depth.
    """
    if bundle.get("type") == "quantum_circuit_full":
        # already full; optionally re-target
        n = int(bundle["num_qubits"]) if target_qubits is None else int(target_qubits)
        depth = int(bundle.get("depth", 1))
        return generate_full_circuit(n, depth=depth)

    if bundle.get("type") != "quantum_circuit_ratio":
        raise ValueError("expand_circuit expects a compressed ratio bundle or full bundle")

    orig_n = int(bundle.get("orig_qubits", 0))
    n = int(target_qubits) if target_qubits is not None else (orig_n if orig_n > 0 else len(bundle.get("kept_qubits", [])))
    depth = infer_depth_from_ops(bundle.get("ops", []))
    # Educational expansion: regenerate canonical pattern for target size
    return generate_full_circuit(n, depth=max(1, depth))


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def infer_depth_from_ops(ops: List[Op]) -> int:
    if not ops:
        return 1
    # Each repetition in pattern_ops inserts exactly 2 barriers
    # H... barrier CX... barrier H...
    barrier_count = 0
    for op in ops:
        if op.get("name") == "barrier":
            barrier_count += 1
    reps = barrier_count // 2
    return max(1, int(reps))


def gate_counts(ops: List[Op]) -> Dict[str, int]:
    counts: Dict[str, int] = {"h": 0, "cx": 0, "barrier": 0}
    for op in ops:
        nm = op.get("name")
        if nm in counts:
            counts[nm] += 1
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    return counts


def metrics_from_bundles(full_a: Dict[str, Any], full_b: Dict[str, Any]) -> pd.DataFrame:
    if full_a.get("type") != "quantum_circuit_full" or full_b.get("type") != "quantum_circuit_full":
        raise ValueError("metrics_from_bundles expects two full circuit bundles")
    a_counts = gate_counts(full_a.get("ops", []))
    b_counts = gate_counts(full_b.get("ops", []))
    a_depth = infer_depth_from_ops(full_a.get("ops", []))
    b_depth = infer_depth_from_ops(full_b.get("ops", []))
    rows = [
        {"metric": "h_count", "a": a_counts["h"], "b": b_counts["h"], "abs_diff": abs(a_counts["h"] - b_counts["h"])},
        {"metric": "cx_count", "a": a_counts["cx"], "b": b_counts["cx"], "abs_diff": abs(a_counts["cx"] - b_counts["cx"])},
        {"metric": "barrier_count", "a": a_counts["barrier"], "b": b_counts["barrier"], "abs_diff": abs(a_counts["barrier"] - b_counts["barrier"])},
        {"metric": "total_count", "a": a_counts["total"], "b": b_counts["total"], "abs_diff": abs(a_counts["total"] - b_counts["total"])},
        {"metric": "depth", "a": a_depth, "b": b_depth, "abs_diff": abs(a_depth - b_depth)},
    ]
    return pd.DataFrame(rows)


def metrics_from_paths(a_path: str, b_path: str) -> pd.DataFrame:
    return metrics_from_bundles(load_model(a_path), load_model(b_path))


# -----------------------------------------------------------------------------
# OpenQASM export (no Qiskit dependency required)
# -----------------------------------------------------------------------------


def export_qasm(bundle: Dict[str, Any], output_path: str) -> None:
    """Export a full circuit bundle to OpenQASM 2.0. If given a compressed bundle,
    it will first expand back to a full circuit using recorded metadata.
    """
    if bundle.get("type") != "quantum_circuit_full":
        bundle = expand_circuit(bundle)
    n = int(bundle["num_qubits"])
    ops = bundle.get("ops", [])

    lines: List[str] = []
    lines.append("OPENQASM 2.0;")
    lines.append('include "qelib1.inc";')
    lines.append(f"qreg q[{n}];")
    # Map ops
    for op in ops:
        nm = op.get("name")
        if nm == "h":
            q = int(op["qubits"][0])
            lines.append(f"h q[{q}];")
        elif nm == "cx":
            q0 = int(op["qubits"][0])
            q1 = int(op["qubits"][1])
            lines.append(f"cx q[{q0}],q[{q1}];")
        elif nm == "barrier":
            # apply barrier to all qubits for simplicity
            allq = ",".join([f"q[{i}]" for i in range(n)])
            lines.append(f"barrier {allq};")
        else:
            # skip unknown
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Utility to convert to Qiskit (optional, if available)
# -----------------------------------------------------------------------------


def to_qiskit_circuit(bundle: Dict[str, Any]):  # type: ignore[no-untyped-def]
    """Returns a qiskit.QuantumCircuit if Qiskit Terra is installed; otherwise raises.
    This is optional and not required for tests.
    """
    try:
        from qiskit import QuantumCircuit  # type: ignore
    except Exception as e:  # pragma: no cover - optional path
        raise RuntimeError("Qiskit Terra is required: pip install qiskit-terra") from e

    if bundle.get("type") != "quantum_circuit_full":
        bundle = expand_circuit(bundle)
    n = int(bundle["num_qubits"])
    qc = QuantumCircuit(n)
    for op in bundle.get("ops", []):
        nm = op.get("name")
        if nm == "h":
            (q,) = op["qubits"]
            qc.h(int(q))
        elif nm == "cx":
            q0, q1 = op["qubits"]
            qc.cx(int(q0), int(q1))
        elif nm == "barrier":
            qc.barrier()
    return qc
