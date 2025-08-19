from pathlib import Path

import importlib.util
import pytest

from phi import quantum as qmod


def test_generate_and_save_load_roundtrip(tmp_path: Path):
    n = 5
    d = 3
    full = qmod.generate_full_circuit(num_qubits=n, depth=d, seed=123)
    assert full["type"] == "quantum_circuit_full"
    assert int(full["num_qubits"]) == n
    assert int(full["depth"]) == d

    ops = full.get("ops", [])
    assert isinstance(ops, list)
    # For our pattern: per repetition -> H(n) + barrier + CX(n) + barrier + H(n) = 3n + 2
    expected_ops = d * (3 * n + 2)
    assert len(ops) == expected_ops
    # 2 barriers per repetition
    barrier_count = sum(1 for op in ops if op.get("name") == "barrier")
    assert barrier_count == 2 * d

    p = tmp_path / "quantum_full.json"
    qmod.save_model(full, str(p))
    loaded = qmod.load_model(str(p))
    assert loaded["type"] == "quantum_circuit_full"
    assert int(loaded["num_qubits"]) == n
    assert int(loaded["depth"]) == d


def test_compress_expand_and_metrics(tmp_path: Path):
    n = 6
    d = 2
    full = qmod.generate_full_circuit(num_qubits=n, depth=d, seed=0)

    cfg = qmod.QuantumConfig(strategy="ratio", ratio=2, method="interp")
    comp = qmod.compress_circuit(full, cfg)
    assert comp["type"] == "quantum_circuit_ratio"
    assert comp["strategy"] == "ratio"
    assert int(comp["ratio"]) == 2
    assert int(comp["orig_qubits"]) == n
    assert comp["kept_qubits"] == [0, 2, 4]

    recon = qmod.expand_circuit(comp)
    assert recon["type"] == "quantum_circuit_full"
    assert int(recon["num_qubits"]) == n
    # Depth should be inferred from barriers in compressed ops
    assert int(recon.get("depth", 0)) == d

    a = tmp_path / "orig.json"
    b = tmp_path / "recon.json"
    qmod.save_model(full, str(a))
    qmod.save_model(recon, str(b))

    mdf = qmod.metrics_from_paths(str(a), str(b))
    assert hasattr(mdf, "to_csv")
    assert set(["metric", "a", "b", "abs_diff"]).issubset(mdf.columns)
    assert len(mdf) == 5  # h, cx, barrier, total, depth


def test_export_qasm(tmp_path: Path):
    n = 4
    d = 1
    full = qmod.generate_full_circuit(num_qubits=n, depth=d)
    qasm_p = tmp_path / "circuit.qasm"
    qmod.export_qasm(full, str(qasm_p))
    assert qasm_p.exists()
    text = qasm_p.read_text(encoding="utf-8")
    assert "OPENQASM 2.0;" in text
    assert 'include "qelib1.inc";' in text
    assert f"qreg q[{n}];" in text


def test_to_qiskit_circuit_if_available():
    if importlib.util.find_spec("qiskit") is None:
        pytest.skip("Qiskit not installed")
    full = qmod.generate_full_circuit(num_qubits=3, depth=2)
    qc = qmod.to_qiskit_circuit(full)
    # Basic sanity checks on returned object
    assert hasattr(qc, "num_qubits")
    assert int(qc.num_qubits) == 3
