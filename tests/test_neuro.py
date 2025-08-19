from pathlib import Path

import numpy as np

from phi import neuro as nmod


def test_generate_and_save_load_roundtrip(tmp_path: Path):
    full = nmod.generate_full_network(
        nodes=32,
        model="ws",
        ws_k=6,
        ws_p=0.1,
        ba_m=3,
        seed=123,
        state_init="random",
    )
    assert full["type"] == "neuro_network_full"
    assert int(full["nodes"]) == 32
    assert "state_npz_b64" in full and "edges_npz_b64" in full

    p = tmp_path / "neuro_full.json"
    nmod.save_model(full, str(p))
    loaded = nmod.load_model(str(p))
    assert loaded["type"] == "neuro_network_full"
    assert int(loaded["nodes"]) == 32


def test_compress_expand_shapes_and_metrics(tmp_path: Path):
    full = nmod.generate_full_network(
        nodes=60,
        model="ws",
        ws_k=4,
        ws_p=0.1,
        ba_m=2,
        seed=0,
        state_init="random",
    )

    cfg = nmod.NeuroConfig(strategy="ratio", ratio=4, method="interp")
    comp = nmod.compress_network(full, cfg)
    assert comp["type"] == "neuro_network_ratio"
    assert comp["strategy"] == "ratio"
    assert int(comp["ratio"]) == 4
    assert int(comp["ds_nodes"]) >= 2

    recon = nmod.expand_network(comp, target_nodes=int(full["nodes"]), method="interp", seed=1)
    assert recon["type"] == "neuro_network_full"
    assert int(recon["nodes"]) == int(full["nodes"])

    orig_p = tmp_path / "neuro_full.json"
    recon_p = tmp_path / "neuro_recon.json"
    nmod.save_model(full, str(orig_p))
    nmod.save_model(recon, str(recon_p))

    mdf = nmod.metrics_from_paths(str(orig_p), str(recon_p))
    assert hasattr(mdf, "to_csv")
    assert set(["mse_state", "deg_l1"]) <= set(mdf.columns)
    assert len(mdf) == 1


def test_simulation_shapes(tmp_path: Path):
    full = nmod.generate_full_network(
        nodes=20,
        model="ba",
        ba_m=2,
        seed=1,
        state_init="zeros",
    )
    steps = 10
    traj = nmod.simulate_states(
        full,
        steps=steps,
        dt=0.1,
        leak=0.1,
        input_drive=0.05,
        noise_std=0.0,
        seed=123,
    )
    assert traj.shape == (steps + 1, 20)
    assert np.isfinite(traj).all()
