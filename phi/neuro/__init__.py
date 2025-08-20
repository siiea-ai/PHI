from __future__ import annotations

import importlib.util
from pathlib import Path as _Path

# Load the sibling file 'phi/neuro.py' as a separate module name to avoid package collision
_impl = None
try:
    _impl_path = _Path(__file__).resolve().parents[1] / "neuro.py"
    spec = importlib.util.spec_from_file_location("phi._neuro_impl", str(_impl_path))
    _mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(_mod)  # type: ignore[attr-defined]
    _impl = _mod
except Exception as e:  # defer error to attribute access
    _impl = e  # type: ignore[assignment]

# Re-export BCI utilities from the submodule package
from .bci import (  # noqa: E402
    Scheduler,
    ConstantScheduler,
    CosineScheduler,
    CosineWithPhiRestarts,
    BCIConfig,
    Environment,
    OnlineLinearDecoder,
    Controller,
    simulate,
)

# Names to forward from the implementation module file
__forward_names__ = [
    "NeuroConfig",
    "generate_full_network",
    "compress_network",
    "expand_network",
    "load_network_any",
    "simulate_states",
    "save_model",
    "load_model",
    "save_adjacency_image",
    "metrics_from_paths",
    "load_signal_any",
    "make_pulse_signal",
]

if isinstance(_impl, Exception):
    def __getattr__(name: str):
        if name in __forward_names__:
            raise ImportError(f"phi.neuro failed to load implementation: {_impl}")
        raise AttributeError(name)
else:
    for _n in __forward_names__:
        globals()[_n] = getattr(_impl, _n)

__all__ = __forward_names__ + [
    "Scheduler",
    "ConstantScheduler",
    "CosineScheduler",
    "CosineWithPhiRestarts",
    "BCIConfig",
    "Environment",
    "OnlineLinearDecoder",
    "Controller",
    "simulate",
]