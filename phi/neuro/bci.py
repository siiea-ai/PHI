"""
phi.neuro.bci

Closed-loop BCI simulation with online adaptive decoder and φ-scheduled controller.

Key components:
- Environment: generates neural-like signals influenced by a latent target and control input
- Decoder: online linear regression on signal features to predict latent target
- Controller: computes control to stabilize the target; gains scheduled via Scheduler
- Schedulers: constant, cosine, cosine-with-φ-restarts

This module is designed for reproducible experiments and integrates with phi.signals.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Callable

import math
import os
import json

import numpy as np

from phi.signals import feature_vector


# ------------------------------- Schedulers ------------------------------ #

class Scheduler:
    def value(self, t: int) -> float:
        raise NotImplementedError


@dataclass
class ConstantScheduler(Scheduler):
    v: float = 1.0

    def value(self, t: int) -> float:
        return float(self.v)


@dataclass
class CosineScheduler(Scheduler):
    period: int = 200
    min_v: float = 0.2
    max_v: float = 1.0

    def value(self, t: int) -> float:
        # 0..period cosine wave mapped to [min_v, max_v]
        c = 0.5 * (1 + math.cos(2 * math.pi * (t % self.period) / max(1, self.period)))
        return float(self.min_v + (self.max_v - self.min_v) * c)


@dataclass
class CosineWithPhiRestarts(Scheduler):
    T0: int = 200
    phi: float = (1 + 5 ** 0.5) / 2  # 1.618...
    min_v: float = 0.2
    max_v: float = 1.0

    def __post_init__(self) -> None:
        self._t_end = self.T0
        self._period = self.T0

    def value(self, t: int) -> float:
        if t >= self._t_end:
            # increase period by φ on each restart
            self._period = int(max(1, round(self._period * self.phi)))
            self._t_end += self._period
        c = 0.5 * (1 + math.cos(2 * math.pi * ((t) % self._period) / max(1, self._period)))
        return float(self.min_v + (self.max_v - self.min_v) * c)


@dataclass
class LinearScheduler(Scheduler):
    start_v: float = 1.0
    end_v: float = 0.2
    duration: int = 500

    def value(self, t: int) -> float:
        if self.duration <= 0:
            return float(self.end_v)
        # linear interpolate from start_v to end_v over [0, duration]
        alpha = min(max(int(t), 0), int(self.duration)) / float(self.duration)
        return float(self.start_v + (self.end_v - self.start_v) * alpha)


@dataclass
class StepScheduler(Scheduler):
    initial: float = 1.0
    gamma: float = 0.5
    period: int = 200

    def value(self, t: int) -> float:
        p = max(1, int(self.period))
        k = int(max(0, int(t)) // p)
        return float(self.initial * (self.gamma ** k))


# --- Cooperative interrupt for cancellation/timeouts ---
class SimulationInterrupt(Exception):
    """Raised by callers via on_step to cooperatively stop simulate()."""
    pass


# -------------------------------- Config -------------------------------- #

@dataclass
class BCIConfig:
    fs: float = 256.0            # sampling rate (Hz)
    window_sec: float = 1.0      # window duration per step
    steps: int = 1000            # number of closed-loop steps
    seed: int = 42

    # Environment dynamics
    process_noise: float = 0.02  # latent noise
    drift: float = 0.001         # slow drift magnitude
    ctrl_effect: float = 0.05    # how strong control nudges the latent

    # Decoder learning
    base_lr: float = 0.05

    # Controller gains
    base_gain: float = 0.5

    # Signal composition
    theta_hz: float = 6.0
    gamma_hz: float = 40.0
    noise_std: float = 1.0
    snr_scale: float = 0.6       # how strongly latent modulates the gamma envelope


# ------------------------------ Environment ----------------------------- #

class Environment:
    """Neural-like signal generator with a latent target influenced by control.

    - latent y_t evolves with slow drift and noise
    - signal x_t (window) is mixture of theta and gamma bands + noise
    - gamma envelope is modulated by sigmoid(y_t) for positive coupling
    - control u_t nudges the latent toward 0 (homeostatic objective)
    """

    def __init__(self, cfg: BCIConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.y = 0.0
        self.t = 0

    def step(self, u: float) -> Tuple[np.ndarray, float]:
        # latent dynamics: small drift + noise + control
        drift = self.cfg.drift * (self.rng.standard_normal())
        noise = self.cfg.process_noise * self.rng.standard_normal()
        self.y = self.y + drift + noise + self.cfg.ctrl_effect * float(u)

        # generate windowed signal
        n = int(self.cfg.fs * self.cfg.window_sec)
        t_arr = (np.arange(n) + self.t * n) / self.cfg.fs
        self.t += 1

        # oscillations
        theta = np.sin(2 * np.pi * self.cfg.theta_hz * t_arr)
        gamma_carrier = np.sin(2 * np.pi * self.cfg.gamma_hz * t_arr)
        # gamma envelope scales with latent via sigmoid
        env = 1.0 / (1.0 + np.exp(-self.y))  # [0,1]
        gamma = (0.3 + self.cfg.snr_scale * env) * gamma_carrier

        # colored-ish noise: AR(1) filter on white noise for 1/f-ish spectrum
        e = self.rng.standard_normal(n) * self.cfg.noise_std
        ar = np.empty_like(e)
        a = 0.9
        ar[0] = e[0]
        for i in range(1, n):
            ar[i] = a * ar[i - 1] + math.sqrt(1 - a * a) * e[i]

        x = theta + gamma + 0.5 * ar
        return x.astype(float), float(self.y)


# -------------------------------- Decoder ------------------------------- #

class OnlineLinearDecoder:
    def __init__(self, n_features: int, lr: float, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.w = 0.01 * self.rng.standard_normal(n_features)
        self.lr = float(lr)

    def predict(self, feats: np.ndarray) -> float:
        return float(np.dot(self.w, feats))

    def update(self, feats: np.ndarray, target: float, lr_scale: float = 1.0) -> float:
        y_hat = self.predict(feats)
        err = float(target - y_hat)
        self.w += (self.lr * lr_scale) * err * feats
        return err


# ------------------------------- Controller ----------------------------- #

@dataclass
class Controller:
    base_gain: float
    scheduler: Scheduler

    def control(self, y_hat: float, t: int) -> Tuple[float, float]:
        s = self.scheduler.value(t)
        u = -self.base_gain * s * float(y_hat)  # homeostatic
        return u, s


# --------------------------------- Runner -------------------------------- #

def simulate(
    cfg: BCIConfig,
    scheduler: Scheduler,
    out_dir: Optional[str] = None,
    save_features: bool = False,
    save_windows: bool = False,
    save_config: bool = True,
    on_step: Optional[Callable[[int], None]] = None,
) -> Dict[str, np.ndarray]:
    """Run a closed-loop simulation and return logs.

    Returns dict with keys: 'y', 'y_hat', 'err', 'sched', and summary metrics.
    If out_dir is provided, saves per-step CSV.
    """
    rng = np.random.default_rng(cfg.seed)
    env = Environment(cfg)

    # Warm-up feature dimension from a dummy window
    x0, _ = env.step(0.0)
    feats0, names = feature_vector(x0, cfg.fs)
    dec = OnlineLinearDecoder(n_features=feats0.size, lr=cfg.base_lr, seed=cfg.seed)
    ctrl = Controller(base_gain=cfg.base_gain, scheduler=scheduler)

    T = int(cfg.steps)
    y = np.zeros(T)
    y_true = np.zeros(T)  # latent used as target at each step
    y_hat = np.zeros(T)
    err = np.zeros(T)
    sched_vals = np.zeros(T)

    # Optional buffers for dataset creation
    feat_dim = feats0.size
    F = np.zeros((T, feat_dim)) if save_features else None
    Xw = np.zeros((T, x0.size)) if save_windows else None

    # include first step's features in loop properly
    x_buf = x0

    for t in range(T):
        # optional heartbeat to external caller
        if on_step is not None:
            try:
                on_step(t)
            except SimulationInterrupt:
                # bubble up cooperative interrupts (cancel/timeout)
                raise
            except Exception:
                pass
        # record current latent as supervised target (before stepping env)
        y_true[t] = env.y

        feats, _ = feature_vector(x_buf, cfg.fs)
        if F is not None:
            F[t, :] = feats
        if Xw is not None:
            Xw[t, :] = x_buf
        pred = dec.predict(feats)
        u, s_val = ctrl.control(pred, t)
        e = dec.update(feats, target=env.y, lr_scale=s_val)
        # step environment using control
        x_buf, y_next = env.step(u)

        y[t] = y_next
        y_hat[t] = pred
        err[t] = e
        sched_vals[t] = s_val

    # Metrics
    mse = float(np.mean((y - y_hat) ** 2))
    mae = float(np.mean(np.abs(y - y_hat)))
    # time to small error threshold
    thr = max(0.05, 0.1 * np.std(y))
    above = np.abs(y - y_hat) > thr
    ttc = int(np.argmax(~above)) if np.any(~above) else T

    logs: Dict[str, np.ndarray] = {
        "y": y,
        "y_true": y_true,
        "y_hat": y_hat,
        "err": err,
        "sched": sched_vals,
    }

    logs_summary: Dict[str, float] = {
        "mse": mse,
        "mae": mae,
        "ttc": float(ttc),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # Save per-step logs
        import pandas as pd  # local import to avoid hard dependency elsewhere

        df = pd.DataFrame({
            "t": np.arange(T),
            "y": y,
            "y_true": y_true,
            "y_hat": y_hat,
            "err": err,
            "sched": sched_vals,
        })
        df.to_csv(os.path.join(out_dir, "bci_timeseries.csv"), index=False)
        with open(os.path.join(out_dir, "bci_summary.json"), "w", encoding="utf-8") as f:
            json.dump(logs_summary, f, indent=2)

        # Optional: save features per-step
        if F is not None:
            fdf = pd.DataFrame(F, columns=names)
            fdf.insert(0, "t", np.arange(T))
            fdf["y_true"] = y_true
            fdf["y_hat"] = y_hat
            fdf["err"] = err
            fdf["sched"] = sched_vals
            fdf.to_csv(os.path.join(out_dir, "bci_features.csv"), index=False)

        # Optional: save raw windows
        if Xw is not None:
            np.savez(os.path.join(out_dir, "bci_windows.npz"), X=Xw, fs=cfg.fs, window_sec=cfg.window_sec)

        # Optional: save config and scheduler info for loaders
        if save_config:
            sch_info: Dict[str, Dict[str, float]]
            if isinstance(scheduler, ConstantScheduler):
                sch_info = {"type": "constant", "params": {"v": float(scheduler.v)}}
            elif isinstance(scheduler, CosineScheduler):
                sch_info = {"type": "cosine", "params": {"period": int(scheduler.period), "min_v": float(scheduler.min_v), "max_v": float(scheduler.max_v)}}
            elif isinstance(scheduler, CosineWithPhiRestarts):
                sch_info = {"type": "cosine_phi", "params": {"T0": int(scheduler.T0), "phi": float(scheduler.phi), "min_v": float(scheduler.min_v), "max_v": float(scheduler.max_v)}}
            elif isinstance(scheduler, LinearScheduler):
                sch_info = {"type": "linear", "params": {"start_v": float(scheduler.start_v), "end_v": float(scheduler.end_v), "duration": int(scheduler.duration)}}
            elif isinstance(scheduler, StepScheduler):
                sch_info = {"type": "step", "params": {"initial": float(scheduler.initial), "gamma": float(scheduler.gamma), "period": int(scheduler.period)}}
            else:
                sch_info = {"type": scheduler.__class__.__name__, "params": {}}

            cfg_dict = asdict(cfg)
            cfg_dict["feature_names"] = names
            cfg_dict["window_size"] = int(x0.size)
            cfg_dict["scheduler"] = sch_info
            with open(os.path.join(out_dir, "bci_config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg_dict, f, indent=2)

    # Attach summary
    logs.update({k: np.asarray([v]) for k, v in logs_summary.items()})
    return logs
