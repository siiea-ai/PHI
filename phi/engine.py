from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict
import json

import numpy as np
import pandas as pd

# Optional plotting dependency
try:  # pragma: no cover
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    _HAS_MPL = False

from PIL import Image, ImageDraw

from . import fractal as fractal_mod
from . import harmonizer as harmonizer_mod
from .constants import PHI


@dataclass
class FractalConfig:
    strategy: str = "phi"            # "phi" or "ratio"
    depth: int = 4                   # phi strategy
    min_segment: int = 8             # phi strategy
    ratio: int = 2                   # ratio strategy
    smooth_window: int = 5           # phi expansion smoothing
    method: str = "interp"            # ratio expansion method: interp|hold


class FractalEngine:
    """High-level engine that integrates compression, expansion, harmonization,
    and optional visualization around the fractal models in `phi.fractal`.
    """

    def __init__(self, config: Optional[FractalConfig] = None) -> None:
        self.config = config or FractalConfig()
        self.models: Dict[str, dict] | None = None
        self.columns: List[str] | None = None
        self.input_rows: int = 0

    # ---- Compression / Expansion ----
    def compress(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, dict]:
        if columns is None:
            columns = df.select_dtypes(include=["number"]).columns.tolist()
        self.columns = columns
        self.input_rows = int(len(df))
        self.models = fractal_mod.compress_dataframe(
            df,
            columns=columns,
            depth=self.config.depth,
            min_segment=self.config.min_segment,
            strategy=self.config.strategy.lower(),
            ratio=self.config.ratio,
        )
        return self.models

    def expand(self, length: Optional[int] = None) -> pd.DataFrame:
        if self.models is None:
            raise ValueError("No models loaded. Call compress() or load_model() first.")
        return fractal_mod.expand_to_dataframe(
            self.models,
            length=length,
            smooth_window=self.config.smooth_window,
            method=self.config.method.lower(),
        )

    # ---- Persistence ----
    def save_model(self, path: str) -> None:
        if self.models is None:
            raise ValueError("No models to save. Call compress() first.")
        bundle = {
            "version": 1,
            "type": "phi-fractal-models",
            "phi": PHI,
            "input_rows": int(self.input_rows),
            "columns": list(self.models.keys()),
            "models": self.models,
            "config": self.config.__dict__,
        }
        fractal_mod.save_model(bundle, path)

    def load_model(self, path: str) -> None:
        bundle = fractal_mod.load_model(path)
        models = bundle.get("models")
        if not isinstance(models, dict):
            raise ValueError("Invalid model file: missing 'models' dict")
        self.models = models
        cols = bundle.get("columns")
        if isinstance(cols, list):
            self.columns = [str(c) for c in cols]
        self.input_rows = int(bundle.get("input_rows", 0))
        cfg = bundle.get("config")
        if isinstance(cfg, dict):
            # Shallow update to keep compatibility
            self.config = FractalConfig(**{**self.config.__dict__, **cfg})

    # ---- Harmonization ----
    def harmonize_split(self, s: pd.Series, total: float = 1.0, delta: float = 0.1) -> pd.DataFrame:
        return harmonizer_mod.harmonize_resource_split(s, total=total, delta=delta)

    def harmonize_backoff(self, s: pd.Series, base: float = 0.1, max_delay: float = 10.0, beta: float = 0.5) -> pd.DataFrame:
        return harmonizer_mod.harmonize_backoff(s, base=base, max_delay=max_delay, beta=beta)

    # ---- Visualization ----
    def plot_series(
        self,
        df_or_s: pd.DataFrame | pd.Series,
        output_path: str,
        column: Optional[str] = None,
        width: int = 800,
        height: int = 400,
        background: str = "white",
        line_color: str = "black",
    ) -> None:
        """Save a simple line plot of a Series/first column of DataFrame.

        Uses matplotlib if available, otherwise falls back to a minimal Pillow plot.
        """
        if isinstance(df_or_s, pd.DataFrame):
            if column is None:
                if len(df_or_s.columns) == 0:
                    raise ValueError("DataFrame has no columns to plot")
                column = str(df_or_s.columns[0])
            s = df_or_s[column].astype(float)
        else:
            s = df_or_s.astype(float)

        if _HAS_MPL:  # pragma: no cover
            fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
            ax.plot(np.arange(len(s)), s.to_numpy(), color=line_color, linewidth=1.5)
            ax.set_title(column or "series")
            ax.set_xlabel("t")
            ax.set_ylabel("value")
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)
            return

        # Fallback: simple Pillow-based plotting
        img = Image.new("RGB", (width, height), color=background)
        draw = ImageDraw.Draw(img)
        y = s.to_numpy()
        if len(y) < 2:
            img.save(output_path)
            return
        y_min, y_max = float(np.min(y)), float(np.max(y))
        if y_max <= y_min:
            y_max = y_min + 1.0
        # margins
        mx, my = 10, 10
        W, H = width - 2 * mx, height - 2 * my
        # scale points
        def pt(i: int, val: float) -> tuple[int, int]:
            x = mx + int((i / (len(y) - 1)) * W)
            yy = (val - y_min) / (y_max - y_min)
            ypix = my + int((1.0 - yy) * H)
            return x, ypix
        last = pt(0, float(y[0]))
        for i in range(1, len(y)):
            cur = pt(i, float(y[i]))
            draw.line([last, cur], fill=line_color, width=2)
            last = cur
        img.save(output_path)

    def plot_compare(
        self,
        original_df: pd.DataFrame,
        recon_df: pd.DataFrame,
        output_path: str,
        column: Optional[str] = None,
        width: int = 800,
        height: int = 400,
        background: str = "white",
        orig_color: str = "#1f77b4",  # blue
        recon_color: str = "#ff7f0e",  # orange
    ) -> None:
        """Overlay original vs reconstructed series and save a plot.

        Uses matplotlib if available, otherwise falls back to a minimal Pillow plot.
        """
        # choose column
        if column is None:
            # pick first common column
            common = [c for c in original_df.columns if c in recon_df.columns]
            if len(common) == 0:
                raise ValueError("No common columns between original and reconstructed data")
            column = str(common[0])

        a = pd.to_numeric(original_df[column], errors="coerce").astype(float).to_numpy()
        b = pd.to_numeric(recon_df[column], errors="coerce").astype(float).to_numpy()
        n = int(min(len(a), len(b)))
        if n <= 1:
            # not enough points to plot lines
            if _HAS_MPL:  # pragma: no cover
                fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
                ax.plot(np.arange(n), a[:n], color=orig_color, label="original")
                ax.plot(np.arange(n), b[:n], color=recon_color, label="reconstructed")
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_path)
                plt.close(fig)
            else:
                img = Image.new("RGB", (width, height), color=background)
                img.save(output_path)
            return

        a = a[:n]
        b = b[:n]

        if _HAS_MPL:  # pragma: no cover
            fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
            ax.plot(np.arange(n), a, color=orig_color, linewidth=1.5, label="original")
            ax.plot(np.arange(n), b, color=recon_color, linewidth=1.5, label="reconstructed")
            ax.set_title(f"compare: {column}")
            ax.set_xlabel("t")
            ax.set_ylabel("value")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path)
            plt.close(fig)
            return

        # Fallback Pillow plot overlay
        img = Image.new("RGB", (width, height), color=background)
        draw = ImageDraw.Draw(img)
        y_all = np.concatenate([a, b])
        y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            img.save(output_path)
            return
        if y_max <= y_min:
            y_max = y_min + 1.0
        mx, my = 10, 10
        W, H = width - 2 * mx, height - 2 * my
        def pt(i: int, val: float) -> tuple[int, int]:
            x = mx + int((i / (n - 1)) * W)
            yy = (val - y_min) / (y_max - y_min)
            ypix = my + int((1.0 - yy) * H)
            return x, ypix
        # plot original
        last = pt(0, float(a[0]))
        for i in range(1, n):
            cur = pt(i, float(a[i]))
            draw.line([last, cur], fill=orig_color, width=2)
            last = cur
        # plot reconstructed
        last = pt(0, float(b[0]))
        for i in range(1, n):
            cur = pt(i, float(b[i]))
            draw.line([last, cur], fill=recon_color, width=2)
            last = cur
        img.save(output_path)

    def analyze(
        self,
        original_df: pd.DataFrame,
        recon_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute error/quality metrics comparing original vs reconstructed series.

        Returns a DataFrame with metrics per column (mae, rmse, r2, corr, etc.)
        and approximate compression info when models are present.
        """
        if columns is None:
            columns = [c for c in original_df.columns if pd.api.types.is_numeric_dtype(original_df[c])]
        rows = []
        for col in columns:
            if col not in recon_df.columns:
                continue
            a = pd.to_numeric(original_df[col], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(recon_df[col], errors="coerce").to_numpy(dtype=float)
            n = int(min(len(a), len(b)))
            if n == 0:
                continue
            a, b = a[:n], b[:n]
            err = a - b
            mae = float(np.nanmean(np.abs(err)))
            mse = float(np.nanmean(err ** 2))
            rmse = float(np.sqrt(mse))
            mask = np.isfinite(a) & (np.abs(a) > 1e-12)
            mape = float(np.nanmean(np.abs(err[mask] / a[mask])) * 100.0) if np.any(mask) else float("nan")
            ss_res = float(np.nansum(err ** 2))
            ss_tot = float(np.nansum((a - np.nanmean(a)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            corr = float(np.corrcoef(a, b)[0, 1]) if n > 1 else float("nan")
            rows.append({
                "column": col,
                "n": n,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "mape_pct": mape,
                "r2": r2,
                "corr": corr,
                "orig_min": float(np.nanmin(a)),
                "orig_max": float(np.nanmax(a)),
                "orig_mean": float(np.nanmean(a)),
                "orig_std": float(np.nanstd(a)),
                "recon_min": float(np.nanmin(b)),
                "recon_max": float(np.nanmax(b)),
                "recon_mean": float(np.nanmean(b)),
                "recon_std": float(np.nanstd(b)),
            })
        dfm = pd.DataFrame(rows)
        # compression info (best-effort)
        model_size_bytes = float("nan")
        if self.models is not None:
            try:
                model_size_bytes = float(len(json.dumps(self.models)))
            except Exception:
                model_size_bytes = float("nan")
        try:
            raw_size_bytes = float(original_df[columns].to_numpy().nbytes)
        except Exception:
            raw_size_bytes = float("nan")
        ratio = raw_size_bytes / model_size_bytes if (np.isfinite(raw_size_bytes) and np.isfinite(model_size_bytes) and model_size_bytes > 0) else float("nan")
        if not dfm.empty:
            dfm["model_size_bytes"] = model_size_bytes
            dfm["raw_size_bytes"] = raw_size_bytes
            dfm["raw_to_model_ratio"] = ratio
        return dfm
