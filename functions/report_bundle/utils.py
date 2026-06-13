from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import re

import matplotlib.pyplot as plt
import numpy as np


def reports_dir(out_dir: Optional[Path] = None) -> Path:
    output = Path(out_dir) if out_dir is not None else Path.cwd() / "reports"
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_close(fig, filepath: Path, dpi: int = 200) -> Path:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


def to_1d_timeseries(mu: np.ndarray) -> np.ndarray:
    values = np.asarray(mu, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"Expected [T,d], got shape {values.shape}")
    return values.mean(axis=1)


def to_1d_variance(cov: np.ndarray) -> np.ndarray:
    values = np.asarray(cov, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"Expected [T,d,d], got shape {values.shape}")
    return np.diagonal(values, axis1=1, axis2=2).mean(axis=1)


def clip_T(y: np.ndarray, v: np.ndarray, T_plot: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    total = min(len(y), len(v))
    if T_plot is not None:
        total = min(total, int(T_plot))
    return y[:total], v[:total], total


def safe_std_from_var(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(v, dtype=float), 0.0))


def tail_slice(y: np.ndarray, v: np.ndarray, K: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if K is None:
        return y, v
    count = int(K)
    if count <= 0:
        raise ValueError("K must be a positive integer.")
    return (y, v) if len(y) <= count else (y[-count:], v[-count:])


def slug_filename(value: str, maxlen: int = 140) -> str:
    slug = re.sub(r"[^\w.\-]+", "_", value).strip("._-")
    return slug[:maxlen]
