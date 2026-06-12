from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from odem.arrays import load_arrays


def finite_series(values, *, name: str) -> np.ndarray:
    series = np.asarray(values, dtype=float).reshape(-1)
    if len(series) == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.isfinite(series).all():
        raise ValueError(f"{name} contains nonfinite values")
    return series


def padded_limits(series: Iterable[np.ndarray], *, frac: float = 0.08) -> tuple[float, float]:
    values = np.concatenate([np.asarray(item, dtype=float).reshape(-1) for item in series])
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return -1.0, 1.0
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = max(1.0, abs(lo) * 0.1)
    else:
        pad = (hi - lo) * frac
    return lo - pad, hi + pad


def short_label(label: str, max_len: int = 18) -> str:
    return label if len(label) <= max_len else label[: max_len - 1] + "..."


def run_arrays(path: str | Path) -> dict[str, np.ndarray]:
    return load_arrays(path)
