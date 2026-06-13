from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    run_dir = Path(path)
    return {npy_path.stem: np.load(npy_path, allow_pickle=False) for npy_path in run_dir.glob("*.npy")}


def load_optional_array(path: str | Path) -> np.ndarray | None:
    filepath = Path(path)
    return np.load(filepath, allow_pickle=False) if filepath.exists() else None


def all_numeric_finite(arrays: Mapping[str, np.ndarray]) -> bool:
    return all(np.isfinite(array).all() for array in arrays.values() if np.issubdtype(array.dtype, np.number))


def state_estimate_issue(x: np.ndarray, estimates: np.ndarray) -> str | None:
    if x.ndim != 2:
        return f"x_noisy must be a 2D array, got shape {x.shape}"
    if estimates.ndim != 3:
        return f"gen_x_estimates must be a 3D array, got shape {estimates.shape}"
    if len(x) == 0:
        return "x_noisy must contain at least one time step"
    if estimates.shape[0] == 0 or estimates.shape[1] < 1:
        return f"gen_x_estimates must contain time steps and posterior samples, got shape {estimates.shape}"
    estimate = estimates[:, 0, :]
    if x.shape != estimate.shape:
        return f"x_noisy shape {x.shape} must match gen_x_estimates[:, 0, :] shape {estimate.shape}"
    if not np.isfinite(x).all():
        return "x_noisy contains nonfinite values"
    if not np.isfinite(estimate).all():
        return "gen_x_estimates contains nonfinite values"
    return None


def extract_state_estimate(x: np.ndarray, estimates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    issue = state_estimate_issue(x, estimates)
    if issue:
        raise ValueError(issue)
    return x, estimates[:, 0, :]
