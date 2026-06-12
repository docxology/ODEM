from __future__ import annotations

from typing import Any

import torch


def quadratic_form(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    flat = vector.reshape(-1)
    return torch.dot(flat, matrix @ flat)


def ensure_finite_tensor(value: Any, *, name: str, phase: str, timestep: int | None = None) -> None:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if torch.isfinite(tensor).all():
        return
    where = phase if timestep is None else f"{phase} t={timestep}"
    raise ValueError(f"{where}: {name} contains nonfinite values")


def safe_precision_inverse(
    covariance: torch.Tensor | None,
    *,
    previous_precision: torch.Tensor | None = None,
    phase: str = "precision inverse",
    timestep: int | None = None,
) -> torch.Tensor:
    if covariance is None or not torch.is_tensor(covariance) or not torch.isfinite(covariance).all():
        if previous_precision is not None:
            return previous_precision
        where = phase if timestep is None else f"{phase} t={timestep}"
        raise ValueError(f"{where}: covariance is missing or nonfinite")
    try:
        precision = torch.linalg.inv(covariance.detach().clone())
    except RuntimeError:
        if previous_precision is not None:
            return previous_precision
        where = phase if timestep is None else f"{phase} t={timestep}"
        raise ValueError(f"{where}: covariance could not be inverted")
    ensure_finite_tensor(precision, name="precision", phase=phase, timestep=timestep)
    return precision
