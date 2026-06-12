from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


OUTPUT_NAMES = (
    "vfe",
    "accuracy",
    "complexity",
    "gen_sensations",
    "gen_predictions",
    "x_clean",
    "x_noisy",
    "y",
    "gen_x_estimates",
    "theta",
    "lambda_x",
    "lambda_y",
    "cov_lambda_x",
    "cov_lambda_y",
    "cov_theta",
    "y_white_noise",
    "y_sigma_schedule",
    "y_colored_noise",
    "y_context_lengths",
    "x_white_noise",
    "x_sigma_schedule",
    "x_colored_noise",
    "x_context_lengths",
    "prior_theta_eta",
    "prior_theta_pi",
    "free_action",
    "mse",
)
RAW_OUTPUT_NAMES = OUTPUT_NAMES[:-2]
DERIVED_PRECISION_NAMES = ("x_pi_schedule", "y_pi_schedule")


@dataclass(frozen=True)
class RunOutputs:
    raw_arrays: Mapping[str, np.ndarray]
    free_action: float
    mse: float

    @classmethod
    def from_tuple(cls, outputs: tuple[Any, ...]) -> "RunOutputs":
        if len(outputs) != len(OUTPUT_NAMES):
            raise ValueError(f"ODEM output length must be {len(OUTPUT_NAMES)}, got {len(outputs)}")
        values = dict(zip(OUTPUT_NAMES, outputs, strict=True))
        return cls(
            raw_arrays={name: to_numpy(values[name]) for name in RAW_OUTPUT_NAMES},
            free_action=scalar_float(values["free_action"]),
            mse=scalar_float(values["mse"]),
        )

    @property
    def derived_arrays(self) -> dict[str, np.ndarray]:
        return derive_precision_schedules(self.raw_arrays)

    @property
    def snapshot_scalars(self) -> dict[str, float]:
        return {"fa": self.free_action, "mse": self.mse}


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def scalar_float(value: Any) -> float:
    array = to_numpy(value)
    if array.shape not in {(), (1,)}:
        raise ValueError(f"value must be a finite scalar, got shape {array.shape}")
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError("value must be a finite scalar")
    return result


def derive_precision_schedules(raw_arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    schedules = {}
    for sigma_name, precision_name in (("x_sigma_schedule", "x_pi_schedule"), ("y_sigma_schedule", "y_pi_schedule")):
        sigma = np.asarray(raw_arrays[sigma_name], dtype=float)
        if not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError(f"{sigma_name} must contain positive finite values")
        schedules[precision_name] = 1 / (sigma**2)
    return schedules
