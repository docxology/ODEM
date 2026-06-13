from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from odem.artifacts import RunBundle


def write_complete_bundle(
    root: Path,
    *,
    run_id: str = "complete-run",
    combo_index: int = 0,
    vfe: np.ndarray | None = None,
    x_noisy: np.ndarray | None = None,
    gen_x_estimates: np.ndarray | None = None,
    status: str = "completed",
) -> RunBundle:
    bundle = RunBundle.create(root, combo_index=combo_index, run_id=run_id)
    x = np.asarray([[1.0, 2.0], [2.0, 4.0]], dtype=float) if x_noisy is None else np.asarray(x_noisy, dtype=float)
    x_hat = (
        np.asarray([[[1.5, 1.5]], [[2.0, 5.0]]], dtype=float)
        if gen_x_estimates is None
        else np.asarray(gen_x_estimates, dtype=float)
    )
    timesteps = int(len(x))
    scalar_series = np.asarray([2.0, 3.0], dtype=float) if vfe is None else np.asarray(vfe, dtype=float)
    if len(scalar_series) != timesteps:
        scalar_series = np.resize(scalar_series, timesteps)

    arrays: dict[str, Any] = {
        "vfe": scalar_series,
        "accuracy": np.linspace(10.0, 20.0, timesteps),
        "complexity": np.linspace(12.0, 23.0, timesteps),
        "gen_sensations": np.ones((timesteps, 1)),
        "gen_predictions": np.ones((timesteps, 1)) * 1.5,
        "x_clean": x,
        "x_noisy": x,
        "y": np.ones((timesteps, 1)) * 0.5,
        "gen_x_estimates": x_hat,
        "theta": np.linspace(30.0, 29.5, timesteps).reshape(timesteps, 1),
        "lambda_x": np.linspace(6.0, 6.1, timesteps).reshape(timesteps, 1),
        "lambda_y": np.linspace(2.0, 2.1, timesteps).reshape(timesteps, 1),
        "cov_lambda_x": np.ones((timesteps, 1, 1)) * 0.1,
        "cov_lambda_y": np.ones((timesteps, 1, 1)) * 0.2,
        "cov_theta": np.ones((timesteps, 1, 1)) * 0.3,
        "y_white_noise": np.zeros((timesteps, 1)),
        "y_sigma_schedule": np.ones((timesteps, 1)) * 0.1,
        "y_colored_noise": np.zeros((timesteps, 1)),
        "y_context_lengths": np.ones(timesteps),
        "x_white_noise": np.zeros_like(x),
        "x_sigma_schedule": np.ones_like(x) * 0.05,
        "x_colored_noise": np.zeros_like(x),
        "x_context_lengths": np.ones(timesteps),
        "prior_theta_eta": np.ones((timesteps, 1)) * 30.0,
        "prior_theta_pi": np.ones((timesteps, 1)) * 0.1,
        "y_pi_schedule": np.ones((timesteps, 1)) * 100.0,
        "x_pi_schedule": np.ones_like(x) * 400.0,
    }
    for name, value in arrays.items():
        bundle.save_array(name, value)

    bundle.save_json("combo", {"combo_index": combo_index})
    bundle.save_json(
        "snapshot",
        {
            "fa": float(np.sum(scalar_series)) if np.isfinite(scalar_series).all() else None,
            "mse": 0.0,
            "gm": {"dynamics": "lorenz", "likelihood": "identity", "kx": 2, "ky": 1},
            "gp": {"name": "glv", "dt": 0.1, "T": 0.2},
        },
    )
    bundle.write_manifest(status=status)
    return bundle
