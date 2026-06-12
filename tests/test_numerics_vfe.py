from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from functions.vfe_calculation.compute_covariance_posteriors import _quadratic_form as covariance_quadratic_form
from functions.vfe_calculation.compute_covariance_posteriors import compute as compute_covariance_posteriors
from functions.vfe_calculation.compute_vfe import _quadratic_form as vfe_quadratic_form


def test_quadratic_form_matches_explicit_matmul_for_flat_row_and_column_vectors():
    precision = torch.tensor([[2.0, 0.5, 0.25], [0.5, 3.0, 0.75], [0.25, 0.75, 4.0]], dtype=torch.float64)
    vectors = [
        torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64),
        torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64),
        torch.tensor([[1.0], [-2.0], [0.5]], dtype=torch.float64),
    ]

    for vector in vectors:
        flat = vector.reshape(-1)
        expected = flat @ precision @ flat
        assert torch.allclose(vfe_quadratic_form(vector, precision), expected)
        assert torch.allclose(covariance_quadratic_form(vector, precision), expected)


def test_covariance_posteriors_invalid_key_raises():
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    scalar = torch.tensor([1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="key"):
        compute_covariance_posteriors(
            tensor,
            tensor,
            lambda x: x[:, :1],
            lambda x, theta: x,
            scalar,
            scalar,
            torch.eye(1, dtype=torch.float64),
            scalar,
            scalar,
            torch.eye(1, dtype=torch.float64),
            scalar,
            scalar,
            torch.eye(1, dtype=torch.float64),
            scalar,
            scalar,
            scalar,
            scalar,
            "bad-key",
            1e-6,
            "cpu",
        )


def test_precision_plot_grids_round_trip_against_precision_lambda_functions():
    root = Path(__file__).resolve().parents[1]
    precision_to_log = _load_module(root / "functions" / "plotting" / "plot_precision_to_log(precision).py", "precision_to_log")
    log_to_precision = _load_module(root / "functions" / "plotting" / "plot_log(precision)_to_precision.py", "log_to_precision")

    expected_precision = np.array([[0.5, 1.0], [2.0, 4.0]])
    variance_precision = np.array([[0.1, 0.2], [0.3, 0.4]])
    eta = np.log(expected_precision) - 0.5 * np.log(1 + variance_precision / expected_precision**2)
    pi = 1 / np.log(1 + variance_precision / expected_precision**2)

    recovered_precision = np.exp(eta + 0.5 / pi)
    recovered_variance = (np.exp(1 / pi) - 1) * np.exp(2 * eta + 1 / pi)

    assert np.allclose(recovered_precision, expected_precision)
    assert np.allclose(recovered_variance, variance_precision)

    E_grid, Var_grid, eta_grid, pi_grid, _ = precision_to_log.compute_precision_to_log_grid(
        expected_precision_range=(0.5, 4.0),
        precision_variance_range=(0.1, 0.4),
        points=2,
    )
    recovered_E = np.exp(eta_grid + 0.5 / pi_grid)
    recovered_Var = (np.exp(1 / pi_grid) - 1) * np.exp(2 * eta_grid + 1 / pi_grid)
    assert np.allclose(recovered_E, E_grid)
    assert np.allclose(recovered_Var, Var_grid)

    eta_lambda, pi_lambda, expected_grid, variance_grid, _ = log_to_precision.compute_log_to_precision_grid(
        eta_range=(-1.0, 1.0),
        pi_range=(2.0, 4.0),
        points=2,
    )
    roundtrip_eta = np.log(expected_grid) - 0.5 * np.log(1 + variance_grid / expected_grid**2)
    roundtrip_pi = 1 / np.log(1 + variance_grid / expected_grid**2)
    assert np.allclose(roundtrip_eta, eta_lambda)
    assert np.allclose(roundtrip_pi, pi_lambda)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
