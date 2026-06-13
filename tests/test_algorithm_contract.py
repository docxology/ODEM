from __future__ import annotations

import numpy as np
import pytest
import torch

from algorithms import ODEM
from functions.initialisation import seed
from odem.config import load_sweep
from odem.model_registry import resolve_model_functions
from odem.numerics import ensure_finite_tensor, safe_precision_inverse
from odem.run_outputs import OUTPUT_NAMES, RunOutputs


def test_algorithm_smoke_outputs_are_finite_detached_and_quiet(capsys):
    seed.generate(42)
    sweep = load_sweep("configs/smoke.yaml")
    combo = sweep.combos[0]
    resolved = resolve_model_functions(
        generative_process=combo.gp_name,
        dynamics=combo.f_name,
        likelihood=combo.g_name,
    )

    raw_outputs = ODEM.start(
        combo.kx,
        combo.ky,
        resolved.dynamics,
        resolved.likelihood,
        combo.f_name,
        combo.gp_name,
        combo.dt,
        combo.T,
        combo.E_theta,
        combo.sigma_theta,
        combo.E_pi_x,
        combo.sigma_lambda_x,
        combo.E_pi_y,
        combo.sigma_lambda_y,
        combo.nu_x,
        combo.kappa_x,
        combo.lambda_eta_adapt,
        combo.lambda_eta_rate,
        combo.lambda_eta_t_0,
        combo.lambda_eta_gamma,
        combo.lambda_beta,
        combo.theta_eta_adapt,
        combo.theta_eta_rate,
        combo.theta_eta_t_0,
        combo.theta_eta_gamma,
        combo.theta_interval,
        combo.theta_beta,
        combo.jitter,
        sweep.noise,
        combo.carry_cov,
        combo.device,
        tqdm_disable=True,
    )
    captured = capsys.readouterr()
    outputs = RunOutputs.from_tuple(raw_outputs)

    assert len(raw_outputs) == len(OUTPUT_NAMES)
    assert "Free Action=" not in captured.out
    assert "VFE:" not in captured.out
    assert all(np.isfinite(array).all() for array in outputs.raw_arrays.values())
    assert all(np.isfinite(array).all() for array in outputs.derived_arrays.values())


def test_numerical_finite_checks_include_phase_and_timestep():
    with pytest.raises(ValueError, match="D-step t=3.*delta"):
        ensure_finite_tensor(torch.tensor([1.0, float("nan")]), name="delta", phase="D-step", timestep=3)


def test_safe_precision_inverse_uses_fallback_for_bad_covariances():
    previous = torch.eye(2, dtype=torch.float64) * 7.0
    bad_cov = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]], dtype=torch.float64)

    precision = safe_precision_inverse(bad_cov, previous_precision=previous, phase="M-step", timestep=4)

    assert torch.equal(precision, previous)
