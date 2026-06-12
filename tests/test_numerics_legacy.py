import importlib.util
from pathlib import Path

import numpy as np
import torch

from functions import lambda_to_precision, precision_to_lambda
from functions.noise_generation.generate_colored_noise import gaussian_kernel
from functions.initialisation import initialise_sequential
from odem.config import load_sweep
from odem.experiment import resolve_combo_slice


def test_precision_lambda_round_trip_preserves_expected_precision():
    expected_precision = torch.tensor(50.0, dtype=torch.float64)
    sigma_lambda = torch.tensor(0.5, dtype=torch.float64)

    lambda_eta, lambda_var = precision_to_lambda.compute(expected_precision, sigma_lambda)
    recovered_mean, recovered_var = lambda_to_precision.compute(lambda_eta, lambda_var)

    assert torch.allclose(recovered_mean, expected_precision)
    assert recovered_var > 0


def test_gaussian_kernel_degenerate_sigma_falls_back_to_delta():
    kernel = gaussian_kernel(size=5, sigma=1e-24)

    assert np.isfinite(kernel).all()
    assert np.isclose(kernel.sum(), 1.0)
    assert kernel[2] == 1.0


def test_default_parameter_yaml_loads_same_legacy_combo_count():
    sweep = load_sweep("parameters.yaml")
    legacy_parameters, noise = initialise_sequential.set("parameters.yaml")

    legacy_count = 1
    for value in legacy_parameters:
        legacy_count *= len(value) if isinstance(value, list) else 1

    assert sweep.combination_count == 3024
    assert legacy_count == sweep.combination_count
    assert noise["y"]["y_cn_kernel_size"] == 51


def test_smoke_config_is_single_combo_for_fast_runtime_checks():
    sweep = load_sweep("configs/smoke.yaml")

    assert sweep.combination_count == 1
    assert sweep.combos[0].T == 0.3


def test_resolve_combo_slice_uses_slurm_environment(monkeypatch):
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "3")

    assert resolve_combo_slice(10) == (4, 7)


def test_precision_plot_modules_are_import_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    module_paths = [
        repo_root / "functions" / "plotting" / "plot_precision_to_log(precision).py",
        repo_root / "functions" / "plotting" / "plot_log(precision)_to_precision.py",
    ]

    for idx, module_path in enumerate(module_paths):
        spec = importlib.util.spec_from_file_location(f"precision_plot_helper_{idx}", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

    assert not list(tmp_path.glob("*.pdf"))
