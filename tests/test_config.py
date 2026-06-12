import pytest
import torch

from odem.config import ParameterSweep, load_sweep
from tests.fixtures.configs import write_parameter_config


def test_parameter_sweep_expands_scalars_and_lists(tmp_path):
    sweep = load_sweep(write_parameter_config(tmp_path / "parameters.yaml"))

    assert isinstance(sweep, ParameterSweep)
    assert sweep.combination_count == 16
    assert sweep.noise["x"]["x_cn_kernel_size"] == 51

    first = sweep.combos[0]
    assert first.kx == 2
    assert first.ky == 1
    assert first.gp_name == "glv"
    assert first.f_name == "lorenz"
    assert first.g_name == "identity"
    assert first.kappa_x == 1.0
    assert torch.is_tensor(first.E_theta)
    assert first.to_parameter_tuple()[0:7] == (2, 1, "glv", 0.1, 1.0, "lorenz", "identity")


def test_parameter_sweep_rejects_invalid_generalised_coordinate_contract(tmp_path):
    config_path = write_parameter_config(tmp_path / "parameters.yaml", kx=3, ky=1)

    with pytest.raises(ValueError, match="kx must equal ky \\+ 1"):
        load_sweep(config_path)


def test_parameter_sweep_rejects_even_colored_noise_kernel(tmp_path):
    config_path = write_parameter_config(tmp_path / "parameters.yaml")
    text = config_path.read_text(encoding="utf-8").replace("x_cn_kernel_size: 51", "x_cn_kernel_size: 50")
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="x_cn_kernel_size must be odd"):
        load_sweep(config_path)


def test_parameter_sweep_rejects_empty_axis_lists(tmp_path):
    config_path = write_parameter_config(tmp_path / "parameters.yaml")
    text = config_path.read_text(encoding="utf-8").replace("kappa_x: [1.0, 0.5]", "kappa_x: []")
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="kappa_x.*non-empty"):
        load_sweep(config_path)


def test_parameter_sweep_rejects_string_booleans_for_adapt_flags(tmp_path):
    config_path = write_parameter_config(tmp_path / "parameters.yaml")
    text = (
        config_path.read_text(encoding="utf-8")
        .replace("carry_cov: true", "carry_cov: \"false\"")
        .replace("    adapt: true", "    adapt: \"false\"", 1)
    )
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="must be a boolean"):
        load_sweep(config_path)


def test_parameter_sweep_rejects_zero_noise_sigma_when_precision_schedule_is_saved(tmp_path):
    config_path = write_parameter_config(tmp_path / "parameters.yaml")
    text = config_path.read_text(encoding="utf-8").replace("x_wn_sigma: [[0.05, 0.05, linear]]", "x_wn_sigma: [[0.0, 0.05, linear]]")
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="white-noise sigma values must be positive"):
        load_sweep(config_path)
