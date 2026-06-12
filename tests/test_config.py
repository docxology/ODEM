from pathlib import Path

import pytest
import torch

from odem.config import ParameterSweep, load_sweep


def _write_config(path: Path, *, kx: int = 2, ky: int = 1) -> Path:
    path.write_text(
        f"""
priors:
  theta:
    E_theta: [30.0]
    sigma_theta: [9.0]
  lambda:
    E_pi_x: [500.0]
    sigma_lambda_x: [0.1]
    E_pi_y: [10.0, 20.0]
    sigma_lambda_y: [0.1, 0.5]
optimizer:
  carry_cov: true
  x:
    name: ozaki
    nu: [-4]
    kappa_x: [1.0, 0.5]
  lambda:
    adapt: true
    eta:
      rate: 0.0001
      t_0: 10
      gamma: 0.3
    inter: 1
    beta: [0.0]
  theta:
    adapt: true
    eta:
      rate: 0.0001
      t_0: 10
      gamma: 0.3
    inter: [2, 4]
    beta: [0.0]
  jitter: 1e-6
gp:
  noise:
    x_wn_mu: 0.0
    x_wn_sigma: [[0.05, 0.05, linear]]
    x_cn_kernel_size: 51
    x_cn_kernel_sigma: 0.005
  name: glv
  dt: 0.1
  T: 1.0
gm:
  dynamics: lorenz
  likelihood: identity
  kx: {kx}
  ky: {ky}
  noise:
    y_wn_mu: 0.0
    y_wn_sigma: [[0.1, 0.1, linear]]
    y_cn_kernel_size: 51
    y_cn_kernel_sigma: 0.005
""",
        encoding="utf-8",
    )
    return path


def test_parameter_sweep_expands_scalars_and_lists(tmp_path):
    sweep = load_sweep(_write_config(tmp_path / "parameters.yaml"))

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
    assert first.to_legacy_tuple()[0:7] == (2, 1, "glv", 0.1, 1.0, "lorenz", "identity")


def test_parameter_sweep_rejects_invalid_generalised_coordinate_contract(tmp_path):
    config_path = _write_config(tmp_path / "parameters.yaml", kx=3, ky=1)

    with pytest.raises(ValueError, match="kx must equal ky \\+ 1"):
        load_sweep(config_path)


def test_parameter_sweep_rejects_even_colored_noise_kernel(tmp_path):
    config_path = _write_config(tmp_path / "parameters.yaml")
    text = config_path.read_text(encoding="utf-8").replace("x_cn_kernel_size: 51", "x_cn_kernel_size: 50")
    config_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="x_cn_kernel_size must be odd"):
        load_sweep(config_path)
