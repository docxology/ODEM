from __future__ import annotations

from pathlib import Path


def write_parameter_config(path: Path, *, kx: int = 2, ky: int = 1) -> Path:
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


def write_two_combo_config(path: Path) -> Path:
    path.write_text(
        """
priors:
  theta:
    E_theta: [30.0]
    sigma_theta: [9.0]
  lambda:
    E_pi_x: [500.0]
    sigma_lambda_x: [0.1]
    E_pi_y: [10.0, 20.0]
    sigma_lambda_y: [0.1]
optimizer:
  carry_cov: true
  x:
    name: ozaki
    nu: [-4]
    kappa_x: [1.0]
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
    inter: [2]
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
  T: 0.3
gm:
  dynamics: lorenz
  likelihood: identity
  kx: 2
  ky: 1
  noise:
    y_wn_mu: 0.0
    y_wn_sigma: [[0.1, 0.1, linear]]
    y_cn_kernel_size: 51
    y_cn_kernel_sigma: 0.005
""",
        encoding="utf-8",
    )
    return path
