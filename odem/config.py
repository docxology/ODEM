from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import importlib
import itertools

from ruamel.yaml import YAML
import torch


SCHEDULE_MODES = {"linear", "exp", "sigmoid", "log", "gaussian"}


@dataclass(frozen=True)
class ExperimentCombo:
    combo_index: int
    kx: int
    ky: int
    gp_name: str
    dt: float
    T: float
    f_name: str
    g_name: str
    E_theta: torch.Tensor
    sigma_theta: torch.Tensor
    E_pi_x: torch.Tensor
    sigma_lambda_x: torch.Tensor
    E_pi_y: torch.Tensor
    sigma_lambda_y: torch.Tensor
    nu_x: torch.Tensor
    kappa_x: float
    lambda_eta_adapt: bool
    lambda_eta_rate: float
    lambda_eta_t_0: float
    lambda_eta_gamma: float
    lambda_interval: int
    lambda_beta: float
    theta_eta_adapt: bool
    theta_eta_rate: float
    theta_eta_t_0: float
    theta_eta_gamma: float
    theta_interval: int
    theta_beta: float
    carry_cov: bool
    jitter: float
    algorithm_name: str = "ODEM"
    device: str = "cpu"

    def to_legacy_tuple(self) -> tuple[Any, ...]:
        return (
            self.kx,
            self.ky,
            self.gp_name,
            self.dt,
            self.T,
            self.f_name,
            self.g_name,
            self.E_theta,
            self.sigma_theta,
            self.E_pi_x,
            self.sigma_lambda_x,
            self.E_pi_y,
            self.sigma_lambda_y,
            self.nu_x,
            self.kappa_x,
            self.lambda_eta_adapt,
            self.lambda_eta_rate,
            self.lambda_eta_t_0,
            self.lambda_eta_gamma,
            self.lambda_interval,
            self.lambda_beta,
            self.theta_eta_adapt,
            self.theta_eta_rate,
            self.theta_eta_t_0,
            self.theta_eta_gamma,
            self.theta_interval,
            self.theta_beta,
            self.carry_cov,
            self.jitter,
            self.algorithm_name,
            self.device,
        )


@dataclass(frozen=True)
class ParameterSweep:
    source: Path
    raw: dict[str, Any]
    noise: dict[str, dict[str, Any]]
    axes: tuple[tuple[Any, ...], ...]
    combos: tuple[ExperimentCombo, ...]

    @property
    def combination_count(self) -> int:
        return len(self.combos)

    def to_legacy_parameters(self) -> list[Any]:
        return [list(axis) if len(axis) != 1 else axis[0] for axis in self.axes]


def load_sweep(path: str | Path = "parameters.yaml") -> ParameterSweep:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file_obj:
        params = YAML(typ="safe").load(file_obj)

    _validate_top_level(params)
    _validate_model_modules(params)
    noise = _build_noise(params)
    axes = _build_axes(params)

    combos = tuple(
        ExperimentCombo(combo_index=idx, **dict(zip(_combo_field_names(), values, strict=True)))
        for idx, values in enumerate(itertools.product(*axes))
    )
    return ParameterSweep(source=config_path, raw=params, noise=noise, axes=axes, combos=combos)


def _combo_field_names() -> tuple[str, ...]:
    return (
        "kx",
        "ky",
        "gp_name",
        "dt",
        "T",
        "f_name",
        "g_name",
        "E_theta",
        "sigma_theta",
        "E_pi_x",
        "sigma_lambda_x",
        "E_pi_y",
        "sigma_lambda_y",
        "nu_x",
        "kappa_x",
        "lambda_eta_adapt",
        "lambda_eta_rate",
        "lambda_eta_t_0",
        "lambda_eta_gamma",
        "lambda_interval",
        "lambda_beta",
        "theta_eta_adapt",
        "theta_eta_rate",
        "theta_eta_t_0",
        "theta_eta_gamma",
        "theta_interval",
        "theta_beta",
        "carry_cov",
        "jitter",
        "algorithm_name",
        "device",
    )


def _build_axes(params: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    priors = params["priors"]
    opt = params["optimizer"]
    gp = params["gp"]
    gm = params["gm"]

    tensor_axis = lambda values: tuple(torch.tensor(v, dtype=torch.float64) for v in _as_axis(values))
    nu_axis = tuple(torch.tensor(v, dtype=torch.float64) for v in _as_axis(opt["x"]["nu"]))

    axes: tuple[Iterable[Any], ...] = (
        (int(gm["kx"]),),
        (int(gm["ky"]),),
        (str(gp["name"]),),
        (float(gp["dt"]),),
        (float(gp["T"]),),
        (str(gm["dynamics"]),),
        (str(gm["likelihood"]),),
        tensor_axis(priors["theta"]["E_theta"]),
        tensor_axis(priors["theta"]["sigma_theta"]),
        tensor_axis(priors["lambda"]["E_pi_x"]),
        tensor_axis(priors["lambda"]["sigma_lambda_x"]),
        tensor_axis(priors["lambda"]["E_pi_y"]),
        tensor_axis(priors["lambda"]["sigma_lambda_y"]),
        nu_axis,
        tuple(float(v) for v in _as_axis(opt["x"]["kappa_x"])),
        (bool(opt["lambda"]["adapt"]),),
        (float(opt["lambda"]["eta"]["rate"]),),
        (float(opt["lambda"]["eta"]["t_0"]),),
        (float(opt["lambda"]["eta"]["gamma"]),),
        (int(opt["lambda"]["inter"]),),
        tuple(float(v) for v in _as_axis(opt["lambda"]["beta"])),
        (bool(opt["theta"]["adapt"]),),
        (float(opt["theta"]["eta"]["rate"]),),
        (float(opt["theta"]["eta"]["t_0"]),),
        (float(opt["theta"]["eta"]["gamma"]),),
        tuple(int(v) for v in _as_axis(opt["theta"]["inter"])),
        tuple(float(v) for v in _as_axis(opt["theta"]["beta"])),
        (bool(opt["carry_cov"]),),
        (float(opt["jitter"]),),
        ("ODEM",),
        ("cpu",),
    )
    return tuple(tuple(axis) for axis in axes)


def _build_noise(params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    gp_noise = params["gp"]["noise"]
    gm_noise = params["gm"]["noise"]

    return {
        "y": {
            "y_wn_mu": float(gm_noise["y_wn_mu"]),
            "y_wn_sigma": gm_noise["y_wn_sigma"],
            "y_cn_kernel_size": int(gm_noise["y_cn_kernel_size"]),
            "y_cn_kernel_sigma": float(gm_noise["y_cn_kernel_sigma"]),
            "y_h_value": torch.tensor(0.0, dtype=torch.float64, requires_grad=True),
            "y_lambda_value": torch.tensor(1 / (float(gm_noise["y_cn_kernel_sigma"]) ** 2), dtype=torch.float64),
        },
        "x": {
            "x_wn_mu": float(gp_noise["x_wn_mu"]),
            "x_wn_sigma": gp_noise["x_wn_sigma"],
            "x_cn_kernel_size": int(gp_noise["x_cn_kernel_size"]),
            "x_cn_kernel_sigma": float(gp_noise["x_cn_kernel_sigma"]),
            "x_h_value": torch.tensor(0.0, dtype=torch.float64, requires_grad=True),
            "x_lambda_value": torch.tensor(1 / (float(gp_noise["x_cn_kernel_sigma"]) ** 2), dtype=torch.float64),
        },
    }


def _validate_top_level(params: dict[str, Any]) -> None:
    for key in ("priors", "optimizer", "gp", "gm"):
        if key not in params:
            raise ValueError(f"Missing required configuration section: {key}")

    gm = params["gm"]
    gp = params["gp"]
    kx, ky = int(gm["kx"]), int(gm["ky"])
    if kx != ky + 1:
        raise ValueError("kx must equal ky + 1 for generalised-coordinate consistency")
    if float(gp["dt"]) <= 0 or float(gp["T"]) <= 0:
        raise ValueError("gp.dt and gp.T must be positive")

    _validate_noise("x", params["gp"]["noise"], "x")
    _validate_noise("y", params["gm"]["noise"], "y")
    _validate_positive_axis("E_pi_x", params["priors"]["lambda"]["E_pi_x"])
    _validate_positive_axis("E_pi_y", params["priors"]["lambda"]["E_pi_y"])
    _validate_positive_axis("sigma_lambda_x", params["priors"]["lambda"]["sigma_lambda_x"])
    _validate_positive_axis("sigma_lambda_y", params["priors"]["lambda"]["sigma_lambda_y"])
    _validate_positive_axis("sigma_theta", params["priors"]["theta"]["sigma_theta"])


def _validate_model_modules(params: dict[str, Any]) -> None:
    gp_name = str(params["gp"]["name"])
    f_name = str(params["gm"]["dynamics"])
    g_name = str(params["gm"]["likelihood"])

    try:
        importlib.import_module(f"functions.generative_process.{gp_name.split('/')[0]}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unknown generative process: {gp_name}") from exc

    dynamics = importlib.import_module("functions.generative_model.dynamics")
    likelihood = importlib.import_module("functions.generative_model.likelihood")
    if not hasattr(dynamics, f_name):
        raise ValueError(f"Unknown dynamics function: {f_name}")
    if not hasattr(likelihood, g_name):
        raise ValueError(f"Unknown likelihood function: {g_name}")


def _validate_noise(label: str, noise: dict[str, Any], prefix: str) -> None:
    kernel_size = int(noise[f"{prefix}_cn_kernel_size"])
    kernel_sigma = float(noise[f"{prefix}_cn_kernel_sigma"])
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"{prefix}_cn_kernel_size must be odd and positive")
    if kernel_sigma <= 0:
        raise ValueError(f"{prefix}_cn_kernel_sigma must be positive")

    contexts = noise[f"{prefix}_wn_sigma"]
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"{label} white-noise sigma schedule must be a non-empty list")
    for context in contexts:
        if len(context) != 3:
            raise ValueError(f"{label} white-noise schedule entries must be [start, end, mode]")
        start, end, mode = context
        if float(start) < 0 or float(end) < 0:
            raise ValueError(f"{label} white-noise sigma values must be non-negative")
        if str(mode) not in SCHEDULE_MODES:
            raise ValueError(f"Unknown {label} noise schedule mode: {mode}")


def _validate_positive_axis(name: str, values: Any) -> None:
    for value in _as_axis(values):
        tensor = torch.as_tensor(value, dtype=torch.float64)
        if torch.any(tensor <= 0):
            raise ValueError(f"{name} values must be positive")


def _as_axis(value: Any) -> tuple[Any, ...]:
    if isinstance(value, list):
        return tuple(value)
    return (value,)
