from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import importlib


@dataclass(frozen=True)
class ResolvedModelFunctions:
    process_name: str
    dynamics_name: str
    likelihood_name: str
    process_build: Callable
    dynamics: Callable
    likelihood: Callable


def resolve_model_functions(*, generative_process: str, dynamics: str, likelihood: str) -> ResolvedModelFunctions:
    process_name = _normalise_process_name(generative_process)
    try:
        process_module = importlib.import_module(f"functions.generative_process.{process_name}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unknown generative process: {generative_process}") from exc
    if not hasattr(process_module, "build"):
        raise ValueError(f"Generative process has no build() function: {generative_process}")

    dynamics_module = importlib.import_module("functions.generative_model.dynamics")
    likelihood_module = importlib.import_module("functions.generative_model.likelihood")
    if not hasattr(dynamics_module, dynamics):
        raise ValueError(f"Unknown dynamics function: {dynamics}")
    if not hasattr(likelihood_module, likelihood):
        raise ValueError(f"Unknown likelihood function: {likelihood}")

    return ResolvedModelFunctions(
        process_name=process_name,
        dynamics_name=dynamics,
        likelihood_name=likelihood,
        process_build=getattr(process_module, "build"),
        dynamics=getattr(dynamics_module, dynamics),
        likelihood=getattr(likelihood_module, likelihood),
    )


def validate_model_names(*, generative_process: str, dynamics: str, likelihood: str) -> None:
    resolve_model_functions(generative_process=generative_process, dynamics=dynamics, likelihood=likelihood)


def resolve_process_build(generative_process: str) -> Callable:
    process_name = _normalise_process_name(generative_process)
    try:
        process_module = importlib.import_module(f"functions.generative_process.{process_name}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unknown generative process: {generative_process}") from exc
    if not hasattr(process_module, "build"):
        raise ValueError(f"Generative process has no build() function: {generative_process}")
    return getattr(process_module, "build")


def _normalise_process_name(name: str) -> str:
    return str(name).split("/")[0]
