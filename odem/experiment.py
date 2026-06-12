from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import importlib
import json
import os
import traceback

import numpy as np

from algorithms import ODEM
from functions import make_json_safe
from functions.initialisation import seed
from functions.saving import serializable
from odem.analysis import summarize_run, write_summary_json
from odem.artifacts import RunBundle
from odem.config import ExperimentCombo, ParameterSweep, load_sweep
from odem.visualization import create_state_animation, dispatch_static_plots, render_diagnostic_dashboard


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


@dataclass(frozen=True)
class RunnerOptions:
    config_path: Path = Path("parameters.yaml")
    results_dir: Path = Path("results")
    logs_dir: Path = Path("logs")
    seed_value: int = 42
    start_index: int | None = None
    end_index: int | None = None
    max_combos: int | None = None
    use_slurm_env: bool = True
    static_plots: bool = True
    dashboard: bool = True
    animations: bool = False
    tqdm_disable: bool = False


class ExperimentRunner:
    def __init__(self, options: RunnerOptions):
        self.options = options

    def run(self) -> dict[str, Any]:
        seed.generate(self.options.seed_value)
        sweep = load_sweep(self.options.config_path)
        start, end = resolve_combo_slice(
            sweep.combination_count,
            start_index=self.options.start_index,
            end_index=self.options.end_index,
            max_combos=self.options.max_combos,
            use_slurm_env=self.options.use_slurm_env,
        )

        completed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []

        for combo in sweep.combos[start:end]:
            try:
                bundle = run_combo(
                    combo,
                    sweep,
                    results_dir=self.options.results_dir,
                    static_plots=self.options.static_plots,
                    dashboard=self.options.dashboard,
                    animations=self.options.animations,
                    tqdm_disable=self.options.tqdm_disable,
                )
                completed.append(make_json_safe.convert(combo.to_legacy_tuple()))
                summaries.append(summarize_run(bundle.path))
            except Exception as exc:  # keep sweep jobs moving while preserving details
                failed.append(
                    {
                        "combo": make_json_safe.convert(combo.to_legacy_tuple()),
                        "combo_idx": combo.combo_index,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        log_bundle = RunBundle.create(self.options.logs_dir, combo_index=None)
        log_bundle.save_json("completed_combos", completed)
        log_bundle.save_json("failed_combos", failed)
        log_bundle.save_json("summary", summaries)
        log_bundle.write_manifest(status="completed" if not failed else "completed_with_failures")

        if summaries:
            write_summary_json(summaries, self.options.results_dir / "summary.json")

        return {
            "completed": completed,
            "failed": failed,
            "summaries": summaries,
            "log_dir": str(log_bundle.path),
        }


def run_combo(
    combo: ExperimentCombo,
    sweep: ParameterSweep,
    *,
    results_dir: str | Path,
    static_plots: bool,
    dashboard: bool,
    animations: bool,
    tqdm_disable: bool,
) -> RunBundle:
    dynamics = importlib.import_module("functions.generative_model.dynamics")
    likelihood = importlib.import_module("functions.generative_model.likelihood")
    f = getattr(dynamics, combo.f_name)
    g = getattr(likelihood, combo.g_name)

    outputs = ODEM.start(
        combo.kx,
        combo.ky,
        f,
        g,
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
        tqdm_disable=tqdm_disable,
    )
    output = dict(zip(OUTPUT_NAMES, outputs, strict=True))

    free_action = float(output["free_action"].detach() if hasattr(output["free_action"], "detach") else output["free_action"])
    mse = float(output["mse"])
    run_id = f"C{combo.combo_index}_{free_action:.2f}_{mse:.4f}"
    bundle = RunBundle.create(results_dir, combo_index=combo.combo_index, run_id=_unique_run_id(results_dir, run_id))
    bundle.save_json("combo", make_json_safe.convert(combo.to_legacy_tuple()))

    for name in OUTPUT_NAMES[:-2]:
        bundle.save_array(name, output[name])

    y_sigma_schedule = np.asarray(output["y_sigma_schedule"])
    x_sigma_schedule = np.asarray(output["x_sigma_schedule"])
    bundle.save_array("y_pi_schedule", 1 / (y_sigma_schedule**2))
    bundle.save_array("x_pi_schedule", 1 / (x_sigma_schedule**2))

    snapshot = build_snapshot(combo, sweep.noise, output)
    bundle.save_json("snapshot", snapshot)

    if static_plots:
        for artifact_path in dispatch_static_plots(bundle.path):
            bundle.record_file(artifact_path, kind="visualization")
    if dashboard:
        bundle.record_file(render_diagnostic_dashboard(bundle.path), kind="visualization")
    if animations:
        bundle.record_file(create_state_animation(bundle.path), kind="animation")

    bundle.write_manifest(status="completed", extra={"summary": summarize_run(bundle.path)})
    return bundle


def build_snapshot(combo: ExperimentCombo, noise: dict[str, dict[str, Any]], output: dict[str, Any]) -> dict[str, Any]:
    free_action = output["free_action"]
    free_action_value = float(free_action.detach() if hasattr(free_action, "detach") else free_action)
    snapshot = {
        "device": combo.device,
        "algorithm_name": combo.algorithm_name,
        "priors": {
            "theta": {"E_theta": combo.E_theta, "sigma_theta": combo.sigma_theta},
            "lambda": {
                "E_pi_x": combo.E_pi_x,
                "sigma_lambda_x": combo.sigma_lambda_x,
                "E_pi_y": combo.E_pi_y,
                "sigma_lambda_y": combo.sigma_lambda_y,
            },
        },
        "optimizer": {
            "x": {"nu": combo.nu_x, "kappa_x": combo.kappa_x},
            "lambda": {
                "adapt": combo.lambda_eta_adapt,
                "eta": {
                    "rate": combo.lambda_eta_rate,
                    "t_0": combo.lambda_eta_t_0,
                    "gamma": combo.lambda_eta_gamma,
                },
                "inter": combo.lambda_interval,
                "beta": combo.lambda_beta,
            },
            "theta": {
                "adapt": combo.theta_eta_adapt,
                "eta": {
                    "rate": combo.theta_eta_rate,
                    "t_0": combo.theta_eta_t_0,
                    "gamma": combo.theta_eta_gamma,
                },
                "inter": combo.theta_interval,
                "beta": combo.theta_beta,
            },
            "jitter": combo.jitter,
            "carry_cov": combo.carry_cov,
        },
        "gp": {
            "noise": {
                "x_wn_mu": noise["x"]["x_wn_mu"],
                "x_wn_sigma": noise["x"]["x_wn_sigma"],
                "x_cn_kernel_size": noise["x"]["x_cn_kernel_size"],
                "x_cn_kernel_sigma": noise["x"]["x_cn_kernel_sigma"],
                "x_h_value": noise["x"]["x_h_value"],
                "x_lambda_value": noise["x"]["x_lambda_value"],
            },
            "name": combo.gp_name,
            "dt": combo.dt,
            "T": combo.T,
        },
        "gm": {
            "dynamics": combo.f_name,
            "likelihood": combo.g_name,
            "kx": combo.kx,
            "ky": combo.ky,
            "y_h_value": noise["y"]["y_h_value"],
            "y_lambda_value": noise["y"]["y_lambda_value"],
            "noise": {
                "y_wn_mu": noise["y"]["y_wn_mu"],
                "y_wn_sigma": noise["y"]["y_wn_sigma"],
                "y_cn_kernel_size": noise["y"]["y_cn_kernel_size"],
                "y_cn_kernel_sigma": noise["y"]["y_cn_kernel_sigma"],
            },
        },
        "fa": free_action_value,
        "mse": float(output["mse"]),
    }
    return {k: serializable.serialize(v) for k, v in snapshot.items()}


def resolve_combo_slice(
    total: int,
    *,
    start_index: int | None = None,
    end_index: int | None = None,
    max_combos: int | None = None,
    use_slurm_env: bool = True,
) -> tuple[int, int]:
    if start_index is None and end_index is None and use_slurm_env and "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
        splits = np.array_split(np.arange(total), task_count)
        if task_id >= len(splits) or len(splits[task_id]) == 0:
            return 0, 0
        start, end = int(splits[task_id][0]), int(splits[task_id][-1]) + 1
    else:
        start = 0 if start_index is None else int(start_index)
        end = total if end_index is None else int(end_index)

    start = max(0, min(start, total))
    end = max(start, min(end, total))
    if max_combos is not None:
        end = min(end, start + int(max_combos))
    return start, end


def _unique_run_id(results_dir: str | Path, run_id: str) -> str:
    base = Path(results_dir)
    candidate = run_id
    counter = 1
    while (base / candidate).exists():
        candidate = f"{run_id}_{counter}"
        counter += 1
    return candidate
