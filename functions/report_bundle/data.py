from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import json

import numpy as np

from odem.analysis import summarize_run
from odem.arrays import load_optional_array


@dataclass(frozen=True)
class ReportCombo:
    e_pi_y: float | None
    kappa_x: float | None
    theta_interval: int | None
    theta_beta: float | None
    lambda_beta: float | None


def compute_best_for_kx(parent_dir: Path, kx_value: int, e_pi_y_constraints: Sequence[float]) -> dict[float, dict[str, Any]]:
    sub_dir = Path(parent_dir) / f"kx={kx_value}"
    best = {float(value): _empty_record() for value in e_pi_y_constraints}
    if not sub_dir.exists():
        return best

    for folder in sorted(path for path in sub_dir.iterdir() if path.is_dir()):
        combo = load_report_combo(folder / "combo.json")
        if combo.e_pi_y is None or combo.e_pi_y not in best:
            continue
        try:
            summary = summarize_run(folder)
        except ValueError:
            continue
        if not summary.get("valid_for_ranking"):
            continue

        record = best[combo.e_pi_y]
        if combo.kappa_x is not None:
            record["kappa_x_all"].append(float(combo.kappa_x))
        free_action = summary.get("free_action")
        if free_action is None or free_action >= record["min_fa"]:
            continue

        record.update(
            {
                "min_fa": float(free_action),
                "mse": summary.get("mse"),
                "accuracy": summary.get("accuracy_total"),
                "complexity": summary.get("complexity_total"),
                "lambda_x": load_optional_array(folder / "lambda_x.npy"),
                "lambda_y": load_optional_array(folder / "lambda_y.npy"),
                "theta": load_optional_array(folder / "theta.npy"),
                "cov_lambda_x": load_optional_array(folder / "cov_lambda_x.npy"),
                "cov_lambda_y": load_optional_array(folder / "cov_lambda_y.npy"),
                "cov_theta": load_optional_array(folder / "cov_theta.npy"),
                "kappa_x": combo.kappa_x,
                "theta_interval": combo.theta_interval,
                "theta_beta": combo.theta_beta,
                "lambda_beta": combo.lambda_beta,
                "folder": folder,
            }
        )
    return best


def compute_best_by_kx(
    parent_dir: Path,
    kx_values: Sequence[int],
    e_pi_y_constraints: Sequence[float],
) -> dict[int, dict[float, dict[str, Any]]]:
    return {int(kx): compute_best_for_kx(parent_dir, int(kx), e_pi_y_constraints) for kx in kx_values}


def load_report_combo(path: Path) -> ReportCombo:
    if not path.exists():
        return ReportCombo(None, None, None, None, None)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        return ReportCombo(
            e_pi_y=_first_float(value.get("E_pi_y")),
            kappa_x=_optional_float(value.get("kappa_x")),
            theta_interval=_optional_int(value.get("theta_interval")),
            theta_beta=_optional_float(value.get("theta_beta")),
            lambda_beta=_optional_float(value.get("lambda_beta")),
        )
    if isinstance(value, list) and len(value) >= 27:
        return ReportCombo(
            e_pi_y=_first_float(value[11]),
            kappa_x=_optional_float(value[14]),
            theta_interval=_optional_int(value[25]),
            theta_beta=_optional_float(value[26]),
            lambda_beta=_optional_float(value[20]),
        )
    return ReportCombo(None, None, None, None, None)


def _empty_record() -> dict[str, Any]:
    return {
        "min_fa": float("inf"),
        "mse": None,
        "accuracy": None,
        "complexity": None,
        "lambda_x": None,
        "lambda_y": None,
        "theta": None,
        "cov_lambda_x": None,
        "cov_lambda_y": None,
        "cov_theta": None,
        "kappa_x": None,
        "kappa_x_all": [],
        "theta_interval": None,
        "theta_beta": None,
        "lambda_beta": None,
        "folder": None,
    }


def _first_float(value: Any) -> float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    return None if len(array) == 0 else float(array[0])


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(np.asarray(value, dtype=float).reshape(-1)[0])


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(np.asarray(value).reshape(-1)[0])
