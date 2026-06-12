from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from functions.report_bundle.utils import reports_dir, save_close


def save_metric_by_constraint(best_by_kx: dict, metric: str, *, out_dir: Optional[Path] = None, filename: str) -> Path:
    output = reports_dir(out_dir) / filename
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160, constrained_layout=True)
    for kx, records in sorted(best_by_kx.items()):
        x = []
        y = []
        for constraint, record in sorted(records.items()):
            value = record.get(metric)
            if value is None or not np.isfinite(value):
                continue
            x.append(float(constraint))
            y.append(float(value))
        if x:
            ax.plot(x, y, marker="o", label=f"kx={kx}")
    ax.set_xlabel("E_pi_y")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    return save_close(fig, output)


def save_mse_vs_ratio(best_by_kx: dict, E_pi_x_fix: float, *, out_dir: Optional[Path] = None, filename: str = "mse_vs_ratio.png", **_: Any) -> Path:
    scaled = {}
    for kx, records in best_by_kx.items():
        scaled[kx] = {float(constraint) / float(E_pi_x_fix): record for constraint, record in records.items()}
    return save_metric_by_constraint(scaled, "mse", out_dir=out_dir, filename=filename)


def save_fa_decomposition(best_by_kx: dict, *, out_dir: Optional[Path] = None, filename: str = "fa_decomposition.png", **_: Any) -> Path:
    return save_metric_by_constraint(best_by_kx, "min_fa", out_dir=out_dir, filename=filename)


def save_kappa_best(best_by_kx: dict, *, out_dir: Optional[Path] = None, filename: str = "kappa_best.png", **_: Any) -> Path:
    return save_metric_by_constraint(best_by_kx, "kappa_x", out_dir=out_dir, filename=filename)


def save_theta_interval_best(best_by_kx: dict, *, out_dir: Optional[Path] = None, filename: str = "theta_interval_best.png", **_: Any) -> Path:
    return save_metric_by_constraint(best_by_kx, "theta_interval", out_dir=out_dir, filename=filename)


def save_theta_beta_best(best_by_kx: dict, *, out_dir: Optional[Path] = None, filename: str = "theta_beta_best.png", **_: Any) -> Path:
    return save_metric_by_constraint(best_by_kx, "theta_beta", out_dir=out_dir, filename=filename)


def save_lambda_beta_best(best_by_kx: dict, *, out_dir: Optional[Path] = None, filename: str = "lambda_beta_best.png", **_: Any) -> Path:
    return save_metric_by_constraint(best_by_kx, "lambda_beta", out_dir=out_dir, filename=filename)
