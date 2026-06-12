from __future__ import annotations

import numpy as np

from functions.plotting import common, style


def plot_vfe_outputs(data) -> list:
    vfe = np.asarray(data.vfe, dtype=float).reshape(-1)
    return [
        common.plot_series(vfe, data.output_dir / "vfe.pdf", title="Variational free energy", labels=["VFE"]),
        common.plot_series(common.cumulative(vfe), data.output_dir / "fa.pdf", title="Cumulative free action", labels=["Free action"]),
    ]


def plot_precision_outputs(data) -> list:
    outputs = [
        _plot_with_ci(data.lambda_x, data.cov_lambda_x, data.output_dir / "lambda_x.pdf", title="State log-precision"),
        _plot_with_ci(data.lambda_y, data.cov_lambda_y, data.output_dir / "lambda_y.pdf", title="Observation log-precision"),
        _plot_with_ci(data.theta, data.cov_theta, data.output_dir / "theta.pdf", title="Theta posterior"),
    ]

    accuracy = np.asarray(data.accuracy, dtype=float).reshape(-1)
    complexity = np.asarray(data.complexity, dtype=float).reshape(-1)
    denom = accuracy + complexity
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    fig, ax = style.new_axis()
    ax.plot(accuracy / denom, label="accuracy fraction", color=style.COLORS[0])
    ax.plot(complexity / denom, label="complexity fraction", color=style.COLORS[3])
    ax.set_ylim(0.0, 1.0)
    style.style_axis(ax, title="Accuracy/complexity balance")
    outputs.append(style.save_pdf(fig, data.output_dir / "accuracy_complexity_tradeoff.pdf"))
    return outputs


def _plot_with_ci(mean, covariance, output_path, *, title: str):
    values = np.asarray(mean, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    std = common.diagonal_std(covariance)
    fig, ax = style.new_axis()
    t = np.arange(values.shape[0])
    for idx in range(values.shape[1]):
        color = style.COLORS[idx % len(style.COLORS)]
        y = values[:, idx]
        ax.plot(t, y, color=color, label=f"d{idx}")
        if std.ndim == 2 and idx < std.shape[1] and len(std) == len(y):
            ax.fill_between(t, y - 1.64 * std[:, idx], y + 1.64 * std[:, idx], color=color, alpha=0.16)
    style.style_axis(ax, title=title)
    return style.save_pdf(fig, output_path)
