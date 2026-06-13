from __future__ import annotations

import numpy as np

from functions.plotting import style


def plot_series(data, output_path, *, title: str, ylabel: str | None = None, labels: list[str] | None = None):
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    fig, ax = style.new_axis()
    for idx in range(values.shape[1]):
        label = labels[idx] if labels and idx < len(labels) else f"d{idx}"
        ax.plot(values[:, idx], color=style.COLORS[idx % len(style.COLORS)], label=label)
    style.style_axis(ax, title=title, ylabel=ylabel)
    return style.save_pdf(fig, output_path)


def cumulative(values):
    return np.cumsum(np.asarray(values, dtype=float).reshape(-1))


def diagonal_std(covariance) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim == 3:
        return np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
    if cov.ndim == 2:
        return np.sqrt(np.maximum(cov, 0.0))
    return np.zeros((len(cov), 1), dtype=float)
