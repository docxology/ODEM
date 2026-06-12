from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def compute_log_to_precision_grid(eta_range=(-5.0, 5.0), pi_range=(1.0, 1024.0), points=200):
    eta_vals = np.linspace(*eta_range, points)
    pi_vals = np.linspace(*pi_range, points)
    eta_lambda, pi_lambda = np.meshgrid(eta_vals, pi_vals)
    expected_precision = np.exp(eta_lambda + 0.5 / pi_lambda)
    variance_precision = (np.exp(1 / pi_lambda) - 1) * np.exp(2 * eta_lambda + 1 / pi_lambda)
    std_precision = np.sqrt(variance_precision)
    return eta_lambda, pi_lambda, expected_precision, variance_precision, std_precision


def save_log_to_precision_plots(output_dir="."):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    eta_lambda, pi_lambda, expected_precision, variance_precision, std_precision = compute_log_to_precision_grid()

    saved = [
        _save_contour(
            eta_lambda,
            pi_lambda,
            expected_precision,
            levels=[0.5, 1.0, 2.0, 5.0, 10.0],
            filename=output / "precision_eta.pdf",
            title=r"Expected Value of Precision E[$\Pi$]",
            colorbar_label=r"E[$\Pi$]",
            x_label=r"$\eta_\Lambda$ (mean of log-precision)",
            y_label=r"$\Pi_\Lambda$ (precision of log-precision)",
            legend_label=lambda level: rf"E[$\Pi$] = {level}",
            cmap="viridis",
        ),
        _save_contour(
            eta_lambda,
            pi_lambda,
            variance_precision,
            levels=[0.01, 0.1, 1.0, 10.0, 100.0],
            filename=output / "precision_var.pdf",
            title=r"Variance of Precision Var[$\Pi$]",
            colorbar_label=r"Var[$\Pi$]",
            x_label=r"$\eta_\Lambda$ (mean of log-precision)",
            y_label=r"$\Pi_\Lambda$ (precision of log-precision)",
            legend_label=lambda level: rf"Var[$\Pi$] = {level}",
            cmap="plasma",
        ),
        _save_contour(
            eta_lambda,
            pi_lambda,
            std_precision,
            levels=[0.1, 0.3, 0.5, 1.0, 2.0],
            filename=output / "precision_std.pdf",
            title="Implied Standard Deviation",
            colorbar_label=r"Implied Standard Deviation",
            x_label=r"$\eta_\Lambda$ (mean of log-precision)",
            y_label=r"$\Pi_\Lambda$ (precision of log-precision)",
            legend_label=lambda level: rf"Std ~= {level}",
            cmap="inferno",
        ),
    ]
    return saved


def _save_contour(
    x_grid,
    y_grid,
    z_grid,
    *,
    levels,
    filename,
    title,
    colorbar_label,
    x_label,
    y_label,
    legend_label,
    cmap,
):
    colors = ["cyan", "lime", "yellow", "orange", "white"]
    linestyles = ["-", "--", ":", "-.", "-"]

    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(x_grid, y_grid, z_grid, levels=100, cmap=cmap)
    legend_lines = []
    for level, color, linestyle in zip(levels, colors, linestyles):
        ax.contour(x_grid, y_grid, z_grid, levels=[level], colors=color, linestyles=linestyle, linewidths=2)
        legend_lines.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2, label=legend_label(level)))
    ax.legend(handles=legend_lines, loc="upper left", fontsize=10)
    fig.colorbar(contour, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_yscale("log")
    ax.grid(True)
    fig.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return filename


if __name__ == "__main__":
    save_log_to_precision_plots()
