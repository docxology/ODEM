from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def compute_precision_to_log_grid(
    expected_precision_range=(0.1, 10.0),
    precision_variance_range=(0.01, 100.0),
    points=200,
):
    E_vals = np.linspace(*expected_precision_range, points)
    Var_vals = np.linspace(*precision_variance_range, points)
    E_grid, Var_grid = np.meshgrid(E_vals, Var_vals)
    pi_grid = 1 / np.log(1 + Var_grid / E_grid**2)
    eta_grid = np.log(E_grid) - 0.5 * np.log(1 + Var_grid / E_grid**2)
    std_log_space = np.sqrt(1 / pi_grid)
    return E_grid, Var_grid, eta_grid, pi_grid, std_log_space


def save_precision_to_log_plots(output_dir="."):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    E_grid, Var_grid, eta_grid, pi_grid, std_log_space = compute_precision_to_log_grid()

    saved = [
        _save_contour(
            E_grid,
            Var_grid,
            eta_grid,
            levels=[-4, -2, 0, 2, 4],
            filename=output / "lambda_eta.pdf",
            title=r"Mapping from Precision Space to $\eta_\Lambda$",
            colorbar_label=r"$\eta_\Lambda$",
            x_label=r"E[$\Pi$] (Expected Precision)",
            y_label=r"Var[$\Pi$] (Precision Variance)",
            legend_label=lambda level: rf"$\eta$ = {level}",
            cmap="viridis",
        ),
        _save_contour(
            E_grid,
            Var_grid,
            pi_grid,
            levels=[0.1, 0.5, 1.0, 2.0, 5.0],
            filename=output / "lambda_precision.pdf",
            title=r"Mapping from Precision Space to $\Pi_\Lambda$",
            colorbar_label=r"$\Pi_\Lambda$",
            x_label=r"E[$\Pi$] (Expected Precision)",
            y_label=r"Var[$\Pi$] (Precision Variance)",
            legend_label=lambda level: rf"$\Pi_\Lambda$ = {level}",
            cmap="plasma",
        ),
        _save_contour(
            E_grid,
            Var_grid,
            std_log_space,
            levels=[0.5, 1.0, 2.0, 4.0, 8.0],
            filename=output / "lambda_std.pdf",
            title=r"Mapping from Precision Space to Std of Log-Normal Space",
            colorbar_label=r"Std[log($\Pi$)]",
            x_label=r"E[$\Pi$] (Expected Precision)",
            y_label=r"Var[$\Pi$] (Precision Variance)",
            legend_label=lambda level: rf"$std_\Lambda$ = {level}",
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
    valid_levels = [level for level in levels if z_grid.min() <= level <= z_grid.max()]

    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(x_grid, y_grid, z_grid, levels=100, cmap=cmap)
    legend_lines = []
    for level, color, linestyle in zip(valid_levels, colors, linestyles):
        ax.contour(x_grid, y_grid, z_grid, levels=[level], colors=color, linestyles=linestyle, linewidths=2)
        legend_lines.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=2, label=legend_label(level)))
    if legend_lines:
        ax.legend(handles=legend_lines, loc="upper left", fontsize=10)
    fig.colorbar(contour, ax=ax, label=colorbar_label)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    fig.savefig(filename, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return filename


if __name__ == "__main__":
    save_precision_to_log_plots()
