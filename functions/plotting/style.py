from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


COLORS = ("#0b5cad", "#c9252d", "#2e8540", "#d97706", "#6f4bb2", "#586069")


def new_axis(*, figsize: tuple[float, float] = (8.0, 4.2), dpi: int = 130):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    return fig, ax


def style_axis(ax, *, title: str, xlabel: str = "time step", ylabel: str | None = None, legend: bool = True) -> None:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.22)
    ax.tick_params(axis="both", labelsize=8)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8, frameon=True)


def save_pdf(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
