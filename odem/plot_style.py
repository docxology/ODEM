from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotTheme:
    blue: str = "#0b5cad"
    red: str = "#c9252d"
    green: str = "#2e8540"
    orange: str = "#d97706"
    purple: str = "#6f4bb2"
    gray: str = "#586069"
    grid_alpha: float = 0.22


THEME = PlotTheme()


def apply_axes(ax, *, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=12, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=THEME.grid_alpha)
    ax.tick_params(axis="both", labelsize=8)


def empty_panel(ax, title: str, message: str) -> None:
    ax.set_title(title, fontsize=12, pad=8)
    ax.text(0.5, 0.55, message, ha="center", va="center", fontsize=9, color=THEME.gray, transform=ax.transAxes)
    ax.set_axis_off()


def save_figure(fig, output):
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output
