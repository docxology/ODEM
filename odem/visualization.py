from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def render_diagnostic_dashboard(result_dir: str | Path, output_path: str | Path | None = None) -> Path:
    run_dir = Path(result_dir)
    output = Path(output_path) if output_path else run_dir / "plots" / "diagnostic_dashboard.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    vfe = _load_optional(run_dir / "vfe.npy")
    accuracy = _load_optional(run_dir / "accuracy.npy")
    complexity = _load_optional(run_dir / "complexity.npy")
    x = _load_optional(run_dir / "x_noisy.npy")
    x_hat = _load_optional(run_dir / "gen_x_estimates.npy")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    ax_vfe, ax_free_action, ax_tradeoff, ax_state = axes.ravel()

    if vfe is not None:
        ax_vfe.plot(vfe, color="#8c2d04", label="VFE")
        ax_vfe.set_title("Variational Free Energy")
        ax_vfe.legend()

        ax_free_action.plot(np.cumsum(vfe), color="#08519c", label="Free action")
        ax_free_action.set_title("Cumulative Free Action")
        ax_free_action.legend()

    if accuracy is not None and complexity is not None:
        denom = accuracy + complexity + 1e-12
        ax_tradeoff.plot(accuracy / denom, label="accuracy fraction")
        ax_tradeoff.plot(complexity / denom, label="complexity fraction")
        ax_tradeoff.set_ylim(0, 1)
        ax_tradeoff.set_title("Accuracy/Complexity Balance")
        ax_tradeoff.legend()

    if x is not None and x_hat is not None:
        estimate = x_hat[:, 0, :]
        T = min(len(x), len(estimate))
        ax_state.plot(x[:T, 0], label="x[0]", color="#238b45")
        ax_state.plot(estimate[:T, 0], label="xhat[0]", color="#cb181d", linestyle="--")
        ax_state.set_title("State Tracking")
        ax_state.legend()

    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("time step")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def create_state_animation(
    result_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    fps: int = 12,
    trail: int | None = None,
) -> Path:
    run_dir = Path(result_dir)
    output = Path(output_path) if output_path else run_dir / "animations" / "state_estimation.gif"
    output.parent.mkdir(parents=True, exist_ok=True)

    x = np.load(run_dir / "x_noisy.npy", allow_pickle=False)
    x_hat = np.load(run_dir / "gen_x_estimates.npy", allow_pickle=False)[:, 0, :]
    T = min(len(x), len(x_hat))
    x = x[:T]
    x_hat = x_hat[:T]
    trail_len = T if trail is None else max(1, int(trail))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

    if x.shape[1] >= 2:
        ax.set_xlim(*_padded_limits([x[:, 0], x_hat[:, 0]]))
        ax.set_ylim(*_padded_limits([x[:, 1], x_hat[:, 1]]))
        truth_line, = ax.plot([], [], color="#238b45", label="state")
        estimate_line, = ax.plot([], [], color="#cb181d", linestyle="--", label="estimate")
        ax.set_xlabel("dimension 0")
        ax.set_ylabel("dimension 1")

        def update(frame: int):
            start = max(0, frame - trail_len + 1)
            truth_line.set_data(x[start : frame + 1, 0], x[start : frame + 1, 1])
            estimate_line.set_data(x_hat[start : frame + 1, 0], x_hat[start : frame + 1, 1])
            ax.set_title(f"State estimate t={frame}")
            return truth_line, estimate_line

    else:
        ax.set_xlim(0, T - 1)
        ax.set_ylim(*_padded_limits([x[:, 0], x_hat[:, 0]]))
        truth_line, = ax.plot([], [], color="#238b45", label="state")
        estimate_line, = ax.plot([], [], color="#cb181d", linestyle="--", label="estimate")
        ax.set_xlabel("time step")
        ax.set_ylabel("dimension 0")

        def update(frame: int):
            start = max(0, frame - trail_len + 1)
            t = np.arange(start, frame + 1)
            truth_line.set_data(t, x[start : frame + 1, 0])
            estimate_line.set_data(t, x_hat[start : frame + 1, 0])
            ax.set_title(f"State estimate t={frame}")
            return truth_line, estimate_line

    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    animation = FuncAnimation(fig, update, frames=T, interval=1000 / max(fps, 1), blit=False)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output


def dispatch_static_plots(result_dir: str | Path) -> list[Path]:
    from functions.plotting.plot import PlotDispatcher

    PlotDispatcher(data_dir=str(result_dir)).dispatch()
    return sorted((Path(result_dir) / "plots").glob("*"))


def render_sweep_summary(rows: list[dict], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    finite_rows = [row for row in rows if row.get("free_action") is not None]
    finite_rows = sorted(finite_rows, key=lambda row: row["free_action"])
    if not finite_rows:
        raise ValueError("No rows with free_action values are available for plotting.")

    ranks = np.arange(1, len(finite_rows) + 1)
    free_action = np.asarray([row["free_action"] for row in finite_rows], dtype=float)
    mse = np.asarray([np.nan if row.get("mse") is None else row["mse"] for row in finite_rows], dtype=float)
    labels = [str(row.get("run_id", row.get("path", idx))) for idx, row in enumerate(finite_rows)]

    fig, ax_fa = plt.subplots(figsize=(10, 5), dpi=140)
    ax_fa.plot(ranks, free_action, marker="o", color="#08519c", label="free action")
    ax_fa.set_xlabel("rank by free action")
    ax_fa.set_ylabel("free action", color="#08519c")
    ax_fa.tick_params(axis="y", labelcolor="#08519c")
    ax_fa.grid(True, alpha=0.25)

    ax_mse = ax_fa.twinx()
    ax_mse.plot(ranks, mse, marker="s", color="#cb181d", label="MSE")
    ax_mse.set_ylabel("MSE", color="#cb181d")
    ax_mse.tick_params(axis="y", labelcolor="#cb181d")

    max_labels = min(len(labels), 8)
    ax_fa.set_xticks(ranks[:max_labels])
    ax_fa.set_xticklabels([_short_label(label) for label in labels[:max_labels]], rotation=35, ha="right")
    ax_fa.set_title("Sweep Summary")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _load_optional(path: Path) -> np.ndarray | None:
    return np.load(path, allow_pickle=False) if path.exists() else None


def _padded_limits(series: Iterable[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(s).reshape(-1) for s in series])
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo == hi:
        pad = 1.0 if lo == 0 else abs(lo) * 0.1
    else:
        pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _short_label(label: str, max_len: int = 18) -> str:
    return label if len(label) <= max_len else label[: max_len - 1] + "..."
