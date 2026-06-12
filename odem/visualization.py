from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from odem.arrays import extract_state_estimate, load_optional_array
from odem.plot_data import finite_series, padded_limits, short_label
from odem.plot_style import THEME, apply_axes, empty_panel, save_figure
from odem.validation import validate_run_bundle


def render_diagnostic_dashboard(result_dir: str | Path, output_path: str | Path | None = None) -> Path:
    run_dir = Path(result_dir)
    output = Path(output_path) if output_path else run_dir / "plots" / "diagnostic_dashboard.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    vfe = load_optional_array(run_dir / "vfe.npy")
    accuracy = load_optional_array(run_dir / "accuracy.npy")
    complexity = load_optional_array(run_dir / "complexity.npy")
    x = load_optional_array(run_dir / "x_noisy.npy")
    x_hat = load_optional_array(run_dir / "gen_x_estimates.npy")

    fig, axes = plt.subplots(3, 3, figsize=(12.2, 8.2), dpi=130, constrained_layout=True)
    (
        ax_vfe,
        ax_free_action,
        ax_tradeoff,
        ax_state,
        ax_residual,
        ax_params,
        ax_precision,
        ax_uncertainty,
        ax_validation,
    ) = axes.ravel()

    if vfe is not None:
        vfe_series = finite_series(vfe, name="vfe")
        ax_vfe.plot(vfe_series, color=THEME.orange, label="VFE")
        apply_axes(ax_vfe, title="Variational Free Energy", xlabel="time step")
        ax_vfe.legend(fontsize=8)

        ax_free_action.plot(np.cumsum(vfe_series), color=THEME.blue, label="Free action")
        apply_axes(ax_free_action, title="Cumulative Free Action", xlabel="time step")
        ax_free_action.legend(fontsize=8)
    else:
        empty_panel(ax_vfe, "Variational Free Energy", "vfe.npy missing")
        empty_panel(ax_free_action, "Cumulative Free Action", "vfe.npy missing")

    if accuracy is not None and complexity is not None:
        accuracy = finite_series(accuracy, name="accuracy")
        complexity = finite_series(complexity, name="complexity")
        denom = np.abs(accuracy) + np.abs(complexity)
        denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
        ax_tradeoff.plot(accuracy, label="accuracy", color=THEME.blue)
        ax_tradeoff.plot(complexity, label="complexity", color=THEME.orange)
        ax_tradeoff.plot(np.abs(accuracy) / denom, label="accuracy share", color=THEME.purple, linestyle=":")
        apply_axes(ax_tradeoff, title="Accuracy and Complexity", xlabel="time step")
        ax_tradeoff.legend(fontsize=8)
    else:
        empty_panel(ax_tradeoff, "Accuracy/Complexity Balance", "accuracy/complexity arrays missing")

    if x is not None and x_hat is not None:
        x, estimate = extract_state_estimate(x, x_hat)
        for dim in range(min(x.shape[1], 3)):
            ax_state.plot(x[:, dim], label=f"x[{dim}]", color=_dim_color(dim), linewidth=1.5)
            ax_state.plot(estimate[:, dim], label=f"xhat[{dim}]", color=_dim_color(dim), linestyle="--", linewidth=1.2)
        apply_axes(ax_state, title="State Tracking", xlabel="time step")
        ax_state.legend(fontsize=7, ncols=2)

        residual = np.linalg.norm(x - estimate, axis=1)
        ax_residual.plot(residual, color=THEME.red, marker="o", markersize=3)
        ax_residual.fill_between(np.arange(len(residual)), residual, color=THEME.red, alpha=0.12)
        apply_axes(ax_residual, title=f"Residual Norm (final={residual[-1]:.3g})", xlabel="time step")
    else:
        empty_panel(ax_state, "State Tracking", "state arrays missing")
        empty_panel(ax_residual, "Residual Norm", "state arrays missing")

    theta = load_optional_array(run_dir / "theta.npy")
    if theta is not None:
        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 2 or theta.size == 0 or not np.isfinite(theta).all():
            raise ValueError(f"theta must be a non-empty finite 2D array, got shape {theta.shape}")
        for dim in range(min(theta.shape[1], 4)):
            ax_params.plot(theta[:, dim], label=f"theta[{dim}]", color=_dim_color(dim))
        apply_axes(ax_params, title="Parameter Trace", xlabel="time step")
        ax_params.legend(fontsize=7)
    else:
        empty_panel(ax_params, "Parameter Trace", "theta.npy missing")

    lambda_x = load_optional_array(run_dir / "lambda_x.npy")
    lambda_y = load_optional_array(run_dir / "lambda_y.npy")
    if lambda_x is not None or lambda_y is not None:
        _plot_matrix_series(ax_precision, lambda_x, "lambda_x", THEME.blue)
        _plot_matrix_series(ax_precision, lambda_y, "lambda_y", THEME.orange)
        apply_axes(ax_precision, title="Precision Traces", xlabel="time step")
        ax_precision.legend(fontsize=7)
    else:
        empty_panel(ax_precision, "Precision Traces", "lambda arrays missing")

    covariance_sources = (
        ("cov_theta", load_optional_array(run_dir / "cov_theta.npy"), THEME.purple),
        ("cov_lambda_x", load_optional_array(run_dir / "cov_lambda_x.npy"), THEME.blue),
        ("cov_lambda_y", load_optional_array(run_dir / "cov_lambda_y.npy"), THEME.orange),
    )
    plotted_uncertainty = False
    for label, array, color in covariance_sources:
        if array is None:
            continue
        trace = _covariance_trace(array, name=label)
        ax_uncertainty.plot(trace, label=label, color=color)
        plotted_uncertainty = True
    if plotted_uncertainty:
        apply_axes(ax_uncertainty, title="Posterior Covariance Trace", xlabel="time step")
        ax_uncertainty.legend(fontsize=7)
    else:
        empty_panel(ax_uncertainty, "Posterior Covariance Trace", "covariance arrays missing")

    _render_validation_panel(ax_validation, run_dir)

    save_figure(fig, output)
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
    x_hat_raw = np.load(run_dir / "gen_x_estimates.npy", allow_pickle=False)
    x, x_hat = extract_state_estimate(x, x_hat_raw)
    T = len(x)
    trail_len = T if trail is None else max(1, int(trail))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

    if x.shape[1] >= 2:
        ax.set_xlim(*padded_limits([x[:, 0], x_hat[:, 0]]))
        ax.set_ylim(*padded_limits([x[:, 1], x_hat[:, 1]]))
        (truth_line,) = ax.plot([], [], color=THEME.green, label="state")
        (estimate_line,) = ax.plot([], [], color=THEME.red, linestyle="--", label="estimate")
        (truth_point,) = ax.plot([], [], marker="o", color=THEME.green, markersize=6)
        (estimate_point,) = ax.plot([], [], marker="o", color=THEME.red, markersize=6)
        status_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=9, color=THEME.gray)
        ax.set_xlabel("dimension 0")
        ax.set_ylabel("dimension 1")

        def update(frame: int):
            start = max(0, frame - trail_len + 1)
            truth_line.set_data(x[start : frame + 1, 0], x[start : frame + 1, 1])
            estimate_line.set_data(x_hat[start : frame + 1, 0], x_hat[start : frame + 1, 1])
            truth_point.set_data([x[frame, 0]], [x[frame, 1]])
            estimate_point.set_data([x_hat[frame, 0]], [x_hat[frame, 1]])
            status_text.set_text(f"frame {frame + 1}/{T}\nerror={np.linalg.norm(x[frame] - x_hat[frame]):.3g}")
            ax.set_title(f"State estimate t={frame}")
            return truth_line, estimate_line, truth_point, estimate_point, status_text

    else:
        ax.set_xlim(0, T - 1)
        ax.set_ylim(*padded_limits([x[:, 0], x_hat[:, 0]]))
        (truth_line,) = ax.plot([], [], color=THEME.green, label="state")
        (estimate_line,) = ax.plot([], [], color=THEME.red, linestyle="--", label="estimate")
        (truth_point,) = ax.plot([], [], marker="o", color=THEME.green, markersize=6)
        (estimate_point,) = ax.plot([], [], marker="o", color=THEME.red, markersize=6)
        status_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=9, color=THEME.gray)
        ax.set_xlabel("time step")
        ax.set_ylabel("dimension 0")

        def update(frame: int):
            start = max(0, frame - trail_len + 1)
            t = np.arange(start, frame + 1)
            truth_line.set_data(t, x[start : frame + 1, 0])
            estimate_line.set_data(t, x_hat[start : frame + 1, 0])
            truth_point.set_data([frame], [x[frame, 0]])
            estimate_point.set_data([frame], [x_hat[frame, 0]])
            status_text.set_text(f"frame {frame + 1}/{T}\nerror={abs(x[frame, 0] - x_hat[frame, 0]):.3g}")
            ax.set_title(f"State estimate t={frame}")
            return truth_line, estimate_line, truth_point, estimate_point, status_text

    ax.grid(True, alpha=THEME.grid_alpha)
    ax.legend(loc="best")
    animation = FuncAnimation(fig, update, frames=T, interval=1000 / max(fps, 1), blit=False)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output


def dispatch_static_plots(result_dir: str | Path) -> list[Path]:
    from functions.plotting.dispatcher import PlotDispatcher

    PlotDispatcher(data_dir=str(result_dir)).dispatch()
    return sorted((Path(result_dir) / "plots").glob("*"))


def render_sweep_summary(rows: list[dict], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    finite_rows = [
        row
        for row in rows
        if row.get("valid_for_ranking", True) and row.get("free_action") is not None and np.isfinite(float(row["free_action"]))
    ]
    finite_rows = sorted(finite_rows, key=lambda row: row["free_action"])
    total_rows = len(rows)
    invalid_bundle_count = sum(1 for row in rows if row.get("valid_bundle") is False)
    invalid_ranking_count = sum(1 for row in rows if not row.get("valid_for_ranking", True))
    if not finite_rows:
        fig, ax = plt.subplots(figsize=(10.0, 5.8), dpi=130, constrained_layout=True)
        ax.text(0.5, 0.55, "No rankable finite runs", ha="center", va="center", fontsize=14)
        ax.text(
            0.5,
            0.44,
            f"total={total_rows} invalid_bundles={invalid_bundle_count} invalid_ranking={invalid_ranking_count}",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
        save_figure(fig, output)
        return output

    ranks = np.arange(1, len(finite_rows) + 1)
    free_action = np.asarray([row["free_action"] for row in finite_rows], dtype=float)
    mse = np.asarray([np.nan if row.get("mse") is None else row["mse"] for row in finite_rows], dtype=float)
    labels = [str(row.get("run_id", row.get("path", idx))) for idx, row in enumerate(finite_rows)]

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), dpi=130, constrained_layout=True)
    ax_rank, ax_scatter, ax_status, ax_best = axes.ravel()

    ax_rank.plot(ranks, free_action, marker="o", color=THEME.blue, label="free action")
    ax_rank.set_xlim(*padded_limits([ranks], frac=0.12))
    ax_rank.set_ylim(*padded_limits([free_action], frac=0.12))
    apply_axes(ax_rank, title="Free Action by Rank", xlabel="rank", ylabel="free action")

    finite_mse = np.isfinite(mse)
    if np.isfinite(mse).any():
        scatter = ax_scatter.scatter(free_action[finite_mse], mse[finite_mse], c=ranks[finite_mse], cmap="viridis", s=42)
        fig.colorbar(scatter, ax=ax_scatter, label="rank")
        ax_scatter.set_xlim(*padded_limits([free_action], frac=0.12))
        ax_scatter.set_ylim(*padded_limits([mse[finite_mse]], frac=0.12))
        apply_axes(ax_scatter, title="MSE vs Free Action", xlabel="free action", ylabel="MSE")
    else:
        empty_panel(ax_scatter, "MSE vs Free Action", "no finite MSE values")

    status_labels = ["rankable", "invalid ranking", "invalid bundle"]
    status_values = [len(finite_rows), invalid_ranking_count, invalid_bundle_count]
    ax_status.barh(status_labels, status_values, color=[THEME.green, THEME.orange, THEME.red])
    for index, value in enumerate(status_values):
        ax_status.text(value + max(status_values + [1]) * 0.03, index, str(value), va="center", fontsize=9)
    ax_status.set_xlim(0, max(status_values + [1]) * 1.25)
    apply_axes(ax_status, title="Validation Counts", xlabel="runs")

    best_count = min(len(finite_rows), 8)
    best_rows = finite_rows[:best_count]
    best_values = np.asarray([row["free_action"] for row in best_rows], dtype=float)
    best_labels = [short_label(str(row.get("run_id", row.get("path", idx))), max_len=20) for idx, row in enumerate(best_rows)]
    y_positions = np.arange(best_count)
    ax_best.barh(y_positions, best_values, color=THEME.blue)
    ax_best.set_yticks(y_positions)
    ax_best.set_yticklabels(best_labels, fontsize=8)
    ax_best.invert_yaxis()
    ax_best.set_xlim(*padded_limits([best_values], frac=0.16))
    apply_axes(ax_best, title="Top Rankable Runs", xlabel="free action")

    fig.suptitle(f"Sweep Summary: {len(finite_rows)} rankable / {total_rows} total", fontsize=13, fontweight="bold")
    save_figure(fig, output)
    return output


def _short_label(label: str, max_len: int = 18) -> str:
    return short_label(label, max_len=max_len)


def _dim_color(dim: int) -> str:
    colors = (THEME.green, THEME.blue, THEME.orange, THEME.purple)
    return colors[dim % len(colors)]


def _plot_matrix_series(ax, values: np.ndarray | None, label: str, color: str) -> None:
    if values is None:
        return
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a non-empty finite 2D array, got shape {array.shape}")
    for dim in range(min(array.shape[1], 3)):
        ax.plot(array[:, dim], label=f"{label}[{dim}]", color=color, linestyle=["-", "--", ":"][dim % 3])


def _covariance_trace(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or array.size == 0 or array.shape[1] != array.shape[2] or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite square covariance series, got shape {array.shape}")
    return np.trace(array, axis1=1, axis2=2)


def _render_validation_panel(ax, run_dir: Path) -> None:
    ax.set_axis_off()
    manifest_path = run_dir / "manifest.json"
    title = "Validation Snapshot"
    lines: list[str]
    if manifest_path.exists():
        validation = validate_run_bundle(run_dir, require_completed=False)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        status = manifest.get("status", "unknown")
        issue_count = len(validation.errors) + len(validation.warnings)
        lines = [
            f"manifest: {status}",
            f"validation: {'pass' if validation.valid else 'fail'}",
            f"issues: {issue_count}",
            f"artifacts: {len(manifest.get('artifacts', []))}",
        ]
        if validation.errors:
            lines.append(short_label(validation.errors[0], max_len=48))
        elif validation.warnings:
            lines.append(short_label(validation.warnings[0], max_len=48))
    else:
        lines = ["manifest: missing", "validation: not run", "artifacts: 0"]
    ax.set_title(title, fontsize=12, pad=8)
    for index, line in enumerate(lines):
        ax.text(0.04, 0.86 - index * 0.15, line, ha="left", va="top", fontsize=9, color=THEME.gray, transform=ax.transAxes)
