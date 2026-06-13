from __future__ import annotations

import numpy as np

from functions.plotting import common, style


def plot_state_outputs(data) -> list:
    outputs = []
    outputs.append(common.plot_series(data.y, data.output_dir / "y.pdf", title="Observations", labels=_labels("y", data.y)))
    outputs.append(common.plot_series(data.x_noisy, data.output_dir / "x_noisy.pdf", title="Noisy states", labels=_labels("x", data.x_noisy)))

    estimates = np.asarray(data.gen_x_estimates, dtype=float)
    kx = estimates.shape[1]
    dt = float(data.snapshot.get("gp", {}).get("dt", 1.0))
    true_gen_x = [np.asarray(data.x_noisy, dtype=float)]
    for _ in range(kx - 1):
        true_gen_x.append(np.diff(true_gen_x[-1], axis=0, prepend=0) / dt)
    names = ["x" + ("'" * idx) for idx in range(kx)]
    for coord, name in enumerate(names):
        outputs.append(common.plot_series(true_gen_x[coord], data.output_dir / f"{name}.pdf", title=f"{name} true"))
        outputs.append(common.plot_series(estimates[:, coord, :], data.output_dir / f"{name}hat.pdf", title=f"{name} estimate"))

    sensations = np.asarray(data.gen_sensations, dtype=float)
    predictions = np.asarray(data.gen_predictions, dtype=float)
    if sensations.ndim == 3:
        sensations = sensations.reshape(sensations.shape[0], -1)
        predictions = predictions.reshape(predictions.shape[0], -1)
    fig, ax = style.new_axis()
    for idx in range(sensations.shape[1]):
        ax.plot(sensations[:, idx], color=style.COLORS[idx % len(style.COLORS)], label=f"y[{idx}]")
        ax.plot(predictions[:, idx], color=style.COLORS[(idx + 1) % len(style.COLORS)], linestyle="--", label=f"yhat[{idx}]")
    style.style_axis(ax, title="Sensations and predictions")
    outputs.append(style.save_pdf(fig, data.output_dir / "sensations_predictions.pdf"))
    return outputs


def _labels(prefix: str, values) -> list[str]:
    array = np.asarray(values)
    dims = 1 if array.ndim == 1 else array.shape[-1]
    return [f"{prefix}[{idx}]" for idx in range(dims)]
