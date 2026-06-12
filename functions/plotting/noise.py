from __future__ import annotations

from functions.plotting import common


def plot_noise_outputs(data) -> list:
    return [
        common.plot_series(data.x_white_noise, data.output_dir / "x_white_noise.pdf", title="State white noise"),
        common.plot_series(data.y_white_noise, data.output_dir / "y_white_noise.pdf", title="Observation white noise"),
        common.plot_series(data.x_colored_noise, data.output_dir / "x_colored_noise.pdf", title="State colored noise"),
        common.plot_series(data.y_colored_noise, data.output_dir / "y_colored_noise.pdf", title="Observation colored noise"),
        common.plot_series(data.x_sigma_schedule, data.output_dir / "x_sigma_schedule.pdf", title="State sigma schedule"),
        common.plot_series(data.y_sigma_schedule, data.output_dir / "y_sigma_schedule.pdf", title="Observation sigma schedule"),
    ]
