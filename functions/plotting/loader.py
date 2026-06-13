from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


REQUIRED_PLOT_ARRAYS = (
    "vfe",
    "accuracy",
    "complexity",
    "x_noisy",
    "y",
    "gen_x_estimates",
    "gen_sensations",
    "gen_predictions",
    "theta",
    "lambda_x",
    "lambda_y",
    "cov_lambda_x",
    "cov_lambda_y",
    "cov_theta",
    "x_white_noise",
    "x_colored_noise",
    "x_sigma_schedule",
    "y_white_noise",
    "y_colored_noise",
    "y_sigma_schedule",
)


@dataclass(frozen=True)
class PlotRunData:
    run_dir: Path
    output_dir: Path
    snapshot: dict
    arrays: dict[str, np.ndarray]

    @classmethod
    def load(cls, run_dir: str | Path) -> "PlotRunData":
        path = Path(run_dir)
        missing = [f"{name}.npy" for name in REQUIRED_PLOT_ARRAYS if not (path / f"{name}.npy").exists()]
        if missing:
            raise ValueError(f"missing required plot arrays: {', '.join(missing)}")
        snapshot_path = path / "snapshot.json"
        if not snapshot_path.exists():
            raise ValueError("snapshot.json is required for static plots")
        arrays = {name: np.load(path / f"{name}.npy", allow_pickle=False) for name in REQUIRED_PLOT_ARRAYS}
        return cls(
            run_dir=path,
            output_dir=path / "plots",
            snapshot=json.loads(snapshot_path.read_text(encoding="utf-8")),
            arrays=arrays,
        )

    def __getattr__(self, name: str):
        if name in self.arrays:
            return self.arrays[name]
        raise AttributeError(name)
