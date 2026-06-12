from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

import numpy as np


def summarize_run(result_dir: str | Path) -> dict[str, Any]:
    path = Path(result_dir)
    arrays = _load_arrays(path)
    snapshot = _load_json(path / "snapshot.json")
    manifest = _load_json(path / "manifest.json")

    vfe = arrays.get("vfe")
    summary: dict[str, Any] = {
        "path": str(path),
        "run_id": manifest.get("run_id", path.name),
        "combo_index": manifest.get("combo_index"),
        "timesteps": int(len(vfe)) if vfe is not None else None,
        "free_action": float(np.sum(vfe)) if vfe is not None else _optional_float(snapshot.get("fa")),
        "mse": _compute_mse(arrays, snapshot),
        "accuracy_total": _sum_if_present(arrays.get("accuracy")),
        "complexity_total": _sum_if_present(arrays.get("complexity")),
        "final_theta": _last_row(arrays.get("theta")),
        "final_lambda_x": _last_row(arrays.get("lambda_x")),
        "final_lambda_y": _last_row(arrays.get("lambda_y")),
        "all_numeric_outputs_finite": _all_finite(arrays),
    }

    if "gm" in snapshot:
        summary["gm"] = snapshot["gm"]
    if "gp" in snapshot:
        summary["gp"] = snapshot["gp"]
    return summary


def summarize_sweep(parent_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(parent_dir)
    candidates = [p for p in root.iterdir() if p.is_dir() and ((p / "snapshot.json").exists() or (p / "vfe.npy").exists())]
    rows = [summarize_run(path) for path in candidates]
    return sorted(rows, key=lambda row: (float("inf") if row["free_action"] is None else row["free_action"], row["path"]))


def write_summary_json(rows: list[dict[str, Any]], path: str | Path) -> Path:
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return filepath


def write_summary_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "combo_index",
        "timesteps",
        "free_action",
        "mse",
        "accuracy_total",
        "complexity_total",
        "all_numeric_outputs_finite",
        "path",
    ]
    with filepath.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return filepath


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    arrays = {}
    for npy_path in path.glob("*.npy"):
        arrays[npy_path.stem] = np.load(npy_path, allow_pickle=False)
    return arrays


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_if_present(array: np.ndarray | None) -> float | None:
    return None if array is None else float(np.sum(array))


def _last_row(array: np.ndarray | None) -> list[float] | None:
    if array is None or len(array) == 0:
        return None
    last = np.asarray(array[-1]).reshape(-1)
    return [float(v) for v in last]


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _compute_mse(arrays: dict[str, np.ndarray], snapshot: dict[str, Any]) -> float | None:
    if "x_noisy" in arrays and "gen_x_estimates" in arrays:
        x = arrays["x_noisy"]
        x_hat = arrays["gen_x_estimates"][:, 0, :]
        T = min(len(x), len(x_hat))
        return float(np.mean((x[:T] - x_hat[:T]) ** 2))
    return _optional_float(snapshot.get("mse"))


def _all_finite(arrays: dict[str, np.ndarray]) -> bool:
    return all(np.isfinite(array).all() for array in arrays.values() if np.issubdtype(array.dtype, np.number))
