from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

import numpy as np

from odem.arrays import all_numeric_finite, load_arrays, state_estimate_issue
from odem.summary_schema import SUMMARY_CSV_FIELDNAMES
from odem.validation import validate_run_bundle


def summarize_run(result_dir: str | Path) -> dict[str, Any]:
    path = Path(result_dir)
    validation = validate_run_bundle(path)
    if not validation.valid:
        details = "; ".join(validation.errors)
        raise ValueError(f"invalid run bundle: {details}")

    arrays = load_arrays(path)
    snapshot = _load_json(path / "snapshot.json")
    manifest = _load_json(path / "manifest.json")

    vfe = arrays.get("vfe")
    issues = list(validation.warnings)
    mse, mse_issues = _compute_mse(arrays, snapshot)
    issues.extend(mse_issues)
    free_action = _finite_optional_float(float(np.sum(vfe)) if vfe is not None else snapshot.get("fa"))
    if free_action is None:
        issues.append("free_action is missing or nonfinite")
    all_numeric_outputs_finite = all_numeric_finite(arrays)
    summary: dict[str, Any] = {
        "path": str(path),
        "run_id": manifest.get("run_id", path.name),
        "combo_index": manifest.get("combo_index"),
        "timesteps": int(len(vfe)) if vfe is not None else None,
        "free_action": free_action,
        "mse": mse,
        "accuracy_total": _sum_if_present(arrays.get("accuracy")),
        "complexity_total": _sum_if_present(arrays.get("complexity")),
        "final_theta": _last_row(arrays.get("theta")),
        "final_lambda_x": _last_row(arrays.get("lambda_x")),
        "final_lambda_y": _last_row(arrays.get("lambda_y")),
        "all_numeric_outputs_finite": all_numeric_outputs_finite,
        "valid_bundle": validation.valid,
        "valid_for_ranking": bool(free_action is not None and all_numeric_outputs_finite and not mse_issues),
        "manifest_status": validation.manifest_status,
        "issues": issues,
    }

    if "gm" in snapshot:
        summary["gm"] = snapshot["gm"]
    if "gp" in snapshot:
        summary["gp"] = snapshot["gp"]
    return summary


def summarize_sweep(parent_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(parent_dir)
    candidates = [p for p in root.iterdir() if p.is_dir() and ((p / "snapshot.json").exists() or (p / "vfe.npy").exists())]
    rows = []
    for path in candidates:
        try:
            rows.append(summarize_run(path))
        except ValueError as exc:
            rows.append(
                {
                    "path": str(path),
                    "run_id": path.name,
                    "combo_index": None,
                    "timesteps": None,
                    "free_action": None,
                    "mse": None,
                    "accuracy_total": None,
                    "complexity_total": None,
                    "all_numeric_outputs_finite": False,
                    "valid_bundle": False,
                    "valid_for_ranking": False,
                    "manifest_status": None,
                    "issues": [str(exc)],
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            not row.get("valid_for_ranking", False),
            float("inf") if row["free_action"] is None else row["free_action"],
            row["path"],
        ),
    )


def write_summary_json(rows: list[dict[str, Any]], path: str | Path) -> Path:
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return filepath


def write_summary_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SUMMARY_CSV_FIELDNAMES})
    return filepath


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


def _finite_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _compute_mse(arrays: dict[str, np.ndarray], snapshot: dict[str, Any]) -> tuple[float | None, list[str]]:
    if "x_noisy" in arrays and "gen_x_estimates" in arrays:
        x = arrays["x_noisy"]
        estimates = arrays["gen_x_estimates"]
        issue = state_estimate_issue(x, estimates)
        if issue:
            return None, [issue]
        x_hat = estimates[:, 0, :]
        mse = float(np.mean((x - x_hat) ** 2))
        return (mse if np.isfinite(mse) else None), ([] if np.isfinite(mse) else ["mse is nonfinite"])
    return _finite_optional_float(snapshot.get("mse")), []
