from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from odem.arrays import state_estimate_issue
from odem.checksums import sha256_file
from odem.visual_checks import validate_visual_artifact


REQUIRED_RAW_ARRAY_NAMES = frozenset(
    {
        "vfe",
        "accuracy",
        "complexity",
        "gen_sensations",
        "gen_predictions",
        "x_clean",
        "x_noisy",
        "y",
        "gen_x_estimates",
        "theta",
        "lambda_x",
        "lambda_y",
        "cov_lambda_x",
        "cov_lambda_y",
        "cov_theta",
        "y_white_noise",
        "y_sigma_schedule",
        "y_colored_noise",
        "y_context_lengths",
        "x_white_noise",
        "x_sigma_schedule",
        "x_colored_noise",
        "x_context_lengths",
        "prior_theta_eta",
        "prior_theta_pi",
        "y_pi_schedule",
        "x_pi_schedule",
    }
)
REQUIRED_RUN_ARTIFACTS = frozenset({f"{name}.npy" for name in REQUIRED_RAW_ARRAY_NAMES} | {"combo.json", "snapshot.json"})
REQUIRED_STATIC_PLOT_ARTIFACTS = frozenset(
    {
        "plots/vfe.pdf",
        "plots/fa.pdf",
        "plots/y.pdf",
        "plots/x_noisy.pdf",
        "plots/lambda_x.pdf",
        "plots/lambda_y.pdf",
        "plots/theta.pdf",
        "plots/accuracy_complexity_tradeoff.pdf",
        "plots/sensations_predictions.pdf",
        "plots/x_white_noise.pdf",
        "plots/y_white_noise.pdf",
        "plots/x_colored_noise.pdf",
        "plots/y_colored_noise.pdf",
        "plots/x_sigma_schedule.pdf",
        "plots/y_sigma_schedule.pdf",
    }
)


@dataclass(frozen=True)
class BundleValidation:
    run_dir: Path
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    manifest_status: str | None = None
    missing_paths: tuple[str, ...] = ()
    checksum_mismatches: tuple[str, ...] = ()
    nonfinite_numeric_artifacts: tuple[str, ...] = ()
    shape_mismatches: tuple[str, ...] = ()
    visual_artifact_issues: tuple[str, ...] = ()


def validate_run_bundle(run_dir: str | Path, *, require_completed: bool = True) -> BundleValidation:
    path = Path(run_dir)
    errors: list[str] = []
    warnings: list[str] = []
    missing_paths: list[str] = []
    checksum_mismatches: list[str] = []
    nonfinite_numeric_artifacts: list[str] = []
    shape_mismatches: list[str] = []
    visual_artifact_issues: list[str] = []
    manifest_status: str | None = None
    arrays_by_name: dict[str, np.ndarray] = {}

    manifest_path = path / "manifest.json"
    if not path.exists() or not path.is_dir():
        errors.append(f"run bundle directory does not exist: {path}")
        return _validation(
            path,
            errors,
            warnings,
            manifest_status,
            missing_paths,
            checksum_mismatches,
            nonfinite_numeric_artifacts,
            shape_mismatches,
            visual_artifact_issues,
        )
    if not manifest_path.exists():
        errors.append("manifest.json is missing")
        return _validation(
            path,
            errors,
            warnings,
            manifest_status,
            missing_paths,
            checksum_mismatches,
            nonfinite_numeric_artifacts,
            shape_mismatches,
            visual_artifact_issues,
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"manifest.json is invalid JSON: {exc}")
        return _validation(
            path,
            errors,
            warnings,
            manifest_status,
            missing_paths,
            checksum_mismatches,
            nonfinite_numeric_artifacts,
            shape_mismatches,
            visual_artifact_issues,
        )

    manifest_status = manifest.get("status")
    if require_completed and manifest_status != "completed":
        errors.append(f"manifest status must be completed, got {manifest_status!r}")
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append("manifest metadata must be an object")
        metadata = {}

    records = manifest.get("artifacts", [])
    if not isinstance(records, list):
        errors.append("manifest artifacts must be a list")
        records = []
    records_by_path = {str(record.get("path")): record for record in records if isinstance(record, dict) and record.get("path")}

    if require_completed:
        for required_path in sorted(REQUIRED_RUN_ARTIFACTS):
            if required_path not in records_by_path:
                missing_paths.append(required_path)
                errors.append(f"required artifact missing from manifest: {required_path}")
        if _static_plots_requested(metadata):
            for required_path in sorted(REQUIRED_STATIC_PLOT_ARTIFACTS):
                if required_path not in records_by_path:
                    missing_paths.append(required_path)
                    errors.append(f"required static plot missing from manifest: {required_path}")

    for record_path, record in records_by_path.items():
        artifact_path = path / record_path
        try:
            artifact_path.resolve().relative_to(path.resolve())
        except ValueError:
            errors.append(f"manifest artifact escapes run directory: {record_path}")
            continue
        if not artifact_path.exists():
            missing_paths.append(record_path)
            errors.append(f"manifest artifact file is missing: {record_path}")
            continue

        actual_bytes = artifact_path.stat().st_size
        expected_bytes = record.get("bytes")
        if expected_bytes != actual_bytes:
            checksum_mismatches.append(record_path)
            errors.append(f"byte count mismatch for {record_path}: manifest={expected_bytes} actual={actual_bytes}")

        actual_sha = sha256_file(artifact_path)
        expected_sha = record.get("sha256")
        if expected_sha != actual_sha:
            checksum_mismatches.append(record_path)
            errors.append(f"sha256 mismatch for {record_path}: manifest={expected_sha} actual={actual_sha}")

        if artifact_path.suffix == ".npy":
            try:
                array = np.load(artifact_path, allow_pickle=False)
            except Exception as exc:
                errors.append(f"could not load numeric artifact {record_path}: {exc}")
                continue
            arrays_by_name[artifact_path.stem] = array
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                nonfinite_numeric_artifacts.append(record_path)
                warnings.append(f"nonfinite numeric values in {record_path}")
        elif artifact_path.suffix.lower() in {".png", ".gif", ".pdf"}:
            visual_errors = validate_visual_artifact(artifact_path, label=record_path)
            if visual_errors:
                visual_artifact_issues.append(record_path)
                errors.extend(visual_errors)

    _validate_array_shapes(arrays_by_name, errors, shape_mismatches)

    return _validation(
        path,
        errors,
        warnings,
        manifest_status,
        missing_paths,
        checksum_mismatches,
        nonfinite_numeric_artifacts,
        shape_mismatches,
        visual_artifact_issues,
    )


def _static_plots_requested(metadata: dict[str, object]) -> bool:
    requested = metadata.get("requested_outputs")
    return isinstance(requested, dict) and requested.get("static_plots") is True


def _validation(
    path: Path,
    errors: list[str],
    warnings: list[str],
    manifest_status: str | None,
    missing_paths: list[str],
    checksum_mismatches: list[str],
    nonfinite_numeric_artifacts: list[str],
    shape_mismatches: list[str],
    visual_artifact_issues: list[str],
) -> BundleValidation:
    return BundleValidation(
        run_dir=path,
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        manifest_status=manifest_status,
        missing_paths=tuple(sorted(set(missing_paths))),
        checksum_mismatches=tuple(sorted(set(checksum_mismatches))),
        nonfinite_numeric_artifacts=tuple(sorted(set(nonfinite_numeric_artifacts))),
        shape_mismatches=tuple(sorted(set(shape_mismatches))),
        visual_artifact_issues=tuple(sorted(set(visual_artifact_issues))),
    )


def _validate_array_shapes(arrays: dict[str, np.ndarray], errors: list[str], shape_mismatches: list[str]) -> None:
    time_steps = _time_steps(arrays, errors, shape_mismatches)

    x = arrays.get("x_noisy")
    estimates = arrays.get("gen_x_estimates")
    if x is not None and estimates is not None:
        issue = state_estimate_issue(x, estimates)
        if issue:
            _shape_error(errors, shape_mismatches, "gen_x_estimates.npy", issue)

    _validate_same_shape(arrays, "x_clean", x, errors, shape_mismatches)

    for name in ("theta", "lambda_x", "lambda_y"):
        _validate_matrix_series(arrays, name, time_steps, errors, shape_mismatches)
    for name in ("cov_lambda_x", "cov_lambda_y", "cov_theta"):
        _validate_covariance_series(arrays, name, time_steps, errors, shape_mismatches)


def _time_steps(arrays: dict[str, np.ndarray], errors: list[str], shape_mismatches: list[str]) -> int | None:
    lengths: dict[str, int] = {}
    for name in ("vfe", "accuracy", "complexity"):
        array = arrays.get(name)
        if array is None:
            continue
        if array.ndim != 1 or len(array) == 0:
            _shape_error(errors, shape_mismatches, f"{name}.npy", f"{name} must be a non-empty 1D array, got shape {array.shape}")
        else:
            lengths[name] = int(array.shape[0])

    x = arrays.get("x_noisy")
    if x is not None:
        if x.ndim != 2 or len(x) == 0:
            _shape_error(errors, shape_mismatches, "x_noisy.npy", f"x_noisy must be a non-empty 2D array, got shape {x.shape}")
        else:
            lengths["x_noisy"] = int(x.shape[0])
    y = arrays.get("y")
    if y is not None:
        if y.ndim != 2 or len(y) == 0:
            _shape_error(errors, shape_mismatches, "y.npy", f"y must be a non-empty 2D array, got shape {y.shape}")
        else:
            lengths["y"] = int(y.shape[0])

    if not lengths:
        return None
    expected = next(iter(lengths.values()))
    for name, length in lengths.items():
        if length != expected:
            _shape_error(errors, shape_mismatches, f"{name}.npy", f"{name} has {length} time steps, expected {expected}")
    return expected


def _validate_same_shape(
    arrays: dict[str, np.ndarray],
    name: str,
    reference: np.ndarray | None,
    errors: list[str],
    shape_mismatches: list[str],
) -> None:
    array = arrays.get(name)
    if array is None or reference is None:
        return
    if array.shape != reference.shape:
        _shape_error(errors, shape_mismatches, f"{name}.npy", f"{name} shape {array.shape} must match reference shape {reference.shape}")


def _validate_matrix_series(
    arrays: dict[str, np.ndarray],
    name: str,
    time_steps: int | None,
    errors: list[str],
    shape_mismatches: list[str],
) -> None:
    array = arrays.get(name)
    if array is None:
        return
    if array.ndim != 2 or len(array) == 0:
        _shape_error(errors, shape_mismatches, f"{name}.npy", f"{name} must be a non-empty 2D array, got shape {array.shape}")
    elif time_steps is not None and array.shape[0] != time_steps:
        _shape_error(errors, shape_mismatches, f"{name}.npy", f"{name} has {array.shape[0]} time steps, expected {time_steps}")


def _validate_covariance_series(
    arrays: dict[str, np.ndarray],
    name: str,
    time_steps: int | None,
    errors: list[str],
    shape_mismatches: list[str],
) -> None:
    array = arrays.get(name)
    if array is None:
        return
    if array.ndim != 3 or len(array) == 0 or array.shape[1] != array.shape[2]:
        _shape_error(errors, shape_mismatches, f"{name}.npy", f"{name} must be a non-empty square covariance series, got shape {array.shape}")
    elif time_steps is not None and array.shape[0] != time_steps:
        _shape_error(errors, shape_mismatches, f"{name}.npy", f"{name} has {array.shape[0]} time steps, expected {time_steps}")


def _shape_error(errors: list[str], shape_mismatches: list[str], artifact: str, message: str) -> None:
    shape_mismatches.append(artifact)
    errors.append(message)
