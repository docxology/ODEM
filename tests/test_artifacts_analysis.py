import json
from pathlib import Path

import numpy as np
import pytest

from odem.analysis import summarize_run, summarize_sweep
from odem.artifacts import RunBundle, validate_run_bundle
from odem.validation import BundleValidation as ModularBundleValidation
from odem.validation import validate_run_bundle as modular_validate_run_bundle
from tests.fixtures.bundles import write_complete_bundle


def test_run_bundle_writes_manifest_checksums_and_raw_arrays(tmp_path):
    bundle = RunBundle.create(tmp_path, combo_index=3, run_id="fixed-run")

    array_path = bundle.save_array("vfe", np.array([1.0, 2.0, 3.0]))
    snapshot_path = bundle.save_json("snapshot", {"fa": 6.0, "mse": 0.25})
    manifest_path = bundle.write_manifest(status="completed")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_names = {artifact["name"] for artifact in manifest["artifacts"]}

    assert array_path.exists()
    assert snapshot_path.exists()
    assert manifest["combo_index"] == 3
    assert manifest["status"] == "completed"
    assert {"vfe", "snapshot"} <= artifact_names
    assert all(artifact["sha256"] for artifact in manifest["artifacts"])


def test_summarize_run_uses_raw_arrays_and_snapshot(tmp_path):
    bundle = write_complete_bundle(tmp_path, run_id="summary-run")

    summary = summarize_run(bundle.path)

    assert summary["timesteps"] == 2
    assert summary["free_action"] == 5.0
    assert summary["mse"] == 0.375
    assert summary["final_theta"] == [29.5]
    assert summary["all_numeric_outputs_finite"] is True
    assert summary["valid_for_ranking"] is True
    assert summary["issues"] == []


def test_summarize_sweep_orders_by_free_action(tmp_path):
    for idx, fa in enumerate([3.0, 1.0, 2.0]):
        write_complete_bundle(tmp_path, combo_index=idx, run_id=f"run-{idx}", vfe=np.array([fa, 0.0]))

    rows = summarize_sweep(tmp_path)

    assert [row["free_action"] for row in rows] == [1.0, 2.0, 3.0]
    assert [row["combo_index"] for row in rows] == [1, 2, 0]


def test_run_bundle_can_reopen_and_record_derived_artifacts(tmp_path):
    bundle = RunBundle.create(tmp_path, combo_index=4, run_id="reopen-run")
    bundle.write_manifest(status="completed")
    plot_path = bundle.path / "plots" / "diagnostic_dashboard.png"
    plot_path.parent.mkdir()
    plot_path.write_bytes(b"not a real png, just checksum bytes")

    reopened = RunBundle.open(bundle.path)
    reopened.record_file(plot_path, kind="visualization")
    manifest_path = reopened.write_manifest(status="completed")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["combo_index"] == 4
    assert any(artifact["path"] == "plots/diagnostic_dashboard.png" for artifact in manifest["artifacts"])


def test_validate_run_bundle_rejects_created_or_incomplete_bundle(tmp_path):
    bundle = RunBundle.create(tmp_path, combo_index=0, run_id="created-run")
    bundle.save_array("vfe", np.array([1.0]))
    bundle.write_manifest(status="created")

    validation = validate_run_bundle(bundle.path)

    assert validation.valid is False
    assert any("status" in error for error in validation.errors)
    assert any("snapshot.json" in error for error in validation.errors)
    with pytest.raises(ValueError, match="invalid run bundle"):
        summarize_run(bundle.path)


def test_bundle_validation_is_available_from_dedicated_module(tmp_path):
    bundle = write_complete_bundle(tmp_path, run_id="modular-validation")

    validation = modular_validate_run_bundle(bundle.path)

    assert isinstance(validation, ModularBundleValidation)
    assert validation.valid is True


def test_validate_run_bundle_requires_static_pdfs_when_requested(tmp_path):
    required_static_plots = {
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
    bundle = write_complete_bundle(tmp_path, run_id="static-required")
    bundle.metadata["requested_outputs"] = {"static_plots": True}
    bundle.write_manifest(status="completed")

    validation = validate_run_bundle(bundle.path)

    assert validation.valid is False
    assert "plots/vfe.pdf" in validation.missing_paths

    for rel_path in required_static_plots:
        path = bundle.path / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"%PDF-1.4\n% odem test artifact\n")
        bundle.record_file(path, kind="visualization")
    bundle.write_manifest(status="completed")

    validation = validate_run_bundle(bundle.path)

    assert validation.valid is True

    tampered = bundle.path / "plots" / "vfe.pdf"
    tampered.write_bytes(b"%PDF-1.4\n% tampered\n")
    validation = validate_run_bundle(bundle.path)

    assert validation.valid is False
    assert "plots/vfe.pdf" in validation.checksum_mismatches


def test_validate_run_bundle_detects_checksum_tampering(tmp_path):
    bundle = write_complete_bundle(tmp_path, run_id="tampered-run")
    np.save(bundle.path / "vfe.npy", np.array([99.0, 100.0]))

    validation = validate_run_bundle(bundle.path)

    assert validation.valid is False
    assert any("sha256" in error for error in validation.errors)
    with pytest.raises(ValueError, match="sha256"):
        summarize_run(bundle.path)


def test_summarize_run_flags_mismatched_state_estimate_shapes(tmp_path):
    bundle = write_complete_bundle(
        tmp_path,
        run_id="shape-mismatch",
        x_noisy=np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]),
        gen_x_estimates=np.array([[[0.0, 0.0]]]),
    )

    with pytest.raises(ValueError, match="x_noisy shape"):
        summarize_run(bundle.path)


def test_validate_run_bundle_rejects_state_shape_contract_violations(tmp_path):
    bundle = write_complete_bundle(tmp_path, run_id="invalid-state-shape")
    bundle.save_array("gen_x_estimates", np.array([[[0.0, 0.0]]]))
    bundle.write_manifest(status="completed")

    validation = validate_run_bundle(bundle.path)

    assert validation.valid is False
    assert "gen_x_estimates.npy" in validation.shape_mismatches
    assert any("x_noisy shape" in error for error in validation.errors)


def test_validate_run_bundle_rejects_blank_or_corrupt_recorded_visuals(tmp_path):
    from PIL import Image

    bundle = write_complete_bundle(tmp_path, run_id="bad-visuals")
    blank_png = bundle.path / "plots" / "diagnostic_dashboard.png"
    corrupt_gif = bundle.path / "animations" / "state_estimation.gif"
    blank_png.parent.mkdir(parents=True, exist_ok=True)
    corrupt_gif.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "white").save(blank_png)
    corrupt_gif.write_bytes(b"not a gif")
    bundle.record_file(blank_png, kind="visualization")
    bundle.record_file(corrupt_gif, kind="animation")
    bundle.write_manifest(status="completed")

    validation = validate_run_bundle(bundle.path)

    assert validation.valid is False
    assert "plots/diagnostic_dashboard.png" in validation.visual_artifact_issues
    assert "animations/state_estimation.gif" in validation.visual_artifact_issues
    assert any("nonblank" in error for error in validation.errors)
    assert any("GIF" in error for error in validation.errors)


def test_summarize_sweep_places_nonfinite_free_action_after_valid_runs(tmp_path):
    write_complete_bundle(tmp_path, combo_index=0, run_id="valid-run", vfe=np.array([1.0, 1.0]))
    write_complete_bundle(tmp_path, combo_index=1, run_id="inf-run", vfe=np.array([np.inf, 1.0]))
    write_complete_bundle(tmp_path, combo_index=2, run_id="better-run", vfe=np.array([0.5, 0.25]))

    rows = summarize_sweep(tmp_path)

    assert [row["run_id"] for row in rows] == ["better-run", "valid-run", "inf-run"]
    assert rows[-1]["valid_for_ranking"] is False
    assert rows[-1]["free_action"] is None
