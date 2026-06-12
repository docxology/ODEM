import json
from pathlib import Path

import numpy as np

from odem.analysis import summarize_run, summarize_sweep
from odem.artifacts import RunBundle


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
    bundle = RunBundle.create(tmp_path, combo_index=0, run_id="summary-run")
    bundle.save_array("vfe", np.array([2.0, 3.0]))
    bundle.save_array("accuracy", np.array([10.0, 20.0]))
    bundle.save_array("complexity", np.array([12.0, 23.0]))
    bundle.save_array("theta", np.array([[30.0], [29.5]]))
    bundle.save_array("lambda_x", np.array([[6.0], [6.1]]))
    bundle.save_array("lambda_y", np.array([[2.0], [2.1]]))
    bundle.save_array("x_noisy", np.array([[1.0, 2.0], [2.0, 4.0]]))
    bundle.save_array("gen_x_estimates", np.array([[[1.5, 1.5]], [[2.0, 5.0]]]))
    bundle.save_json("snapshot", {"fa": 5.0, "mse": 0.0, "gm": {"kx": 2}, "gp": {"dt": 0.1}})

    summary = summarize_run(bundle.path)

    assert summary["timesteps"] == 2
    assert summary["free_action"] == 5.0
    assert summary["mse"] == 0.375
    assert summary["final_theta"] == [29.5]
    assert summary["all_numeric_outputs_finite"] is True


def test_summarize_sweep_orders_by_free_action(tmp_path):
    for idx, fa in enumerate([3.0, 1.0, 2.0]):
        bundle = RunBundle.create(tmp_path, combo_index=idx, run_id=f"run-{idx}")
        bundle.save_array("vfe", np.array([fa]))
        bundle.save_json("snapshot", {"fa": fa, "mse": idx})

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
