from __future__ import annotations

import json

import numpy as np

from odem.artifacts import REQUIRED_RUN_ARTIFACTS, validate_run_bundle
from tests.fixtures.cli import run_cli


def test_smoke_run_produces_required_manifest_arrays_and_finite_summary(tmp_path):
    results_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"
    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    reports_dir = tmp_path / "reports"

    run = run_cli(
        "main.py",
        "run",
        "--config",
        "configs/smoke.yaml",
        "--results-dir",
        str(results_dir),
        "--logs-dir",
        str(logs_dir),
        "--max-combos",
        "1",
        "--no-static-plots",
        "--dashboard",
        "--animations",
        "--no-progress",
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "Free Action=" not in run.stdout
    assert "VFE:" not in run.stdout

    run_dirs = [path for path in results_dir.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    validation = validate_run_bundle(run_dir)
    assert validation.valid is True
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    artifact_paths = {artifact["path"] for artifact in manifest["artifacts"]}
    assert REQUIRED_RUN_ARTIFACTS <= artifact_paths
    assert "plots/diagnostic_dashboard.png" in artifact_paths
    assert "animations/state_estimation.gif" in artifact_paths
    assert np.isfinite(np.load(run_dir / "vfe.npy", allow_pickle=False)).all()

    summarize = run_cli(
        "main.py",
        "summarize",
        str(results_dir),
        "--output-json",
        str(summary_json),
        "--output-csv",
        str(summary_csv),
        check=False,
    )
    assert summarize.returncode == 0, summarize.stderr
    rows = json.loads(summary_json.read_text(encoding="utf-8"))
    assert rows[0]["all_numeric_outputs_finite"] is True
    assert rows[0]["valid_for_ranking"] is True
    assert summary_csv.exists()

    report = run_cli(
        "main.py",
        "report",
        str(results_dir),
        "--output-dir",
        str(reports_dir),
        check=False,
    )
    assert report.returncode == 0, report.stderr
    assert (reports_dir / "sweep_report.md").exists()
    assert (reports_dir / "sweep_summary.png").exists()
