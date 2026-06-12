import csv
from pathlib import Path

import numpy as np
import pytest

from odem.experiment import ExperimentRunner, RunnerOptions, resolve_combo_slice
from odem.reporting import create_run_report, create_sweep_report
from odem.visualization import render_sweep_summary
from tests.fixtures.bundles import write_complete_bundle
from tests.fixtures.configs import write_two_combo_config


def test_experiment_runner_logs_successes_and_failures(monkeypatch, tmp_path):
    config_path = write_two_combo_config(tmp_path / "config.yaml")
    results_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"

    def fake_run_combo(combo, sweep, *, results_dir, static_plots, dashboard, animations, tqdm_disable):
        if combo.combo_index == 1:
            raise RuntimeError("deliberate combo failure")
        return write_complete_bundle(results_dir, combo_index=combo.combo_index, run_id="fake-run")

    monkeypatch.setattr("odem.experiment.run_combo", fake_run_combo)

    result = ExperimentRunner(
        RunnerOptions(
            config_path=config_path,
            results_dir=results_dir,
            logs_dir=logs_dir,
            static_plots=False,
            dashboard=False,
            animations=False,
            tqdm_disable=True,
            allow_partial=True,
        )
    ).run()

    assert len(result["completed"]) == 1
    assert len(result["failed"]) == 1
    assert result["failed"][0]["combo_idx"] == 1
    assert "deliberate combo failure" in result["failed"][0]["error"]
    assert (results_dir / "summary.json").exists()
    assert next(logs_dir.glob("*/failed_combos.json")).exists()


def test_experiment_runner_strict_mode_raises_after_logging_failures(monkeypatch, tmp_path):
    config_path = write_two_combo_config(tmp_path / "config.yaml")
    results_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"

    def fake_run_combo(combo, sweep, *, results_dir, static_plots, dashboard, animations, tqdm_disable):
        raise RuntimeError(f"combo {combo.combo_index} failed")

    monkeypatch.setattr("odem.experiment.run_combo", fake_run_combo)

    with pytest.raises(RuntimeError, match="failed"):
        ExperimentRunner(
            RunnerOptions(
                config_path=config_path,
                results_dir=results_dir,
                logs_dir=logs_dir,
                static_plots=False,
                dashboard=False,
                animations=False,
                tqdm_disable=True,
            )
        ).run()

    assert next(logs_dir.glob("*/failed_combos.json")).exists()


def test_reporting_creates_run_and_sweep_markdown_and_html(tmp_path):
    bundle = write_complete_bundle(tmp_path / "results", run_id="report-run", vfe=np.array([2.0, 1.0]))

    run_report = create_run_report(bundle.path)
    sweep_report = create_sweep_report(tmp_path / "results")

    assert run_report.markdown_path.exists()
    assert run_report.html_path.exists()
    assert "Free action" in run_report.markdown_path.read_text(encoding="utf-8")
    assert sweep_report.markdown_path.exists()
    assert "report-run" in sweep_report.markdown_path.read_text(encoding="utf-8")
    assert sweep_report.figure_path is not None
    assert sweep_report.figure_path.exists()
    csv_rows = list(csv.DictReader((sweep_report.markdown_path.parent / "sweep_summary.csv").open(encoding="utf-8")))
    assert csv_rows[0]["valid_bundle"] == "True"
    assert csv_rows[0]["valid_for_ranking"] == "True"
    assert csv_rows[0]["issues"] == "[]"


def test_render_sweep_summary_plot_uses_summary_rows(tmp_path):
    rows = [
        {"run_id": "a", "free_action": 3.0, "mse": 0.3},
        {"run_id": "b", "free_action": 1.0, "mse": 0.1},
        {"run_id": "c", "free_action": 2.0, "mse": 0.2},
    ]

    output = render_sweep_summary(rows, tmp_path / "sweep_summary.png")

    assert output.exists()
    assert output.suffix == ".png"


def test_render_sweep_summary_plot_writes_empty_state_for_no_valid_rows(tmp_path):
    output = render_sweep_summary([], tmp_path / "sweep_summary.png")

    assert output.exists()
    assert output.suffix == ".png"


def test_resolve_combo_slice_rejects_negative_max_combos():
    with pytest.raises(ValueError, match="max_combos"):
        resolve_combo_slice(10, max_combos=-1)


def test_resolve_combo_slice_rejects_empty_explicit_slice():
    with pytest.raises(ValueError, match="empty"):
        resolve_combo_slice(10, start_index=3, end_index=3)


def test_resolve_combo_slice_normalizes_one_based_slurm_array(monkeypatch):
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "3")

    assert resolve_combo_slice(10) == (0, 4)
