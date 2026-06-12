import json
from pathlib import Path

import numpy as np

from odem.artifacts import RunBundle
from odem.experiment import ExperimentRunner, RunnerOptions
from odem.reporting import create_run_report, create_sweep_report
from odem.visualization import render_sweep_summary


def _write_two_combo_config(path: Path) -> Path:
    path.write_text(
        """
priors:
  theta:
    E_theta: [30.0]
    sigma_theta: [9.0]
  lambda:
    E_pi_x: [500.0]
    sigma_lambda_x: [0.1]
    E_pi_y: [10.0, 20.0]
    sigma_lambda_y: [0.1]
optimizer:
  carry_cov: true
  x:
    name: ozaki
    nu: [-4]
    kappa_x: [1.0]
  lambda:
    adapt: true
    eta:
      rate: 0.0001
      t_0: 10
      gamma: 0.3
    inter: 1
    beta: [0.0]
  theta:
    adapt: true
    eta:
      rate: 0.0001
      t_0: 10
      gamma: 0.3
    inter: [2]
    beta: [0.0]
  jitter: 1e-6
gp:
  noise:
    x_wn_mu: 0.0
    x_wn_sigma: [[0.05, 0.05, linear]]
    x_cn_kernel_size: 51
    x_cn_kernel_sigma: 0.005
  name: glv
  dt: 0.1
  T: 0.3
gm:
  dynamics: lorenz
  likelihood: identity
  kx: 2
  ky: 1
  noise:
    y_wn_mu: 0.0
    y_wn_sigma: [[0.1, 0.1, linear]]
    y_cn_kernel_size: 51
    y_cn_kernel_sigma: 0.005
""",
        encoding="utf-8",
    )
    return path


def test_experiment_runner_logs_successes_and_failures(monkeypatch, tmp_path):
    config_path = _write_two_combo_config(tmp_path / "config.yaml")
    results_dir = tmp_path / "results"
    logs_dir = tmp_path / "logs"

    def fake_run_combo(combo, sweep, *, results_dir, static_plots, dashboard, animations, tqdm_disable):
        if combo.combo_index == 1:
            raise RuntimeError("deliberate combo failure")
        bundle = RunBundle.create(results_dir, combo_index=combo.combo_index, run_id="fake-run")
        bundle.save_array("vfe", np.array([1.0, 2.0]))
        bundle.save_array("x_noisy", np.array([[0.0, 0.0], [1.0, 1.0]]))
        bundle.save_array("gen_x_estimates", np.array([[[0.0, 0.0]], [[1.5, 1.0]]]))
        bundle.save_json("snapshot", {"fa": 3.0, "mse": 0.0})
        bundle.write_manifest(status="completed")
        return bundle

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
        )
    ).run()

    assert len(result["completed"]) == 1
    assert len(result["failed"]) == 1
    assert result["failed"][0]["combo_idx"] == 1
    assert "deliberate combo failure" in result["failed"][0]["error"]
    assert (results_dir / "summary.json").exists()
    assert next(logs_dir.glob("*/failed_combos.json")).exists()


def test_reporting_creates_run_and_sweep_markdown_and_html(tmp_path):
    bundle = RunBundle.create(tmp_path / "results", combo_index=0, run_id="report-run")
    bundle.save_array("vfe", np.array([2.0, 1.0]))
    bundle.save_array("x_noisy", np.array([[1.0, 2.0], [2.0, 3.0]]))
    bundle.save_array("gen_x_estimates", np.array([[[1.0, 2.0]], [[2.5, 2.5]]]))
    bundle.save_json("snapshot", {"fa": 3.0, "mse": 0.0, "gm": {"dynamics": "lorenz"}, "gp": {"name": "glv"}})
    bundle.write_manifest(status="completed")

    run_report = create_run_report(bundle.path)
    sweep_report = create_sweep_report(tmp_path / "results")

    assert run_report.markdown_path.exists()
    assert run_report.html_path.exists()
    assert "Free action" in run_report.markdown_path.read_text(encoding="utf-8")
    assert sweep_report.markdown_path.exists()
    assert "report-run" in sweep_report.markdown_path.read_text(encoding="utf-8")


def test_render_sweep_summary_plot_uses_summary_rows(tmp_path):
    rows = [
        {"run_id": "a", "free_action": 3.0, "mse": 0.3},
        {"run_id": "b", "free_action": 1.0, "mse": 0.1},
        {"run_id": "c", "free_action": 2.0, "mse": 0.2},
    ]

    output = render_sweep_summary(rows, tmp_path / "sweep_summary.png")

    assert output.exists()
    assert output.suffix == ".png"
