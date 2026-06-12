import ast

import numpy as np
import pytest

from odem.visualization import create_state_animation, render_diagnostic_dashboard, render_sweep_summary
from tests.fixtures.cli import run_cli


def test_main_is_import_safe_entrypoint():
    module = ast.parse(open("main.py", encoding="utf-8").read())

    assert any(isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'" for node in module.body)


def test_cli_help_is_available_without_running_experiments():
    completed = run_cli("-m", "odem.cli", "--help")

    assert completed.returncode == 0
    assert "run" in completed.stdout
    assert "summarize" in completed.stdout


def test_visualization_helpers_create_dashboard_and_animation(tmp_path):
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    np.save(result_dir / "vfe.npy", np.array([3.0, 2.0, 1.0]))
    np.save(result_dir / "accuracy.npy", np.array([5.0, 4.0, 3.0]))
    np.save(result_dir / "complexity.npy", np.array([8.0, 6.0, 4.0]))
    np.save(result_dir / "x_noisy.npy", np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.5]]))
    np.save(
        result_dir / "gen_x_estimates.npy",
        np.array([[[0.1, 0.0]], [[0.8, 1.2]], [[2.1, 1.4]]]),
    )

    dashboard = render_diagnostic_dashboard(result_dir)
    animation = create_state_animation(result_dir, fps=2)

    assert dashboard.exists()
    assert dashboard.suffix == ".png"
    assert animation.exists()
    assert animation.suffix == ".gif"


def test_dashboard_and_animation_reject_empty_or_bad_rank_state_arrays(tmp_path):
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    np.save(result_dir / "x_noisy.npy", np.empty((0, 2)))
    np.save(result_dir / "gen_x_estimates.npy", np.empty((0, 2)))

    with pytest.raises(ValueError, match="gen_x_estimates"):
        render_diagnostic_dashboard(result_dir)
    with pytest.raises(ValueError, match="gen_x_estimates"):
        create_state_animation(result_dir)


def test_render_sweep_summary_filters_nonfinite_rows(tmp_path):
    rows = [
        {"run_id": "bad", "free_action": np.inf, "mse": 1.0, "valid_for_ranking": False},
        {"run_id": "good", "free_action": 1.0, "mse": 0.1, "valid_for_ranking": True},
    ]

    output = render_sweep_summary(rows, tmp_path / "sweep_summary.png")

    assert output.exists()
