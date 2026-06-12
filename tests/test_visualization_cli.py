import ast
import subprocess
import sys

import numpy as np

from odem.visualization import create_state_animation, render_diagnostic_dashboard


def test_main_is_import_safe_entrypoint():
    module = ast.parse(open("main.py", encoding="utf-8").read())

    assert any(isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'" for node in module.body)


def test_cli_help_is_available_without_running_experiments():
    completed = subprocess.run(
        [sys.executable, "-m", "odem.cli", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )

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
