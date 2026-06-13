from __future__ import annotations

from PIL import Image, ImageChops

from odem.visualization import create_state_animation, render_diagnostic_dashboard, render_sweep_summary
from tests.fixtures.bundles import write_complete_bundle


def _assert_nonblank_image(path, *, max_height: int) -> None:
    image = Image.open(path).convert("RGB")
    extrema = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
    assert extrema is not None
    assert image.size[1] <= max_height


def test_dashboard_png_is_compact_and_nonblank(tmp_path):
    bundle = write_complete_bundle(tmp_path, run_id="visual-run")

    output = render_diagnostic_dashboard(bundle.path)

    _assert_nonblank_image(output, max_height=1250)
    image = Image.open(output)
    assert image.size[0] >= 1400
    assert image.size[1] >= 900


def test_sparse_sweep_summary_has_compact_single_run_layout(tmp_path):
    output = render_sweep_summary(
        [{"run_id": "single-run", "free_action": 5.0, "mse": 0.25, "valid_for_ranking": True}],
        tmp_path / "sweep_summary.png",
    )

    _assert_nonblank_image(output, max_height=950)
    image = Image.open(output)
    assert image.size[0] >= 1200
    assert image.size[1] >= 700


def test_sweep_summary_visualizes_validation_counts_for_invalid_rows(tmp_path):
    rows = [
        {"run_id": "invalid-shape", "free_action": None, "mse": None, "valid_bundle": False, "valid_for_ranking": False},
        {"run_id": "nonfinite", "free_action": None, "mse": 0.8, "valid_bundle": True, "valid_for_ranking": False},
        {"run_id": "rankable", "free_action": 2.0, "mse": 0.2, "valid_bundle": True, "valid_for_ranking": True},
    ]

    output = render_sweep_summary(rows, tmp_path / "sweep_summary.png")

    _assert_nonblank_image(output, max_height=950)
    image = Image.open(output)
    assert image.size[0] >= 1200
    assert image.size[1] >= 700


def test_state_animation_is_generated_from_validated_arrays(tmp_path):
    bundle = write_complete_bundle(tmp_path, run_id="animation-run")

    animation = create_state_animation(bundle.path, fps=2)

    assert animation.exists()
    assert animation.stat().st_size > 0
