from __future__ import annotations

from pathlib import Path

import pytest

from functions.plotting.dispatcher import PlotDispatcher
from tests.fixtures.bundles import write_complete_bundle
from tests.fixtures.cli import REPO_ROOT


def test_plot_dispatcher_uses_split_dispatcher_directly(tmp_path):
    bundle = write_complete_bundle(tmp_path, run_id="plot-run")

    dispatcher = PlotDispatcher(bundle.path)
    dispatcher.dispatch()

    assert (bundle.path / "plots" / "vfe.pdf").exists()
    assert (bundle.path / "plots" / "fa.pdf").exists()
    assert (bundle.path / "plots" / "x_noisy.pdf").exists()


def test_plot_dispatcher_rejects_malformed_run_bundle(tmp_path):
    bundle = write_complete_bundle(tmp_path, run_id="bad-plot-run")
    (bundle.path / "x_noisy.npy").unlink()

    with pytest.raises(ValueError, match="x_noisy.npy"):
        PlotDispatcher(bundle.path).dispatch()


def test_removed_entrypoint_paths_do_not_exist():
    removed_paths = (
        Path("functions") / "plotting" / "plot.py",
        Path("functions") / "report_bundle" / "report_bundle.py",
    )

    for path in removed_paths:
        assert not (REPO_ROOT / path).exists()
