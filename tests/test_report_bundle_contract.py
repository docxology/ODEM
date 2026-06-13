from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from odem.artifacts import RunBundle
from functions.report_bundle.data import compute_best_for_kx
from tests.fixtures.bundles import write_complete_bundle


def _write_report_combo(run_dir: Path, *, e_pi_y: float, kappa_x: float) -> None:
    (run_dir / "combo.json").write_text(
        json.dumps(
            {
                "kx": 2,
                "E_pi_y": [e_pi_y],
                "kappa_x": kappa_x,
                "theta_interval": 2,
                "theta_beta": 0.0,
                "lambda_beta": 0.0,
            }
        ),
        encoding="utf-8",
    )
    bundle = RunBundle.open(run_dir)
    bundle.record_file(run_dir / "combo.json", name="combo", kind="metadata")
    bundle.write_manifest(status="completed")


def test_report_bundle_best_selection_uses_validated_summaries(tmp_path):
    valid = write_complete_bundle(tmp_path / "kx=2", run_id="valid", combo_index=0, vfe=np.array([3.0, 1.0]))
    invalid = write_complete_bundle(tmp_path / "kx=2", run_id="invalid", combo_index=1, vfe=np.array([np.inf, 1.0]))
    _write_report_combo(valid.path, e_pi_y=10.0, kappa_x=1.5)
    _write_report_combo(invalid.path, e_pi_y=10.0, kappa_x=0.5)

    best = compute_best_for_kx(tmp_path, 2, [10.0])

    assert best[10.0]["folder"] == valid.path
    assert best[10.0]["min_fa"] == 4.0
    assert best[10.0]["mse"] == 0.375
    assert best[10.0]["kappa_x_all"] == [1.5]
