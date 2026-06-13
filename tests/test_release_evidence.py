from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tarfile

import pytest
from PIL import Image

from odem.release_evidence import archive_release_evidence, verify_release_evidence
from odem.visualization import create_state_animation, render_diagnostic_dashboard
from tests.fixtures.bundles import write_complete_bundle
from tests.fixtures.cli import REPO_ROOT


def _write_visual_bundle(root: Path, *, run_id: str = "release-run"):
    bundle = write_complete_bundle(root, run_id=run_id)
    dashboard = render_diagnostic_dashboard(bundle.path)
    animation = create_state_animation(bundle.path, fps=2)
    bundle.record_file(dashboard, kind="visualization")
    bundle.record_file(animation, kind="animation")
    bundle.write_manifest(status="completed")
    return bundle


def _archive_release(tmp_path: Path) -> Path:
    results_dir = tmp_path / "results"
    _write_visual_bundle(results_dir)
    release_dir = tmp_path / "release"
    archive_release_evidence(results_dir, release_dir, expected_combos=1)
    return release_dir


def test_release_evidence_helper_archives_and_verifies_visual_artifacts(tmp_path):
    release_dir = _archive_release(tmp_path)

    verification = verify_release_evidence(release_dir, expected_combos=1)
    manifest = verification.manifest

    assert verification.valid is True
    assert verification.errors == ()
    assert manifest["completed_runs"] == 1
    assert manifest["invalid_runs"] == 0
    assert manifest["complete"] is True
    assert manifest["archive_bytes"] > 0
    assert len(manifest["archive_sha256"]) == 64
    assert verification.archive_bytes == manifest["archive_bytes"]
    assert verification.archive_sha256 == manifest["archive_sha256"]
    assert "reports/sweep_summary.png" in verification.checked_paths
    assert "results/release-run/animations/state_estimation.gif" in verification.checked_paths


def test_release_evidence_verifier_script_outputs_json_and_fails_for_missing_tarball(tmp_path):
    release_dir = _archive_release(tmp_path)
    (release_dir / "odem_release_evidence.tar.gz").unlink()

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_release_evidence.py"),
            "--release-dir",
            str(release_dir),
            "--expected-combos",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = json.loads(completed.stdout)

    assert completed.returncode == 1
    assert payload["valid"] is False
    assert any("tarball" in error for error in payload["errors"])


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda release_dir: (release_dir / "odem_release_evidence.tar.gz").write_bytes(b"not a tarball"), "valid tarball"),
        (
            lambda release_dir: Image.new("RGB", (32, 32), "white").save(release_dir / "reports" / "sweep_summary.png"),
            "nonblank",
        ),
        (
            lambda release_dir: (release_dir / "reports" / "index.html").write_text(
                (release_dir / "reports" / "index.html").read_text(encoding="utf-8")
                + '<a href="missing-local-file.png">missing</a>',
                encoding="utf-8",
            ),
            "gallery",
        ),
    ],
)
def test_release_evidence_verifier_detects_static_artifact_defects(tmp_path, mutator, expected):
    release_dir = _archive_release(tmp_path)
    mutator(release_dir)

    verification = verify_release_evidence(release_dir, expected_combos=1)

    assert verification.valid is False
    assert any(expected in error for error in verification.errors)


def test_release_evidence_verifier_detects_invalid_best_run_gif(tmp_path):
    release_dir = _archive_release(tmp_path)
    summary = json.loads((release_dir / "reports" / "sweep_summary.json").read_text(encoding="utf-8"))
    best_gif = Path(summary[0]["path"]) / "animations" / "state_estimation.gif"
    best_gif.write_bytes(b"not a gif")

    verification = verify_release_evidence(release_dir, expected_combos=1)

    assert verification.valid is False
    assert any("GIF" in error for error in verification.errors)


def test_release_evidence_verifier_rejects_incomplete_manifest_unless_allowed(tmp_path):
    release_dir = _archive_release(tmp_path)

    strict = verify_release_evidence(release_dir, expected_combos=2)
    partial = verify_release_evidence(release_dir, expected_combos=2, allow_incomplete=True)

    assert strict.valid is False
    assert any("expected_combos=2" in error for error in strict.errors)
    assert partial.valid is True
    assert partial.warnings


def test_archive_release_evidence_leaves_no_final_manifest_or_tarball_on_archive_failure(tmp_path, monkeypatch):
    from odem import release_evidence

    results_dir = tmp_path / "results"
    _write_visual_bundle(results_dir)
    release_dir = tmp_path / "release"

    def fail_tar_open(*args, **kwargs):
        raise RuntimeError("simulated archive failure")

    monkeypatch.setattr(release_evidence.tarfile, "open", fail_tar_open)

    with pytest.raises(RuntimeError, match="simulated archive failure"):
        release_evidence.archive_release_evidence(results_dir, release_dir, expected_combos=1)

    assert not (release_dir / "release_evidence.json").exists()
    assert not (release_dir / "odem_release_evidence.tar.gz").exists()


def test_release_evidence_archive_contains_manifest_reports_and_results(tmp_path):
    release_dir = _archive_release(tmp_path)
    archive_path = release_dir / "odem_release_evidence.tar.gz"

    with tarfile.open(archive_path, "r:gz") as archive:
        names = set(archive.getnames())

    assert "release_evidence.json" in names
    assert "reports/index.html" in names
    assert "reports/sweep_summary.png" in names
    assert "results/release-run/animations/state_estimation.gif" in names
