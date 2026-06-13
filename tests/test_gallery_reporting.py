from __future__ import annotations

from pathlib import Path

from odem.release_evidence import archive_release_evidence
from odem.reporting import create_run_report, create_sweep_report
from odem.visualization import create_state_animation, render_diagnostic_dashboard
from tests.fixtures.bundles import write_complete_bundle


def test_run_report_emits_static_gallery_with_manifest_assets(tmp_path):
    bundle = write_complete_bundle(tmp_path / "results", run_id="gallery-run")
    bundle.record_file(render_diagnostic_dashboard(bundle.path), kind="visualization")
    bundle.record_file(create_state_animation(bundle.path, fps=2), kind="animation")
    bundle.write_manifest(status="completed")

    report = create_run_report(bundle.path)
    html = report.gallery_path.read_text(encoding="utf-8")

    assert report.gallery_path.name == "index.html"
    assert "diagnostic_dashboard.png" in html
    assert "state_estimation.gif" in html
    assert "Valid bundle" in html
    assert any(path.name == "diagnostic_dashboard.png" for path in report.assets)


def test_sweep_report_emits_gallery_and_links_summary_assets(tmp_path):
    write_complete_bundle(tmp_path / "results", run_id="sweep-gallery-run")

    report = create_sweep_report(tmp_path / "results", output_dir=tmp_path / "reports")
    html = report.gallery_path.read_text(encoding="utf-8")

    assert "sweep_summary.png" in html
    assert "sweep_summary.csv" in html
    assert "Valid for ranking" in html


def test_release_gallery_links_manifest_archive_and_best_run_visuals(tmp_path):
    bundle = write_complete_bundle(tmp_path / "results", run_id="visual-release-run")
    dashboard = render_diagnostic_dashboard(bundle.path)
    animation = create_state_animation(bundle.path, fps=2)
    bundle.record_file(dashboard, kind="visualization")
    bundle.record_file(animation, kind="animation")
    bundle.write_manifest(status="completed")

    manifest = archive_release_evidence(tmp_path / "results", tmp_path / "release", expected_combos=1)
    html = Path(manifest["reports"]["gallery"]).read_text(encoding="utf-8")

    assert "release_evidence.json" in html
    assert "odem_release_evidence.tar.gz" in html
    assert "diagnostic_dashboard.png" in html
    assert "state_estimation.gif" in html
