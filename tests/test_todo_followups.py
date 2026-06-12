from __future__ import annotations

from pathlib import Path
import json
import re
import subprocess
import sys
import tarfile

from ruamel.yaml import YAML

from odem.validation import validate_run_bundle
from tests.fixtures.bundles import write_complete_bundle
from tests.fixtures.cli import REPO_ROOT


def _release_command(tmp_path: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_release_sweep.py"),
        "--config",
        "parameters.yaml",
        "--results-dir",
        str(tmp_path / "results"),
        "--logs-dir",
        str(tmp_path / "logs"),
        "--release-dir",
        str(tmp_path / "release"),
        "--expected-combos",
        "3024",
        "--chunk-size",
        "84",
        "--workers",
        "2",
        *extra,
    ]


def test_todo_scopes_full_sweep_without_open_completed_items():
    todo = (REPO_ROOT / "TODO.md").read_text(encoding="utf-8")

    assert "- [ ]" not in todo
    assert "Open Follow-Up" not in todo
    assert "3024" in todo
    assert "scripts/run_release_sweep.py" in todo
    assert "/tmp/odem-full" in todo
    assert "local archive" in todo.lower()


def test_release_sweep_dry_run_covers_full_parameter_space(tmp_path):
    completed = subprocess.run(
        _release_command(tmp_path, "--dry-run"),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = json.loads(completed.stdout)
    chunks = payload["chunks"]

    assert payload["total_combos"] == 3024
    assert payload["expected_combos"] == 3024
    assert payload["chunk_count"] == 36
    assert payload["workers"] == 2
    assert chunks[0] == {"index": 0, "start": 0, "end": 84}
    assert chunks[-1] == {"index": 35, "start": 2940, "end": 3024}
    assert [chunk["start"] for chunk in chunks[1:]] == [chunk["end"] for chunk in chunks[:-1]]
    assert all(chunk["end"] - chunk["start"] <= 84 for chunk in chunks)
    assert all("--animations" not in " ".join(command) for command in payload["commands"])
    assert all("--no-static-plots" not in " ".join(command) for command in payload["commands"])


def test_release_sweep_rejects_nonempty_outputs_without_force(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "existing.txt").write_text("do not overwrite", encoding="utf-8")

    completed = subprocess.run(
        _release_command(tmp_path, "--dry-run"),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
    assert "--force" in completed.stderr


def test_release_sweep_smoke_runs_archives_and_animates_best_run(tmp_path):
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_release_sweep.py"),
        "--config",
        "configs/smoke.yaml",
        "--results-dir",
        str(tmp_path / "results"),
        "--logs-dir",
        str(tmp_path / "logs"),
        "--release-dir",
        str(tmp_path / "release"),
        "--expected-combos",
        "1",
        "--chunk-size",
        "1",
        "--workers",
        "1",
        "--best-run-animation",
        "--force",
        "--no-progress",
    ]

    subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    manifest = json.loads((tmp_path / "release" / "release_evidence.json").read_text(encoding="utf-8"))
    progress_events = [
        json.loads(line)
        for line in (tmp_path / "release" / "release_progress.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    run_dirs = sorted(path for path in (tmp_path / "results").iterdir() if path.is_dir())

    assert manifest["completed_runs"] == 1
    assert manifest["invalid_runs"] == 0
    assert manifest["complete"] is True
    assert manifest["archive_bytes"] > 0
    assert len(manifest["archive_sha256"]) == 64
    assert Path(manifest["archive_path"]).exists()
    assert (run_dirs[0] / "animations" / "state_estimation.gif").exists()
    assert [event["stage"] for event in progress_events] == [
        "chunk_started",
        "chunk_finished",
        "animation_started",
        "animation_finished",
        "archive_started",
        "archive_finished",
        "verification_started",
        "verification_finished",
    ]
    assert progress_events[-1]["valid"] is True


def test_ci_workflow_runs_matrix_and_smoke_gates():
    workflow_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    workflow = YAML(typ="safe").load(workflow_path.read_text(encoding="utf-8"))
    workflow_text = workflow_path.read_text(encoding="utf-8")

    versions = {str(version) for version in workflow["jobs"]["test"]["strategy"]["matrix"]["python-version"]}

    assert versions == {"3.11", "3.12", "3.13"}
    for command in (
        "pytest -q",
        "python -m compileall -q .",
        "git diff --check",
        "python main.py validate --config parameters.yaml",
        "python main.py validate --config configs/smoke.yaml",
        "python main.py run --config configs/smoke.yaml",
        "python main.py summarize /tmp/odem-smoke/results",
        "python main.py report /tmp/odem-smoke/results",
        "python scripts/benchmark_visuals.py",
        "python scripts/archive_release_evidence.py",
        "python scripts/verify_release_evidence.py",
        "python -m pip wheel . --no-deps -w /tmp/odem-wheel",
    ):
        assert command in workflow_text


def test_visual_benchmark_script_measures_fixture_bundle(tmp_path):
    bundle = write_complete_bundle(tmp_path, run_id="benchmark-run")
    dashboard_path = bundle.path / "plots" / "diagnostic_dashboard.png"
    animation_path = bundle.path / "animations" / "state_estimation.gif"
    dashboard_path.parent.mkdir(parents=True)
    animation_path.parent.mkdir(parents=True)
    dashboard_path.write_bytes(b"previous dashboard bytes")
    animation_path.write_bytes(b"previous animation bytes")
    bundle.record_file(dashboard_path, kind="visualization")
    bundle.record_file(animation_path, kind="animation")
    bundle.write_manifest(status="completed")
    output_json = tmp_path / "benchmark.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "benchmark_visuals.py"),
            "--run-dir",
            str(bundle.path),
            "--output-json",
            str(output_json),
            "--max-dashboard-seconds",
            "60",
            "--max-animation-seconds",
            "60",
            "--fps",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    results = json.loads(output_json.read_text(encoding="utf-8"))
    assert "dashboard_seconds" in results
    assert "animation_seconds" in results
    assert results["dashboard_seconds"] >= 0
    assert results["animation_seconds"] >= 0
    assert "benchmark-run" in completed.stdout
    assert validate_run_bundle(bundle.path).valid is True


def test_visual_benchmark_threshold_failure_exits_nonzero(tmp_path):
    bundle = write_complete_bundle(tmp_path, run_id="slow-threshold")
    output_json = tmp_path / "benchmark.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "benchmark_visuals.py"),
            "--run-dir",
            str(bundle.path),
            "--output-json",
            str(output_json),
            "--max-dashboard-seconds",
            "0",
            "--max-animation-seconds",
            "60",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 1
    assert "dashboard_seconds" in completed.stderr
    assert output_json.exists()


def test_release_evidence_archiver_writes_manifest_and_archive(tmp_path):
    results_dir = tmp_path / "results"
    write_complete_bundle(results_dir, run_id="release-run")
    output_dir = tmp_path / "release"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "archive_release_evidence.py"),
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(output_dir),
            "--expected-combos",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    manifest = json.loads((output_dir / "release_evidence.json").read_text(encoding="utf-8"))
    archive_path = Path(manifest["archive_path"])

    assert manifest["completed_runs"] == 1
    assert manifest["expected_combos"] == 1
    assert manifest["complete"] is True
    assert manifest["archive_bytes"] > 0
    assert len(manifest["archive_sha256"]) == 64
    assert archive_path.exists()
    assert tarfile.is_tarfile(archive_path)


def test_release_evidence_archiver_expected_mismatch_fails_and_allow_incomplete_marks_manifest(tmp_path):
    results_dir = tmp_path / "results"
    write_complete_bundle(results_dir, run_id="release-run")
    strict_dir = tmp_path / "strict-release"
    partial_dir = tmp_path / "partial-release"

    strict = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "archive_release_evidence.py"),
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(strict_dir),
            "--expected-combos",
            "2",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    partial = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "archive_release_evidence.py"),
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(partial_dir),
            "--expected-combos",
            "2",
            "--allow-incomplete",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    manifest = json.loads((partial_dir / "release_evidence.json").read_text(encoding="utf-8"))

    assert strict.returncode == 1
    assert "completed_runs=1" in strict.stderr
    assert partial.returncode == 0
    assert manifest["complete"] is False
    assert manifest["completed_runs"] == 1


def test_release_evidence_archiver_excludes_invalid_bundles(tmp_path):
    results_dir = tmp_path / "results"
    valid = write_complete_bundle(results_dir, run_id="valid-run")
    invalid = write_complete_bundle(results_dir, run_id="invalid-run")
    (invalid.path / "vfe.npy").write_bytes(b"not a valid numpy file")
    output_dir = tmp_path / "release"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "archive_release_evidence.py"),
            "--results-dir",
            str(results_dir),
            "--output-dir",
            str(output_dir),
            "--expected-combos",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    manifest = json.loads((output_dir / "release_evidence.json").read_text(encoding="utf-8"))

    assert valid.path.exists()
    assert manifest["completed_runs"] == 1
    assert manifest["invalid_runs"] == 1


def test_docs_marked_shell_commands_execute(tmp_path):
    snippets: list[str] = []
    marker = "# docs-contract: run"
    for doc in [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]:
        text = doc.read_text(encoding="utf-8")
        for snippet in re.findall(r"```bash\n(.*?)\n```", text, flags=re.DOTALL):
            if marker in snippet:
                snippets.append(snippet.replace(marker, "").strip())

    assert snippets
    for snippet in snippets:
        command = snippet.replace("{tmp}", str(tmp_path))
        subprocess.run(command, cwd=REPO_ROOT, shell=True, text=True, check=True)
