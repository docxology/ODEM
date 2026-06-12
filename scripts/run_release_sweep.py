from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from odem.analysis import summarize_sweep
from odem.config import load_sweep
from odem.release_evidence import verify_release_evidence


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    try:
        return _main(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _main(args: Any) -> int:
    config = Path(args.config)
    results_dir = Path(args.results_dir)
    logs_dir = Path(args.logs_dir)
    release_dir = Path(args.release_dir)
    total = load_sweep(config).combination_count
    expected_combos = int(args.expected_combos)

    if expected_combos != total:
        raise ValueError(f"expected-combos must match config combination count: expected={expected_combos} actual={total}")
    if int(args.chunk_size) < 1:
        raise ValueError("chunk-size must be at least 1")
    if int(args.workers) < 1:
        raise ValueError("workers must be at least 1")

    chunks = _chunks(total, int(args.chunk_size))
    commands = [
        _chunk_command(
            config=config,
            results_dir=results_dir,
            logs_dir=logs_dir,
            chunk=chunk,
            seed=int(args.seed) + int(chunk["start"]),
            no_progress=bool(args.no_progress),
        )
        for chunk in chunks
    ]
    payload = {
        "dry_run": bool(args.dry_run),
        "config": str(config),
        "total_combos": total,
        "expected_combos": expected_combos,
        "chunk_size": int(args.chunk_size),
        "chunk_count": len(chunks),
        "workers": int(args.workers),
        "results_dir": str(results_dir),
        "logs_dir": str(logs_dir),
        "release_dir": str(release_dir),
        "chunks": chunks,
        "commands": commands,
    }

    _prepare_outputs((results_dir, logs_dir, release_dir), force=bool(args.force))
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "release_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    progress_path = release_dir / "release_progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    failures = _run_chunks(commands, release_dir=release_dir, workers=int(args.workers), progress_path=progress_path)
    if failures:
        print(json.dumps({"failed_chunks": failures}, indent=2), file=sys.stderr)
        return 1

    root_summary = results_dir / "summary.json"
    if root_summary.exists():
        root_summary.unlink()
    rows = summarize_sweep(results_dir)
    valid_rows = [row for row in rows if row.get("valid_bundle") and row.get("manifest_status") == "completed"]
    if args.best_run_animation:
        if not valid_rows:
            print("cannot render best-run animation: no valid completed runs", file=sys.stderr)
            return 1
        _write_progress(progress_path, "animation_started", run_dir=valid_rows[0]["path"])
        animation_result = _run_command(
            [sys.executable, "main.py", "plot", valid_rows[0]["path"], "--animations"],
            cwd=REPO_ROOT,
            stdout_path=release_dir / "chunk_logs" / "best_run_animation.stdout.log",
            stderr_path=release_dir / "chunk_logs" / "best_run_animation.stderr.log",
        )
        _write_progress(progress_path, "animation_finished", run_dir=valid_rows[0]["path"], returncode=animation_result["returncode"])
        if animation_result["returncode"] != 0:
            print(json.dumps({"best_run_animation": animation_result}, indent=2), file=sys.stderr)
            return 1

    _write_progress(progress_path, "archive_started", release_dir=str(release_dir))
    archive_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "archive_release_evidence.py"),
        "--results-dir",
        str(results_dir),
        "--output-dir",
        str(release_dir),
        "--expected-combos",
        str(expected_combos),
    ]
    archive = subprocess.run(archive_cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (release_dir / "archive.stdout.log").write_text(archive.stdout, encoding="utf-8")
    (release_dir / "archive.stderr.log").write_text(archive.stderr, encoding="utf-8")
    _write_progress(progress_path, "archive_finished", returncode=archive.returncode)
    if archive.returncode != 0:
        print(archive.stdout, end="")
        print(archive.stderr, end="", file=sys.stderr)
        return archive.returncode

    _write_progress(progress_path, "verification_started", release_dir=str(release_dir))
    verification = verify_release_evidence(release_dir, expected_combos=expected_combos)
    (release_dir / "verification.json").write_text(json.dumps(verification.to_dict(), indent=2), encoding="utf-8")
    _write_progress(progress_path, "verification_finished", valid=verification.valid, errors=list(verification.errors))
    if not verification.valid:
        print(json.dumps(verification.to_dict(), indent=2), file=sys.stderr)
        return 1

    print(archive.stdout, end="")
    return 0


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description="Run the full ODEM release sweep, then summarize, report, and archive evidence.")
    parser.add_argument("--config", default="parameters.yaml", help="YAML sweep config.")
    parser.add_argument("--results-dir", required=True, help="Directory for run bundles.")
    parser.add_argument("--logs-dir", required=True, help="Directory for runner logs.")
    parser.add_argument("--release-dir", required=True, help="Directory for release reports and archive.")
    parser.add_argument("--expected-combos", type=int, required=True, help="Expected completed valid runs.")
    parser.add_argument("--chunk-size", type=int, default=84, help="Number of combos per subprocess chunk.")
    parser.add_argument("--workers", type=int, default=2, help="Maximum concurrent subprocess chunks.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; each chunk adds its start index.")
    parser.add_argument("--best-run-animation", action="store_true", help="Render one GIF animation for the best valid run.")
    parser.add_argument("--force", action="store_true", help="Remove existing non-empty output directories before running.")
    parser.add_argument("--dry-run", action="store_true", help="Print chunk ranges and commands without running.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars in chunk subprocesses.")
    return parser


def _chunks(total: int, chunk_size: int) -> list[dict[str, int]]:
    return [
        {"index": index, "start": start, "end": min(start + chunk_size, total)}
        for index, start in enumerate(range(0, total, chunk_size))
    ]


def _chunk_command(
    *,
    config: Path,
    results_dir: Path,
    logs_dir: Path,
    chunk: dict[str, int],
    seed: int,
    no_progress: bool,
) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "run",
        "--config",
        str(config),
        "--results-dir",
        str(results_dir),
        "--logs-dir",
        str(logs_dir),
        "--start-index",
        str(chunk["start"]),
        "--end-index",
        str(chunk["end"]),
        "--seed",
        str(seed),
        "--no-slurm",
    ]
    if no_progress:
        command.append("--no-progress")
    return command


def _prepare_outputs(paths: tuple[Path, ...], *, force: bool) -> None:
    for path in paths:
        if not path.exists():
            continue
        if not any(path.iterdir()):
            continue
        if not force:
            raise ValueError(f"{path} is non-empty; pass --force to remove it before running")
        _remove_tree(path)


def _remove_tree(path: Path) -> None:
    resolved = path.resolve()
    if resolved == resolved.anchor or resolved == Path.home().resolve() or resolved == REPO_ROOT:
        raise ValueError(f"refusing to remove unsafe output directory: {resolved}")
    shutil.rmtree(resolved)


def _run_chunks(commands: list[list[str]], *, release_dir: Path, workers: int, progress_path: Path) -> list[dict[str, Any]]:
    log_dir = release_dir / "chunk_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_command,
                command,
                cwd=REPO_ROOT,
                stdout_path=log_dir / f"chunk_{idx:03d}.stdout.log",
                stderr_path=log_dir / f"chunk_{idx:03d}.stderr.log",
                progress_path=progress_path,
            ): idx
            for idx, command in enumerate(commands)
        }
        for future in as_completed(futures):
            result = future.result()
            if result["returncode"] != 0:
                failures.append(result)
    return sorted(failures, key=lambda item: item["index"])


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    index = _command_index(stdout_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_path is not None and index >= 0:
        _write_progress(progress_path, "chunk_started", chunk_index=index, command=command)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=cwd, text=True, stdout=stdout, stderr=stderr)
    if progress_path is not None and index >= 0:
        _write_progress(progress_path, "chunk_finished", chunk_index=index, returncode=completed.returncode)
    return {
        "index": index,
        "command": command,
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _command_index(path: Path) -> int:
    stem = path.stem.split(".")[0]
    if stem.startswith("chunk_"):
        return int(stem.removeprefix("chunk_"))
    return -1


def _write_progress(path: Path, stage: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stage": stage, **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
