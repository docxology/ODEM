from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from odem.visualization import create_state_animation, render_diagnostic_dashboard
from odem.artifacts import RunBundle


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description="Measure ODEM dashboard and GIF rendering time for one run bundle.")
    parser.add_argument("--run-dir", required=True, help="Completed run bundle directory.")
    parser.add_argument("--output-json", required=True, help="Path for benchmark JSON output.")
    parser.add_argument("--max-dashboard-seconds", type=float, default=None, help="Optional dashboard runtime ceiling.")
    parser.add_argument("--max-animation-seconds", type=float, default=None, help="Optional GIF runtime ceiling.")
    parser.add_argument("--fps", type=int, default=4, help="Animation frames per second.")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    dashboard_start = time.perf_counter()
    dashboard_path = render_diagnostic_dashboard(run_dir)
    dashboard_seconds = time.perf_counter() - dashboard_start

    animation_start = time.perf_counter()
    animation_path = create_state_animation(run_dir, fps=args.fps)
    animation_seconds = time.perf_counter() - animation_start
    _refresh_manifest(run_dir, dashboard_path=dashboard_path, animation_path=animation_path)

    results = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "dashboard_path": str(dashboard_path),
        "animation_path": str(animation_path),
        "dashboard_seconds": dashboard_seconds,
        "animation_seconds": animation_seconds,
        "max_dashboard_seconds": args.max_dashboard_seconds,
        "max_animation_seconds": args.max_animation_seconds,
    }
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    failures = []
    if args.max_dashboard_seconds is not None and dashboard_seconds > args.max_dashboard_seconds:
        failures.append(f"dashboard_seconds={dashboard_seconds:.3f} exceeded {args.max_dashboard_seconds:.3f}")
    if args.max_animation_seconds is not None and animation_seconds > args.max_animation_seconds:
        failures.append(f"animation_seconds={animation_seconds:.3f} exceeded {args.max_animation_seconds:.3f}")
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(json.dumps(results, indent=2))
    return 0


def _refresh_manifest(run_dir: Path, *, dashboard_path: Path, animation_path: Path) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("status", "completed")
    extra = {key: manifest[key] for key in ("summary",) if key in manifest}
    bundle = RunBundle.open(run_dir)
    bundle.record_file(dashboard_path, kind="visualization")
    bundle.record_file(animation_path, kind="animation")
    bundle.write_manifest(status=status, extra=extra)


if __name__ == "__main__":
    raise SystemExit(main())
