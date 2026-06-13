from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
import json

from odem.analysis import summarize_run, summarize_sweep, write_summary_csv, write_summary_json
from odem.artifacts import RunBundle
from odem.config import load_sweep
from odem.experiment import ExperimentRunner, RunnerOptions
from odem.reporting import create_run_report, create_sweep_report
from odem.visualization import create_state_animation, dispatch_static_plots, render_diagnostic_dashboard


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="odem", description="Run, inspect, and visualize ODEM experiments.")
    subparsers = parser.add_subparsers(dest="command", required=False)

    run = subparsers.add_parser("run", help="Run an ODEM parameter sweep.")
    run.add_argument("--config", default="parameters.yaml", help="YAML configuration path.")
    run.add_argument("--results-dir", default="results", help="Directory for run bundles.")
    run.add_argument("--logs-dir", default="logs", help="Directory for sweep logs.")
    run.add_argument("--start-index", type=int, default=None, help="First combo index to run.")
    run.add_argument("--end-index", type=int, default=None, help="Exclusive ending combo index.")
    run.add_argument("--max-combos", type=int, default=None, help="Limit combos for smoke runs.")
    run.add_argument("--seed", type=int, default=42, help="Global deterministic seed.")
    run.add_argument("--no-slurm", action="store_true", help="Ignore SLURM_ARRAY_TASK_* environment variables.")
    run.add_argument("--no-static-plots", action="store_true", help="Skip static PDF plots.")
    run.add_argument("--dashboard", action="store_true", help="Render compact dashboard PNG; enabled by default.")
    run.add_argument("--no-dashboard", action="store_true", help="Skip compact dashboard PNG.")
    run.add_argument("--animations", action="store_true", help="Generate state-estimation GIF animations.")
    run.add_argument("--allow-partial", action="store_true", help="Return partial results when one or more combinations fail.")
    run.add_argument("--quiet-progress", "--no-progress", action="store_true", help="Disable tqdm progress bars.")

    validate = subparsers.add_parser("validate", help="Validate and count a YAML sweep.")
    validate.add_argument("--config", default="parameters.yaml", help="YAML configuration path.")

    summarize = subparsers.add_parser("summarize", help="Summarize a run bundle or sweep directory.")
    summarize.add_argument("path", help="Run bundle or sweep directory.")
    summarize.add_argument("--json", "--output-json", dest="json_path", default=None, help="Write summary JSON.")
    summarize.add_argument("--csv", "--output-csv", dest="csv_path", default=None, help="Write sweep summary CSV.")

    plot = subparsers.add_parser("plot", help="Render visual artifacts from a run bundle.")
    plot.add_argument("run_dir", help="Run bundle directory.")
    plot.add_argument("--static", action="store_true", help="Render all existing static PDF plots.")
    plot.add_argument("--dashboard", action="store_true", help="Render compact dashboard PNG.")
    plot.add_argument("--animations", action="store_true", help="Render state-estimation GIF.")

    report = subparsers.add_parser("report", help="Create Markdown/HTML reports for a run or sweep.")
    report.add_argument("path", help="Run bundle or sweep directory.")
    report.add_argument("--output-dir", default=None, help="Optional report output directory.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"

    if command == "run":
        try:
            result = _run(args)
        except RuntimeError as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 1
        print(json.dumps({"completed": len(result["completed"]), "failed": len(result["failed"]), "log_dir": result["log_dir"]}, indent=2))
        return 0 if not result["failed"] else 1
    if command == "validate":
        sweep = load_sweep(args.config)
        print(json.dumps({"config": str(sweep.source), "combination_count": sweep.combination_count}, indent=2))
        return 0
    if command == "summarize":
        _summarize(args)
        return 0
    if command == "plot":
        _plot(args)
        return 0
    if command == "report":
        _report(args)
        return 0

    parser.error(f"Unknown command: {command}")
    return 2


def _run(args: Namespace) -> dict:
    options = RunnerOptions(
        config_path=Path(args.config),
        results_dir=Path(args.results_dir),
        logs_dir=Path(args.logs_dir),
        seed_value=args.seed,
        start_index=args.start_index,
        end_index=args.end_index,
        max_combos=args.max_combos,
        use_slurm_env=not args.no_slurm,
        static_plots=not args.no_static_plots,
        dashboard=not args.no_dashboard,
        animations=args.animations,
        tqdm_disable=args.quiet_progress,
        allow_partial=args.allow_partial,
    )
    return ExperimentRunner(options).run()


def _summarize(args: Namespace) -> None:
    path = Path(args.path)
    if (path / "snapshot.json").exists() or (path / "vfe.npy").exists():
        payload = summarize_run(path)
    else:
        payload = summarize_sweep(path)

    if args.json_path:
        write_summary_json(payload if isinstance(payload, list) else [payload], args.json_path)
    if args.csv_path:
        rows = payload if isinstance(payload, list) else [payload]
        write_summary_csv(rows, args.csv_path)
    print(json.dumps(payload, indent=2))


def _plot(args: Namespace) -> None:
    run_dir = Path(args.run_dir)
    outputs = []
    if args.static:
        outputs.extend(dispatch_static_plots(run_dir))
    if args.dashboard or not (args.static or args.animations):
        outputs.append(render_diagnostic_dashboard(run_dir))
    if args.animations:
        outputs.append(create_state_animation(run_dir))

    bundle = RunBundle.open(run_dir)
    for path in outputs:
        kind = "animation" if Path(path).suffix.lower() in {".gif", ".mp4"} else "visualization"
        bundle.record_file(path, kind=kind)
    bundle.write_manifest(status="completed")
    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))


def _report(args: Namespace) -> None:
    path = Path(args.path)
    if (path / "snapshot.json").exists() or (path / "vfe.npy").exists():
        report = create_run_report(path, args.output_dir)
    else:
        report = create_sweep_report(path, args.output_dir)
    print(
        json.dumps(
            {
                "markdown": str(report.markdown_path),
                "html": str(report.html_path),
                "data": None if report.data_path is None else str(report.data_path),
                "figure": None if report.figure_path is None else str(report.figure_path),
                "gallery": None if report.gallery_path is None else str(report.gallery_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
