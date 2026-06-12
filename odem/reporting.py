from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any
import json

from odem.analysis import summarize_run, summarize_sweep, write_summary_csv, write_summary_json
from odem.visualization import render_sweep_summary


@dataclass(frozen=True)
class ReportPaths:
    markdown_path: Path
    html_path: Path
    data_path: Path | None = None
    figure_path: Path | None = None


def create_run_report(run_dir: str | Path, output_dir: str | Path | None = None) -> ReportPaths:
    run_path = Path(run_dir)
    out = Path(output_dir) if output_dir else run_path / "reports"
    out.mkdir(parents=True, exist_ok=True)

    summary = summarize_run(run_path)
    markdown = _run_markdown(summary)
    markdown_path = out / "run_report.md"
    html_path = out / "run_report.html"
    data_path = out / "run_summary.json"

    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_markdown_to_html(markdown, title=f"ODEM Run Report: {summary['run_id']}"), encoding="utf-8")
    data_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ReportPaths(markdown_path=markdown_path, html_path=html_path, data_path=data_path)


def create_sweep_report(parent_dir: str | Path, output_dir: str | Path | None = None) -> ReportPaths:
    sweep_path = Path(parent_dir)
    out = Path(output_dir) if output_dir else sweep_path / "reports"
    out.mkdir(parents=True, exist_ok=True)

    rows = summarize_sweep(sweep_path)
    markdown = _sweep_markdown(rows)
    markdown_path = out / "sweep_report.md"
    html_path = out / "sweep_report.html"
    data_path = write_summary_json(rows, out / "sweep_summary.json")
    write_summary_csv(rows, out / "sweep_summary.csv")
    figure_path = render_sweep_summary(rows, out / "sweep_summary.png") if rows else None

    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_markdown_to_html(markdown, title="ODEM Sweep Report"), encoding="utf-8")
    return ReportPaths(markdown_path=markdown_path, html_path=html_path, data_path=data_path, figure_path=figure_path)


def _run_markdown(summary: dict[str, Any]) -> str:
    artifacts = _artifact_lines(Path(summary["path"]))
    lines = [
        f"# ODEM Run Report: {summary['run_id']}",
        "",
        f"- Combo index: {summary.get('combo_index')}",
        f"- Timesteps: {summary.get('timesteps')}",
        f"- Free action: {_fmt(summary.get('free_action'))}",
        f"- MSE: {_fmt(summary.get('mse'))}",
        f"- Accuracy total: {_fmt(summary.get('accuracy_total'))}",
        f"- Complexity total: {_fmt(summary.get('complexity_total'))}",
        f"- Numeric outputs finite: {summary.get('all_numeric_outputs_finite')}",
        f"- Final theta: {summary.get('final_theta')}",
        f"- Final lambda x: {summary.get('final_lambda_x')}",
        f"- Final lambda y: {summary.get('final_lambda_y')}",
        "",
        "## Model",
        "",
        "```json",
        json.dumps({"gp": summary.get("gp"), "gm": summary.get("gm")}, indent=2),
        "```",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(artifacts or ["No manifest artifacts recorded."])
    lines.append("")
    return "\n".join(lines)


def _sweep_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# ODEM Sweep Report",
        "",
        f"- Runs summarized: {len(rows)}",
        "",
    ]
    if not rows:
        lines.append("No completed run bundles found.")
        return "\n".join(lines) + "\n"

    best = rows[0]
    lines.extend(
        [
            f"- Best run by free action: {best.get('run_id')}",
            f"- Best free action: {_fmt(best.get('free_action'))}",
            f"- Best MSE: {_fmt(best.get('mse'))}",
            "",
            "| Rank | Run ID | Combo | Free action | MSE | Finite |",
            "| ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {run_id} | {combo} | {free_action} | {mse} | {finite} |".format(
                rank=rank,
                run_id=row.get("run_id"),
                combo=row.get("combo_index"),
                free_action=_fmt(row.get("free_action")),
                mse=_fmt(row.get("mse")),
                finite=row.get("all_numeric_outputs_finite"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _artifact_lines(run_dir: Path) -> list[str]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        f"- `{artifact['path']}` ({artifact['kind']}, {artifact['bytes']} bytes, sha256 `{artifact['sha256'][:12]}`)"
        for artifact in manifest.get("artifacts", [])
    ]


def _markdown_to_html(markdown: str, *, title: str) -> str:
    body = []
    in_code = False
    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            body.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
        elif in_code:
            body.append(escape(line))
        elif line.startswith("# "):
            body.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<p>{escape(line)}</p>")
        elif line.startswith("|"):
            body.append(f"<pre>{escape(line)}</pre>")
        elif line:
            body.append(f"<p>{escape(line)}</p>")
        else:
            body.append("")
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            f"<title>{escape(title)}</title>",
            "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;line-height:1.5;padding:0 1rem}pre{background:#f6f8fa;padding:1rem;overflow:auto}code{font-family:ui-monospace,monospace}</style>",
            "</head>",
            "<body>",
            *body,
            "</body>",
            "</html>",
        ]
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
