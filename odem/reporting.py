from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Iterable
import json

from odem.analysis import summarize_run, summarize_sweep, write_summary_csv, write_summary_json
from odem.summary_schema import SWEEP_REPORT_COLUMNS
from odem.visualization import render_sweep_summary


@dataclass(frozen=True)
class ReportPaths:
    markdown_path: Path
    html_path: Path
    data_path: Path | None = None
    figure_path: Path | None = None
    gallery_path: Path | None = None
    assets: tuple[Path, ...] = ()


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
    assets = _manifest_asset_paths(run_path)
    gallery_path = _write_gallery(out, title=f"ODEM Run Gallery: {summary['run_id']}", summary=summary, links=[markdown_path, html_path, data_path], assets=assets)
    return ReportPaths(markdown_path=markdown_path, html_path=html_path, data_path=data_path, gallery_path=gallery_path, assets=tuple(assets))


def create_sweep_report(
    parent_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    release_links: Iterable[str | Path] = (),
) -> ReportPaths:
    sweep_path = Path(parent_dir)
    out = Path(output_dir) if output_dir else sweep_path / "reports"
    out.mkdir(parents=True, exist_ok=True)

    rows = summarize_sweep(sweep_path)
    markdown = _sweep_markdown(rows)
    markdown_path = out / "sweep_report.md"
    html_path = out / "sweep_report.html"
    data_path = write_summary_json(rows, out / "sweep_summary.json")
    write_summary_csv(rows, out / "sweep_summary.csv")
    figure_path = render_sweep_summary(rows, out / "sweep_summary.png")

    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(_markdown_to_html(markdown, title="ODEM Sweep Report"), encoding="utf-8")
    links = [markdown_path, html_path, data_path, out / "sweep_summary.csv", figure_path, *(Path(link) for link in release_links)]
    assets = [figure_path, *_best_sweep_assets(rows)]
    gallery_path = _write_gallery(
        out,
        title="ODEM Sweep Gallery",
        summary={"valid_for_ranking": sum(1 for row in rows if row.get("valid_for_ranking")), "runs": len(rows)},
        links=links,
        assets=assets,
    )
    return ReportPaths(markdown_path=markdown_path, html_path=html_path, data_path=data_path, figure_path=figure_path, gallery_path=gallery_path, assets=tuple(assets))


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
        f"- Valid bundle: {summary.get('valid_bundle')}",
        f"- Valid for ranking: {summary.get('valid_for_ranking')}",
        f"- Validation issues: {_issues(summary.get('issues'))}",
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
    valid_rows = [row for row in rows if row.get("valid_for_ranking")]
    invalid_rows = [row for row in rows if not row.get("valid_for_ranking")]
    lines = [
        "# ODEM Sweep Report",
        "",
        f"- Runs summarized: {len(rows)}",
        f"- Valid for ranking: {len(valid_rows)}",
        f"- Invalid or diagnostic-only runs: {len(invalid_rows)}",
        "",
    ]
    if not valid_rows:
        lines.append("No completed run bundles with finite ranking metrics found.")
        return "\n".join(lines) + "\n"

    best = valid_rows[0]
    lines.extend(
        [
            f"- Best run by free action: {best.get('run_id')}",
            f"- Best free action: {_fmt(best.get('free_action'))}",
            f"- Best MSE: {_fmt(best.get('mse'))}",
            "",
            f"| {' | '.join(SWEEP_REPORT_COLUMNS)} |",
            "| ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {run_id} | {combo} | {free_action} | {mse} | {finite} | {valid} | {issues} |".format(
                rank=rank,
                run_id=row.get("run_id"),
                combo=row.get("combo_index"),
                free_action=_fmt(row.get("free_action")),
                mse=_fmt(row.get("mse")),
                finite=row.get("all_numeric_outputs_finite"),
                valid=row.get("valid_for_ranking"),
                issues=_issues(row.get("issues")),
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


def _manifest_asset_paths(run_dir: Path) -> list[Path]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = []
    for artifact in manifest.get("artifacts", []):
        path = run_dir / artifact.get("path", "")
        if path.suffix.lower() in {".png", ".gif", ".pdf"} and path.exists():
            assets.append(path)
    return sorted(assets)


def _best_sweep_assets(rows: list[dict[str, Any]]) -> list[Path]:
    if not rows:
        return []
    best_path = Path(str(rows[0].get("path", "")))
    candidates = [
        best_path / "plots" / "diagnostic_dashboard.png",
        best_path / "animations" / "state_estimation.gif",
    ]
    return [path for path in candidates if path.exists()]


def _write_gallery(out: Path, *, title: str, summary: dict[str, Any], links: list[Path], assets: list[Path]) -> Path:
    gallery_path = out / "index.html"
    body = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        f"<title>{escape(title)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:0;background:#f6f8fa;color:#24292f}.wrap{max-width:1120px;margin:0 auto;padding:24px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.card{background:white;border:1px solid #d0d7de;border-radius:6px;padding:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-top:18px}.asset{background:white;border:1px solid #d0d7de;border-radius:6px;padding:10px}.asset img{max-width:100%;height:auto;display:block}.links a{margin-right:12px}code{font-family:ui-monospace,monospace}</style>",
        "</head>",
        "<body><main class=\"wrap\">",
        f"<h1>{escape(title)}</h1>",
        "<section class=\"cards\">",
    ]
    for key, value in summary.items():
        label = str(key).replace("_", " ").capitalize()
        body.append(f"<div class=\"card\"><strong>{escape(label)}</strong><br><code>{escape(str(value))}</code></div>")
    body.extend(["</section>", "<section class=\"links\"><h2>Report Files</h2>"])
    for link in links:
        body.append(f"<a href=\"{escape(_rel(out, link))}\">{escape(link.name)}</a>")
    body.extend(["</section>", "<section class=\"grid\"><h2>Assets</h2>"])
    if not assets:
        body.append("<p>No visual assets recorded.</p>")
    for asset in assets:
        rel = escape(_rel(out, asset))
        if asset.suffix.lower() in {".png", ".gif"}:
            body.append(f"<figure class=\"asset\"><a href=\"{rel}\"><img src=\"{rel}\" alt=\"{escape(asset.name)}\"></a><figcaption>{escape(asset.name)}</figcaption></figure>")
        else:
            body.append(f"<div class=\"asset\"><a href=\"{rel}\">{escape(asset.name)}</a></div>")
    body.extend(["</section>", "</main></body></html>"])
    gallery_path.write_text("\n".join(body), encoding="utf-8")
    return gallery_path


def _rel(base: Path, path: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.resolve().as_posix()


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


def _issues(value: Any) -> str:
    if not value:
        return "none"
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)
