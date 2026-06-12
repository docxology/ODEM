from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
import hashlib
import json
import tarfile

from odem.analysis import summarize_sweep
from odem.reporting import create_sweep_report
from odem.visual_checks import validate_gif, validate_nonblank_png


ARCHIVE_NAME = "odem_release_evidence.tar.gz"
MANIFEST_NAME = "release_evidence.json"
REPORT_FILENAMES = (
    "sweep_report.md",
    "sweep_report.html",
    "sweep_summary.csv",
    "sweep_summary.json",
    "sweep_summary.png",
    "index.html",
)


@dataclass(frozen=True)
class ReleaseVerification:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    manifest: dict[str, Any]
    checked_paths: tuple[str, ...]
    archive_bytes: int | None = None
    archive_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "manifest": self.manifest,
            "checked_paths": list(self.checked_paths),
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
        }


def archive_release_evidence(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_combos: int | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    results_path = Path(results_dir)
    release_path = Path(output_dir)
    report_dir = release_path / "reports"
    manifest_path = release_path / MANIFEST_NAME
    archive_path = release_path / ARCHIVE_NAME
    release_path.mkdir(parents=True, exist_ok=True)

    rows = summarize_sweep(results_path)
    completed_runs = sum(1 for row in rows if row.get("valid_bundle") and row.get("manifest_status") == "completed")
    invalid_runs = len(rows) - completed_runs
    complete = expected_combos is None or completed_runs == expected_combos
    report_paths = create_sweep_report(
        results_path,
        output_dir=report_dir,
        release_links=[manifest_path, archive_path],
    )
    manifest = {
        "created_at_utc": _utc_now(),
        "results_dir": _display_path(results_path),
        "output_dir": _display_path(release_path),
        "expected_combos": expected_combos,
        "completed_runs": completed_runs,
        "invalid_runs": invalid_runs,
        "complete": complete,
        "archive_path": _display_path(archive_path),
        "archive_bytes": None,
        "archive_sha256": None,
        "reports": {
            "markdown": str(report_paths.markdown_path),
            "html": str(report_paths.html_path),
            "data": str(report_paths.data_path) if report_paths.data_path else None,
            "figure": str(report_paths.figure_path) if report_paths.figure_path else None,
            "gallery": str(report_paths.gallery_path) if report_paths.gallery_path else None,
        },
    }

    archive_tmp = archive_path.with_name(f".{archive_path.name}.tmp")
    embedded_manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.embedded.tmp")
    final_manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    for tmp in (archive_tmp, embedded_manifest_tmp, final_manifest_tmp):
        if tmp.exists():
            tmp.unlink()

    try:
        embedded_manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with tarfile.open(archive_tmp, "w:gz") as tar:
            tar.add(results_path, arcname="results")
            tar.add(report_dir, arcname="reports")
            tar.add(embedded_manifest_tmp, arcname=MANIFEST_NAME)
        archive_bytes = archive_tmp.stat().st_size
        archive_sha256 = _sha256(archive_tmp)
        manifest = {
            **manifest,
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha256,
            "allow_incomplete": bool(allow_incomplete),
        }
        final_manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        archive_tmp.replace(archive_path)
        final_manifest_tmp.replace(manifest_path)
    except Exception:
        for tmp in (archive_tmp, embedded_manifest_tmp, final_manifest_tmp):
            if tmp.exists():
                tmp.unlink()
        raise
    finally:
        if embedded_manifest_tmp.exists():
            embedded_manifest_tmp.unlink()

    return manifest


def verify_release_evidence(
    release_dir: str | Path,
    *,
    expected_combos: int | None = None,
    allow_incomplete: bool = False,
) -> ReleaseVerification:
    release_path = Path(release_dir)
    manifest_path = release_path / MANIFEST_NAME
    archive_path = release_path / ARCHIVE_NAME
    reports_dir = release_path / "reports"
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []
    manifest: dict[str, Any] = {}
    archive_bytes: int | None = None
    archive_sha256: str | None = None

    if not manifest_path.exists():
        errors.append(f"release manifest missing: {manifest_path}")
        return ReleaseVerification(False, tuple(errors), tuple(warnings), manifest, tuple(checked))
    checked.append(MANIFEST_NAME)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"release manifest is invalid JSON: {exc}")
        return ReleaseVerification(False, tuple(errors), tuple(warnings), manifest, tuple(checked))

    expected = expected_combos if expected_combos is not None else manifest.get("expected_combos")
    completed_runs = manifest.get("completed_runs")
    if expected is not None and completed_runs != expected:
        message = f"completed_runs={completed_runs} did not match expected_combos={expected}"
        if allow_incomplete:
            warnings.append(message)
        else:
            errors.append(message)
    if manifest.get("invalid_runs") not in (0, None):
        errors.append(f"invalid_runs={manifest.get('invalid_runs')} must be 0")
    if manifest.get("complete") is not True:
        message = "release manifest complete=true is required"
        if allow_incomplete:
            warnings.append(message)
        else:
            errors.append(message)

    if not archive_path.exists():
        errors.append(f"release tarball missing: {archive_path}")
    else:
        checked.append(ARCHIVE_NAME)
        archive_bytes = archive_path.stat().st_size
        archive_sha256 = _sha256(archive_path)
        if archive_bytes <= 0:
            errors.append("release tarball must not be empty")
        if manifest.get("archive_bytes") is not None and manifest.get("archive_bytes") != archive_bytes:
            errors.append("release tarball byte count does not match manifest")
        if manifest.get("archive_sha256") and manifest.get("archive_sha256") != archive_sha256:
            errors.append("release tarball SHA-256 does not match manifest")
        if not tarfile.is_tarfile(archive_path):
            errors.append(f"release tarball is not a valid tarball: {archive_path}")
        else:
            try:
                with tarfile.open(archive_path, "r:gz") as tar:
                    names = set(tar.getnames())
                for member in (MANIFEST_NAME, "reports/index.html", "reports/sweep_summary.png"):
                    if member not in names:
                        errors.append(f"release tarball missing expected member: {member}")
            except tarfile.TarError as exc:
                errors.append(f"release tarball is not a valid tarball: {exc}")

    for filename in REPORT_FILENAMES:
        path = reports_dir / filename
        rel = f"reports/{filename}"
        if not path.exists():
            errors.append(f"release report missing: {rel}")
        else:
            checked.append(rel)

    summary_path = reports_dir / "sweep_summary.json"
    rows: list[dict[str, Any]] = []
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                errors.append("reports/sweep_summary.json must contain a list")
            else:
                rows = payload
                if expected is not None and len(rows) != expected:
                    message = f"sweep_summary.json row count {len(rows)} did not match expected_combos={expected}"
                    if allow_incomplete:
                        warnings.append(message)
                    else:
                        errors.append(message)
        except json.JSONDecodeError as exc:
            errors.append(f"reports/sweep_summary.json is invalid JSON: {exc}")

    sweep_png = reports_dir / "sweep_summary.png"
    if sweep_png.exists():
        _check_nonblank_png(sweep_png, errors, checked, "reports/sweep_summary.png")

    if rows:
        best = rows[0]
        best_path = Path(str(best.get("path", "")))
        if best_path.exists():
            dashboard = best_path / "plots" / "diagnostic_dashboard.png"
            if dashboard.exists():
                _check_nonblank_png(dashboard, errors, checked, _result_rel(dashboard))
            else:
                errors.append(f"best-run dashboard missing: {dashboard}")
            animation = best_path / "animations" / "state_estimation.gif"
            if animation.exists():
                _check_gif(animation, errors, checked, _result_rel(animation))
            else:
                warnings.append(f"best-run GIF not present: {animation}")
        else:
            errors.append(f"best-run path from summary does not exist: {best_path}")

    gallery_path = reports_dir / "index.html"
    if gallery_path.exists():
        _check_gallery_links(gallery_path, errors, checked)

    valid = not errors
    return ReleaseVerification(
        valid=valid,
        errors=tuple(errors),
        warnings=tuple(warnings),
        manifest=manifest,
        checked_paths=tuple(dict.fromkeys(checked)),
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
    )


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    text = str(path.absolute())
    if text.startswith("/private/tmp/"):
        return "/tmp/" + text.removeprefix("/private/tmp/")
    return text


def _check_nonblank_png(path: Path, errors: list[str], checked: list[str], label: str) -> None:
    checked.append(label)
    errors.extend(validate_nonblank_png(path, label=label))


def _check_gif(path: Path, errors: list[str], checked: list[str], label: str) -> None:
    checked.append(label)
    errors.extend(validate_gif(path, label=label))


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.links.append(value)


def _check_gallery_links(gallery_path: Path, errors: list[str], checked: list[str]) -> None:
    parser = _LinkParser()
    parser.feed(gallery_path.read_text(encoding="utf-8"))
    for link in parser.links:
        parsed = urlparse(link)
        if parsed.scheme in {"http", "https"}:
            errors.append(f"gallery link must be local-only: {link}")
            continue
        if parsed.scheme == "mailto" or link.startswith("#"):
            continue
        candidate = Path(unquote(parsed.path))
        if not candidate.is_absolute():
            candidate = (gallery_path.parent / candidate).resolve()
        if not candidate.exists():
            errors.append(f"gallery local link is broken: {link}")
        else:
            checked.append(_relative_or_absolute(gallery_path.parent, candidate))


def _relative_or_absolute(base: Path, path: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return str(path)


def _result_rel(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        return "/".join(parts[index:])
    return str(path)
