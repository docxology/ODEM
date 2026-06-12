from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import uuid

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _json_default(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return value.tolist() if value.ndim else value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RunBundle:
    path: Path
    combo_index: int | None = None
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        base_dir: str | Path,
        *,
        combo_index: int | None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RunBundle":
        base = Path(base_dir)
        actual_run_id = run_id or _default_run_id(combo_index)
        path = base / actual_run_id
        path.mkdir(parents=True, exist_ok=False)
        bundle = cls(path=path, combo_index=combo_index, run_id=actual_run_id, metadata=metadata or {})
        bundle.write_manifest(status="created")
        return bundle

    @classmethod
    def open(cls, path: str | Path) -> "RunBundle":
        bundle_path = Path(path)
        manifest_path = bundle_path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            manifest = {}
        return cls(
            path=bundle_path,
            combo_index=manifest.get("combo_index"),
            run_id=manifest.get("run_id", bundle_path.name),
            metadata=manifest.get("metadata", {}),
            artifacts=manifest.get("artifacts", []),
        )

    def save_array(self, name: str, value: Any, *, subdir: str | None = None) -> Path:
        directory = self.path / subdir if subdir else self.path
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / f"{name}.npy"
        np.save(filepath, _to_numpy(value))
        self.record_file(filepath, name=name, kind="raw_array")
        return filepath

    def save_json(self, name: str, value: Any, *, subdir: str | None = None) -> Path:
        directory = self.path / subdir if subdir else self.path
        directory.mkdir(parents=True, exist_ok=True)
        filepath = directory / f"{name}.json"
        filepath.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
        self.record_file(filepath, name=name, kind="metadata")
        return filepath

    def record_file(self, filepath: str | Path, *, name: str | None = None, kind: str = "artifact") -> None:
        path = Path(filepath)
        rel_path = path.relative_to(self.path)
        record = {
            "name": name or path.stem,
            "kind": kind,
            "path": rel_path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        self.artifacts = [a for a in self.artifacts if a["path"] != record["path"]]
        self.artifacts.append(record)

    def write_manifest(self, *, status: str, extra: dict[str, Any] | None = None) -> Path:
        manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "combo_index": self.combo_index,
            "status": status,
            "created_at_utc": _utc_now(),
            "metadata": self.metadata,
            "artifacts": sorted(self.artifacts, key=lambda item: item["path"]),
        }
        if extra:
            manifest.update(extra)
        filepath = self.path / "manifest.json"
        filepath.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
        return filepath


def _default_run_id(combo_index: int | None) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    suffix = uuid.uuid4().hex[:8]
    combo = "NA" if combo_index is None else str(combo_index)
    return f"{timestamp}_{suffix}_C{combo}"
