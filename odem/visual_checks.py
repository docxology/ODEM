from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageSequence


def validate_visual_artifact(path: str | Path, *, label: str | None = None) -> tuple[str, ...]:
    artifact = Path(path)
    name = label or artifact.as_posix()
    suffix = artifact.suffix.lower()
    if suffix == ".png":
        return validate_nonblank_png(artifact, label=name)
    if suffix == ".gif":
        return validate_gif(artifact, label=name)
    if suffix == ".pdf":
        return validate_pdf_header(artifact, label=name)
    return ()


def validate_nonblank_png(path: str | Path, *, label: str | None = None) -> tuple[str, ...]:
    artifact = Path(path)
    name = label or artifact.as_posix()
    try:
        with Image.open(artifact) as image:
            rgb = image.convert("RGB")
            extrema = rgb.getextrema()
    except Exception as exc:
        return (f"PNG could not be opened: {name}: {exc}",)
    if all(low == high for low, high in extrema):
        return (f"PNG must be nonblank: {name}",)
    return ()


def validate_gif(path: str | Path, *, label: str | None = None, min_frames: int = 1) -> tuple[str, ...]:
    artifact = Path(path)
    name = label or artifact.as_posix()
    try:
        with Image.open(artifact) as image:
            frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(image)]
    except Exception as exc:
        return (f"GIF could not be opened: {name}: {exc}",)
    if len(frames) < min_frames:
        return (f"GIF must contain at least {min_frames} frame(s): {name}",)
    first_extrema = frames[0].getextrema()
    if all(low == high for low, high in first_extrema):
        return (f"GIF first frame must be nonblank: {name}",)
    return ()


def validate_pdf_header(path: str | Path, *, label: str | None = None) -> tuple[str, ...]:
    artifact = Path(path)
    name = label or artifact.as_posix()
    try:
        header = artifact.read_bytes()[:5]
    except Exception as exc:
        return (f"PDF could not be opened: {name}: {exc}",)
    if header != b"%PDF-":
        return (f"PDF must start with %PDF- header: {name}",)
    return ()
