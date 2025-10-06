from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str) -> None:
    """Ensure the parent directory of the given path exists."""

    parent = Path(path).resolve().parent
    parent.mkdir(parents=True, exist_ok=True)
