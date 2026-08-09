"""Filesystem helpers for datasets and result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ALLOWED_IMAGE_EXTENSIONS


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (including parents) and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_images(root: str | Path, *, recursive: bool = True) -> list[Path]:
    """Return sorted image files under `root`, filtered by extension."""
    root = Path(root)
    if not root.exists():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    )


def read_json(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write JSON with parent directories created on demand."""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    return path


def save_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Persist a DataFrame as CSV, creating parent directories as needed."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    return path
