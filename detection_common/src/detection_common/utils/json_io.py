"""
Utilities for writing stable, human-readable JSON files.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def json_ready(value: Any) -> Any:
    """
    Return a JSON-friendly version of common project metadata values.
    """
    # Mappings become JSON objects, so keys are normalized to strings.
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}

    # Sequences become JSON arrays after recursively converting items.
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]

    # Paths are stored as portable POSIX-style strings.
    if isinstance(value, Path):
        return value.as_posix()

    # NumPy arrays and similar objects become nested Python lists.
    if hasattr(value, "tolist") and callable(value.tolist):
        return json_ready(value.tolist())

    # NumPy scalar values become plain Python scalar values.
    if hasattr(value, "item") and callable(value.item):
        return value.item()

    return value


def write_json(path: Path | str, data: Mapping[str, Any]) -> int:
    """
    Write sorted, indented JSON with a trailing newline.
    """
    return Path(path).write_text(
        json.dumps(json_ready(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path | str) -> dict[str, Any]:
    """
    Read a JSON object from disk.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"JSON file does not contain an object: {path}")
    return data
