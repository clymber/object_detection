"""
Utilities for stable, human-readable JSON serialization.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO


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


def _json_text(data: Any, *, allow_nan: bool = True) -> str:
    """
    Return data in the project's stable JSON text representation.
    """
    return (
        json.dumps(
            json_ready(data),
            allow_nan=allow_nan,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def json_bytes(data: Any, *, allow_nan: bool = True) -> bytes:
    """
    Encode data in the project's stable UTF-8 JSON representation.
    """
    return _json_text(data, allow_nan=allow_nan).encode("utf-8")


def json_normalize(data: Any, *, allow_nan: bool = True) -> Any:
    """
    Return an independent value containing only JSON-native types.
    """
    return json.loads(json_bytes(data, allow_nan=allow_nan))


def write_json(
    dst: Path | str | TextIO,
    data: Any,
    *,
    allow_nan: bool = True,
) -> int:
    """
    Write sorted, indented JSON with a trailing newline.
    """
    if isinstance(dst, (Path, str)):
        return Path(dst).write_bytes(json_bytes(data, allow_nan=allow_nan))

    return dst.write(_json_text(data, allow_nan=allow_nan))


def read_json(src: Path | str | TextIO) -> Any:
    """
    Read a JSON value from a path or text stream.
    """
    if isinstance(src, (Path, str)):
        data = json.loads(Path(src).read_text(encoding="utf-8"))
    else:
        data = json.load(src)

    return data
