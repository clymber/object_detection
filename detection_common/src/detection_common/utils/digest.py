"""
Utilities for stable content digests.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from PIL import ImageOps

from .json_io import json_bytes


def bytes_digest(data: bytes | bytearray | memoryview, hash_alg: str = "sha256") -> str:
    """
    Hash an in-memory byte sequence and return its hexadecimal digest.
    """
    return hashlib.new(hash_alg, data).hexdigest()


def file_digest(path: Path | str, hash_alg: str = "sha256") -> str:
    """
    Hash a complete file without loading it into memory all at once.
    """
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, hash_alg).hexdigest()


def json_digest(data: Any, hash_alg: str = "sha256", *, allow_nan: bool = True) -> str:
    """
    Hash the project's stable JSON representation of a value.
    """
    return bytes_digest(json_bytes(data, allow_nan=allow_nan), hash_alg=hash_alg)


def image_content_digest(
        image: Path | str | PILImage.Image, hash_alg: str = "sha256"
    ) -> str:
    """
    Hash normalized image dimensions and decoded RGBA pixel content.

    EXIF orientation is applied before conversion to RGBA so equivalent images
    hash identically regardless of metadata or source encoding.
    """
    if isinstance(image, (Path, str)):
        with PILImage.open(image) as opened:
            return image_content_digest(opened, hash_alg=hash_alg)

    normalized = ImageOps.exif_transpose(image).convert("RGBA")
    hasher = hashlib.new(hash_alg)
    hasher.update(struct.pack(">II", normalized.width, normalized.height))
    hasher.update(normalized.tobytes())
    return hasher.hexdigest()
