"""
URL download helpers for cached local resources.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


def cache_download(cache_path: Path | str, url: str) -> Path:
    """
    Download a URL to cache_path if it does not already exist.

    Returns the resolved cache path for the downloaded or existing resource.
    """
    destination = Path(cache_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not destination.exists():
        urlretrieve(url, destination)

    return destination
