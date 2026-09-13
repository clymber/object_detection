"""
Dataset downloads through the optional Roboflow SDK.
"""

from __future__ import annotations

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DATA_ROOT

if TYPE_CHECKING:
    from roboflow.core.dataset import Dataset


class RoboflowFormat(StrEnum):
    """
    Common Roboflow dataset export formats.
    """

    COCO = "coco"
    YOLOV5 = "yolov5"
    YOLOV8 = "yolov8"
    YOLOV11 = "yolov11"


def dataset_path(project_id: str, version: int) -> Path:
    """
    Return the default download location below the configured data root.
    """
    return DATA_ROOT / "sources" / "roboflow" / project_id / str(version)


def roboflow_api_key() -> str:
    """
    Read the Roboflow credential from the process environment.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY must be set")
    return api_key


def download(
    workspace_id: str,
    project_id: str,
    version: int,
    dataset_format: RoboflowFormat,
) -> Dataset:
    """
    Download a dataset version to its configured source-data directory.
    """
    api_key = roboflow_api_key()
    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise ImportError("Install the roboflow package") from error

    location = str(dataset_path(project_id, version))
    client = Roboflow(api_key=api_key)
    project = client.workspace(workspace_id).project(project_id)
    return project.version(version).download(
        dataset_format.value,
        location=location,
        overwrite=False,
    )


def download_by_url(roboflow_url: str, dataset_format: RoboflowFormat) -> Dataset:
    """
    Download a dataset using workspace, project, and version from its URL.
    """
    url_format = "https://.../<workspace>/<project>/dataset/<version>"
    match = re.search(r"/([^/]+)/([^/]+)/dataset/(\d+)(?:[/?#]|$)", roboflow_url)
    if not match:
        raise ValueError("URL error. Format: " + url_format)
    workspace_id, project_id, version = match.groups()
    return download(workspace_id, project_id, int(version), dataset_format)
