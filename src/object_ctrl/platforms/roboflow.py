"""
Legacy Roboflow helpers kept for the existing Renku environment during Stage A.
"""

from __future__ import annotations

import os
import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import PROJECT_ROOT

if TYPE_CHECKING:
    from roboflow.core.dataset import Dataset


class RoboflowFormat(StrEnum):
    """
    Commonly used Roboflow dataset formats.
    """

    COCO = "coco"
    YOLOV5 = "yolov5"
    YOLOV8 = "yolov8"
    YOLOV11 = "yolov11"


def dataset_path(project_id: str, version: int) -> Path:
    """
    Return the legacy download path until Renku migration is complete.
    """
    return PROJECT_ROOT / "datasets" / "sources" / "roboflow" / project_id / str(
        version
    )


def roboflow_api_key() -> str:
    """
    Read the credential from the environment without a source-code default.
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
    Download a Roboflow dataset version into the legacy dataset directory.
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
    Download using workspace metadata parsed from a Roboflow dataset URL.
    """
    url_format = "https://.../<workspace>/<project>/dataset/<version>"
    match = re.search(r"/([^/]+)/([^/]+)/dataset/(\d+)(?:[/?#]|$)", roboflow_url)
    if not match:
        raise ValueError("URL error. Format: " + url_format)
    workspace_id, project_id, version = match.groups()
    return download(workspace_id, project_id, int(version), dataset_format)
