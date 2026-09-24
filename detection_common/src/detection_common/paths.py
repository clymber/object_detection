"""
Stable source-project and runtime location discovery.
"""

from __future__ import annotations

import collections.abc
import os
import time
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

MODEL_IDS = frozenset({"yolo11n", "yolox_tiny", "yolox_nano", "rfdetr_small"})


@dataclass(frozen=True)
class RunAllocation:
    """
    A newly reserved run directory and its original UTC creation time.
    """

    path: Path
    created_at: datetime


@dataclass(frozen=True)
class RuntimePaths:
    """
    Paths owned by an installed source subproject and its workspace.
    """

    subproject_root: Path
    workspace_root: Path
    data_root: Path
    output_root: Path


def allocate_run_directory(
    output_root: Path,
    dataset_name: str,
    model: str,
    *,
    clock: collections.abc.Callable[[], datetime] | None = None,
    sleep: collections.abc.Callable[[float], None] = time.sleep,
) -> RunAllocation:
    """
    Exclusively reserve a timestamped run directory below the shared output root.
    """
    if model not in MODEL_IDS:
        raise ValueError(f"Unsupported model ID: {model}")
    if not dataset_name:
        raise ValueError("dataset_name must not be empty")

    now = clock or (lambda: datetime.now(UTC))
    base = Path(output_root).resolve() / "runs" / dataset_name
    base.mkdir(parents=True, exist_ok=True)

    while True:
        created_at = now().astimezone(UTC)
        timestamp = created_at.strftime("%Y%m%dT%H%M%S")
        run_dir = base / f"{model}_{timestamp}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            next_second = created_at.replace(microsecond=0) + timedelta(seconds=1)
            sleep(max(0.0, (next_second - created_at).total_seconds()))
            continue
        return RunAllocation(path=run_dir, created_at=created_at)


def find_subproject_root(module_file: Path, project_name: str) -> Path:
    """
    Find the owning source project independent of the process working directory.
    """
    current = Path(module_file).resolve().parent
    for candidate in (current, *current.parents):
        manifest = candidate / "pyproject.toml"
        if not manifest.is_file():
            continue
        with manifest.open("rb") as stream:
            project = tomllib.load(stream).get("project", {})
        if project.get("name") == project_name:
            return candidate
    raise RuntimeError(f"Subproject root not found: {project_name}")


def absolute_override(name: str, default: Path) -> Path:
    """
    Resolve an optional absolute environment override or the default path.
    """
    value = os.environ.get(name)
    if value is None:
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path: {value}")
    return path.resolve()


def runtime_paths(module_file: Path, project_name: str) -> RuntimePaths:
    """
    Resolve source ownership and configured shared data/output locations.
    """
    subproject = find_subproject_root(module_file, project_name)
    workspace = absolute_override(
        "OBJECT_DETECTION_WORKSPACE_ROOT", subproject.parent
    )
    return RuntimePaths(
        subproject_root=subproject,
        workspace_root=workspace,
        data_root=absolute_override("OBJECT_DETECTION_DATA_ROOT", workspace / "data"),
        output_root=absolute_override(
            "OBJECT_DETECTION_OUTPUT_ROOT", workspace / "outputs"
        ),
    )
