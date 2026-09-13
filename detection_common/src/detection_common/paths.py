"""
Stable source-project and runtime location discovery.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    """
    Paths owned by an installed source subproject and its workspace.
    """

    subproject_root: Path
    workspace_root: Path
    data_root: Path
    output_root: Path


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
