"""
Shared utility project's own source and runtime roots.
"""

from pathlib import Path

from .paths import runtime_paths

_PATHS = runtime_paths(Path(__file__), "detection-common")
SUBPROJECT_ROOT = _PATHS.subproject_root
WORKSPACE_ROOT = _PATHS.workspace_root
DATA_ROOT = _PATHS.data_root
OUTPUT_ROOT = _PATHS.output_root
