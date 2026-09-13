"""
Dataset project and shared runtime roots, resolved when notebooks import this module.
"""

from pathlib import Path

from detection_common.paths import runtime_paths

_PATHS = runtime_paths(Path(__file__), "dataset-builder")
SUBPROJECT_ROOT = _PATHS.subproject_root
WORKSPACE_ROOT = _PATHS.workspace_root
DATA_ROOT = _PATHS.data_root
OUTPUT_ROOT = _PATHS.output_root
