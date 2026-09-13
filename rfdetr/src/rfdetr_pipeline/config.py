"""
RF-DETR project and shared runtime roots, resolved on notebook import.
"""

from pathlib import Path

from detection_common.paths import runtime_paths

_PATHS = runtime_paths(Path(__file__), "rfdetr-pipeline")
SUBPROJECT_ROOT = _PATHS.subproject_root
WORKSPACE_ROOT = _PATHS.workspace_root
DATA_ROOT = _PATHS.data_root
OUTPUT_ROOT = _PATHS.output_root
