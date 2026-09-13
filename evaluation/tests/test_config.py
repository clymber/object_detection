"""
Verify detection_evaluation resolves its source project outside the workspace cwd.
"""

import subprocess
import sys
from pathlib import Path


def test_project_root_is_owned_by_importing_module(tmp_path: Path) -> None:
    """
    Resolve the active source root from an unrelated notebook startup directory.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from detection_evaluation.config import SUBPROJECT_ROOT; print(SUBPROJECT_ROOT)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == str(Path(__file__).resolve().parents[1])
