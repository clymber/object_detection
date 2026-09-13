"""
Tests for project ownership and shared runtime path discovery.
"""

from pathlib import Path

import pytest

from detection_common.paths import find_subproject_root, runtime_paths


def _manifest(root: Path, name: str) -> Path:
    """
    Create a minimal project manifest and return a source module path.
    """
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\n', encoding="utf-8"
    )
    module = root / "src" / "example" / "config.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    return module


@pytest.mark.parametrize("startup", ["workspace", "subproject", "sibling", "other"])
def test_discovery_ignores_startup_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, startup: str
) -> None:
    """
    Find the module owner even when Jupyter starts elsewhere.
    """
    module = _manifest(tmp_path / "project", "example-project")
    _manifest(tmp_path / "sibling", "other-project")
    directories = {
        "workspace": tmp_path,
        "subproject": module.parent.parent.parent,
        "sibling": tmp_path / "sibling",
        "other": tmp_path.parent,
    }
    monkeypatch.chdir(directories[startup])
    assert find_subproject_root(module, "example-project") == tmp_path / "project"


def test_runtime_defaults_and_absolute_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Use workspace defaults and allow mounted absolute runtime locations.
    """
    module = _manifest(tmp_path / "project", "example-project")
    defaults = runtime_paths(module, "example-project")
    assert defaults.workspace_root == tmp_path
    assert defaults.data_root == tmp_path / "data"
    assert defaults.output_root == tmp_path / "outputs"

    mounted = tmp_path / "mount"
    monkeypatch.setenv("OBJECT_DETECTION_WORKSPACE_ROOT", str(mounted))
    monkeypatch.setenv("OBJECT_DETECTION_DATA_ROOT", str(tmp_path / "dataset"))
    overridden = runtime_paths(module, "example-project")
    assert overridden.workspace_root == mounted
    assert overridden.data_root == tmp_path / "dataset"
    assert overridden.output_root == mounted / "outputs"


def test_reject_relative_override_and_wrong_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Reject ambiguous overrides and manifests owned by other projects.
    """
    module = _manifest(tmp_path / "project", "other-project")
    with pytest.raises(RuntimeError, match="example-project"):
        find_subproject_root(module, "example-project")
    monkeypatch.setenv("OBJECT_DETECTION_DATA_ROOT", "relative/data")
    with pytest.raises(ValueError, match="OBJECT_DETECTION_DATA_ROOT"):
        runtime_paths(module, "other-project")
