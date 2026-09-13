from pathlib import Path

from detection_common.utils.filepath import increment_path


def test_increment_path_returns_base_when_missing(tmp_path: Path) -> None:
    """
    Return the requested path when it is not already occupied.
    """
    path = tmp_path / "experiment"

    assert increment_path(path) == path
    assert not path.exists()


def test_increment_path_appends_two_when_base_exists(tmp_path: Path) -> None:
    """
    Return a dash-two path when the base path already exists.
    """
    path = tmp_path / "experiment"
    path.mkdir()

    assert increment_path(path) == tmp_path / "experiment-2"


def test_increment_path_skips_occupied_suffixes(tmp_path: Path) -> None:
    """
    Return the next available numeric suffix after occupied paths.
    """
    for name in ("experiment", "experiment-2", "experiment-3"):
        (tmp_path / name).mkdir()

    assert increment_path(tmp_path / "experiment") == tmp_path / "experiment-4"
