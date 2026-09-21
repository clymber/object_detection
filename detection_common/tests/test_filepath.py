from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from detection_common import allocate_run_directory
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


def test_allocate_run_directory_uses_utc_timestamp_and_reserves_path(
    tmp_path: Path,
) -> None:
    """
    Create the required timestamped path without overwriting an existing run.
    """
    created_at = datetime(2026, 9, 19, 19, 0, 0, tzinfo=UTC)

    allocation = allocate_run_directory(
        tmp_path,
        "basketball",
        "rfdetr_small",
        clock=lambda: created_at,
    )

    assert allocation.created_at == created_at
    assert allocation.path == (
        tmp_path / "runs" / "basketball" / "rfdetr_small_20260919T190000"
    )
    assert allocation.path.is_dir()


def test_allocate_run_directory_retries_at_next_utc_second(tmp_path: Path) -> None:
    """
    Retry a same-second collision with the next UTC timestamp instead of suffixing.
    """
    times: Iterator[datetime] = iter(
        (
            datetime(2026, 9, 19, 19, 0, 0, tzinfo=UTC),
            datetime(2026, 9, 19, 19, 0, 1, tzinfo=UTC),
        )
    )
    occupied = tmp_path / "runs" / "basketball" / "yolo11n_20260919T190000"
    occupied.mkdir(parents=True)
    delays: list[float] = []

    allocation = allocate_run_directory(
        tmp_path,
        "basketball",
        "yolo11n",
        clock=lambda: next(times),
        sleep=delays.append,
    )

    assert allocation.path == (
        tmp_path / "runs" / "basketball" / "yolo11n_20260919T190001"
    )
    assert allocation.path.is_dir()
    assert delays == [1.0]
