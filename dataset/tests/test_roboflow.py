import sys
import types
from pathlib import Path

import pytest

from dataset_builder import roboflow as roboflow_platform


def test_roboflow_dataset_path_uses_project_and_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Build the default local dataset path from project metadata.
    """
    monkeypatch.setattr(roboflow_platform, "DATA_ROOT", tmp_path)

    path = roboflow_platform.dataset_path(
        "football-0xd5r",
        13,
    )

    assert (
        path
        == tmp_path
        / "sources"
        / "roboflow"
        / "football-0xd5r"
        / "13"
    )


def test_download_dataset_uses_api_metadata_and_default_location(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Return the Roboflow dataset with its local path without a network request.
    """
    calls: dict[str, object] = {}

    class FakeDataset:
        """
        Minimal fake for the Roboflow SDK Dataset.
        """

        def __init__(self, location: str) -> None:
            """
            Store the local dataset location.
            """
            self.location = location

    class FakeRoboflow:
        """
        Minimal fake for the Roboflow SDK client chain.
        """

        def __init__(self, *, api_key: str) -> None:
            """
            Record the API key used to construct the client.
            """
            calls["api_key"] = api_key

        def workspace(self, workspace_id: str) -> "FakeRoboflow":
            """
            Record the requested workspace and return the chained client.
            """
            calls["workspace_id"] = workspace_id
            return self

        def project(self, project_id: str) -> "FakeRoboflow":
            """
            Record the requested project and return the chained client.
            """
            calls["project_id"] = project_id
            return self

        def version(self, version_number: int) -> "FakeRoboflow":
            """
            Record the requested version and return the chained client.
            """
            calls["version_number"] = version_number
            return self

        def download(
            self,
            dataset_format: str,
            *,
            location: str,
            overwrite: bool,
        ) -> FakeDataset:
            """
            Record the download arguments and return a fake dataset.
            """
            calls["dataset_format"] = dataset_format
            calls["location"] = location
            calls["overwrite"] = overwrite
            return FakeDataset(location)

    fake_module = types.ModuleType("roboflow")
    monkeypatch.setattr(fake_module, "Roboflow", FakeRoboflow, raising=False)
    monkeypatch.setitem(sys.modules, "roboflow", fake_module)
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-api-key")
    monkeypatch.setattr(roboflow_platform, "DATA_ROOT", tmp_path)

    dataset = roboflow_platform.download(
        "workspace",
        "football-0xd5r",
        13,
        roboflow_platform.RoboflowFormat.YOLOV11,
    )

    assert isinstance(dataset, FakeDataset)
    assert dataset.location == str(roboflow_platform.dataset_path("football-0xd5r", 13))
    assert calls == {
        "api_key": "test-api-key",
        "workspace_id": "workspace",
        "project_id": "football-0xd5r",
        "version_number": 13,
        "dataset_format": "yolov11",
        "location": str(
            roboflow_platform.dataset_path(
                "football-0xd5r",
                13,
            )
        ),
        "overwrite": False,
    }


def test_download_from_url_uses_dataset_url_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Download a Roboflow dataset using metadata parsed from its dataset URL.
    """
    calls: dict[str, object] = {}

    class FakeDataset:
        """
        Minimal fake for the Roboflow SDK Dataset.
        """

        def __init__(self, location: str) -> None:
            """
            Store the local dataset location.
            """
            self.location = location

    def fake_download(
        workspace_id: str,
        project_id: str,
        version: int,
        dataset_format: roboflow_platform.RoboflowFormat,
    ) -> FakeDataset:
        """
        Record the parsed Roboflow metadata and return a fake dataset.
        """
        calls["workspace_id"] = workspace_id
        calls["project_id"] = project_id
        calls["version"] = version
        calls["dataset_format"] = dataset_format
        return FakeDataset("downloaded")

    monkeypatch.setattr(roboflow_platform, "download", fake_download)

    dataset = roboflow_platform.download_by_url(
        "https://universe.roboflow.com/test-06r5e/"
        "tennis-racket-r6mgq/dataset/4#",
        roboflow_platform.RoboflowFormat.COCO,
    )

    assert isinstance(dataset, FakeDataset)
    assert dataset.location == "downloaded"
    assert calls == {
        "workspace_id": "test-06r5e",
        "project_id": "tennis-racket-r6mgq",
        "version": 4,
        "dataset_format": roboflow_platform.RoboflowFormat.COCO,
    }


def test_download_from_url_rejects_missing_dataset_version() -> None:
    """
    Raise a clear error for Roboflow URLs without a dataset version.
    """
    with pytest.raises(ValueError, match="dataset"):
        roboflow_platform.download_by_url(
            "https://universe.roboflow.com/test-06r5e/tennis-racket-r6mgq",
            roboflow_platform.RoboflowFormat.COCO,
        )


def test_download_dataset_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Raise a clear error before loading Roboflow when no API key is available.
    """
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr(roboflow_platform, "DATA_ROOT", tmp_path)

    with pytest.raises(ValueError, match="ROBOFLOW_API_KEY"):
        roboflow_platform.download(
            "workspace",
            "football-0xd5r",
            13,
            roboflow_platform.RoboflowFormat.YOLOV11,
        )
