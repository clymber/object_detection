import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
import yaml

from ultralytics_pipeline import ultralytics


def test_reserved_run_callback_rejects_a_directory_mismatch(tmp_path: Path) -> None:
    """
    Fail before training when Ultralytics changes the reserved output directory.
    """
    expected = tmp_path / "expected"
    callback = ultralytics.reserved_run_callback(expected)

    callback(Mock(save_dir=expected))
    with pytest.raises(RuntimeError, match="Trainer directory mismatch"):
        callback(Mock(save_dir=tmp_path / "other"))


def test_configure_privacy_skips_settings_removed_by_ultralytics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Apply compatible privacy defaults when an installed schema omits old keys.
    """
    settings = Mock()
    settings.defaults = {
        "datasets_dir": "datasets",
        "sync": True,
        "wandb": True,
    }
    ultralytics_module = ModuleType("ultralytics")
    ultralytics_module.settings = settings
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_module)

    updates = ultralytics.configure_privacy(config_dir=tmp_path)

    assert updates == {
        "sync": False,
        "wandb": False,
        "datasets_dir": str(ultralytics.DATA_ROOT / "sources/ultralytics"),
    }
    settings.update.assert_called_once_with(updates)


def test_configure_privacy_rejects_unsupported_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Report unsupported caller overrides rather than silently hiding mistakes.
    """
    settings = Mock()
    settings.defaults = {"sync": True}
    ultralytics_module = ModuleType("ultralytics")
    ultralytics_module.settings = settings
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_module)

    with pytest.raises(KeyError, match="unknown_setting"):
        ultralytics.configure_privacy(
            config_dir=tmp_path,
            settings_overrides={"unknown_setting": False},
        )

    settings.update.assert_not_called()


def test_ultralytics_dataset_path_uses_project_source_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Build the default local dataset path from the project root.
    """
    monkeypatch.setattr(ultralytics, "DATA_ROOT", tmp_path / "datasets")

    path = ultralytics.dataset_path("coco128")

    assert path == tmp_path / "datasets" / "sources" / "ultralytics" / "coco128"


def test_download_dataset_writes_local_yaml_and_extracts_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Stage an Ultralytics dataset locally without keeping the global path config.
    """
    calls: list[tuple[Path, str]] = []
    dataset_download_url = "https://example.test/coco128.zip"

    def fake_cache_download(cache_path: Path | str, url: str) -> Path:
        """
        Create local fixtures for the official YAML and dataset archive.
        """
        destination = Path(cache_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        calls.append((destination, url))

        if url == ultralytics.official_dataset_yaml_url("coco128"):
            destination.write_text(
                yaml.safe_dump(
                    {
                        "path": "coco128",
                        "train": "images/train2017",
                        "val": "images/train2017",
                        "names": {0: "person"},
                        "download": dataset_download_url,
                    },
                    sort_keys=False,
                )
            )
        elif url == dataset_download_url:
            with ZipFile(destination, "w") as archive:
                archive.writestr("coco128/images/train2017/image.jpg", "")
                archive.writestr("coco128/labels/train2017/image.txt", "")
        else:
            raise AssertionError(f"Unexpected URL: {url}")

        return destination

    monkeypatch.setattr(ultralytics, "DATA_ROOT", tmp_path / "datasets")
    monkeypatch.setattr(ultralytics, "cache_download", fake_cache_download)

    data_yaml = ultralytics.download("coco128")
    dataset_dir = ultralytics.dataset_path("coco128")
    local_config = yaml.safe_load(data_yaml.read_text())

    assert data_yaml == dataset_dir / "data.yaml"
    assert (dataset_dir / "images" / "train2017" / "image.jpg").exists()
    assert local_config["path"] == str(dataset_dir)
    assert local_config["train"] == "images/train2017"
    assert "download" not in local_config
    assert calls == [
        (
            dataset_dir / "coco128.official.yaml",
            ultralytics.official_dataset_yaml_url("coco128"),
        ),
        (dataset_dir.parent / "coco128.zip", dataset_download_url),
    ]


def test_download_dataset_reuses_existing_local_data_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Return the local data YAML without fetching metadata when it is usable.
    """
    dataset_dir = tmp_path / "datasets" / "sources" / "ultralytics" / "coco128"
    (dataset_dir / "images" / "train2017").mkdir(parents=True)
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train2017",
                "val": "images/train2017",
                "names": {0: "person"},
            },
            sort_keys=False,
        )
    )

    def fake_cache_download(cache_path: Path | str, url: str) -> Path:
        """
        Fail if the reusable local data YAML path does not short-circuit.
        """
        raise AssertionError(f"Unexpected download call for {url} to {cache_path}")

    monkeypatch.setattr(ultralytics, "DATA_ROOT", tmp_path / "datasets")
    monkeypatch.setattr(ultralytics, "cache_download", fake_cache_download)

    assert ultralytics.download("coco128") == data_yaml


def test_download_dataset_regenerates_stale_local_data_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Download metadata and payload when local data YAML points to missing data.
    """
    calls: list[tuple[Path, str]] = []
    dataset_download_url = "https://example.test/coco128.zip"
    dataset_dir = tmp_path / "datasets" / "sources" / "ultralytics" / "coco128"
    dataset_dir.mkdir(parents=True)
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_dir),
                "train": "images/train2017",
                "val": "images/train2017",
                "names": {0: "person"},
            },
            sort_keys=False,
        )
    )

    def fake_cache_download(cache_path: Path | str, url: str) -> Path:
        """
        Create local fixtures for repairing a stale local data YAML.
        """
        destination = Path(cache_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        calls.append((destination, url))

        if url == ultralytics.official_dataset_yaml_url("coco128"):
            destination.write_text(
                yaml.safe_dump(
                    {
                        "path": "coco128",
                        "train": "images/train2017",
                        "val": "images/train2017",
                        "names": {0: "person"},
                        "download": dataset_download_url,
                    },
                    sort_keys=False,
                )
            )
        elif url == dataset_download_url:
            with ZipFile(destination, "w") as archive:
                archive.writestr("coco128/images/train2017/image.jpg", "")
        else:
            raise AssertionError(f"Unexpected URL: {url}")

        return destination

    monkeypatch.setattr(ultralytics, "DATA_ROOT", tmp_path / "datasets")
    monkeypatch.setattr(ultralytics, "cache_download", fake_cache_download)

    assert ultralytics.download("coco128") == data_yaml
    assert (dataset_dir / "images" / "train2017" / "image.jpg").exists()
    assert calls == [
        (
            dataset_dir / "coco128.official.yaml",
            ultralytics.official_dataset_yaml_url("coco128"),
        ),
        (dataset_dir.parent / "coco128.zip", dataset_download_url),
    ]


def test_download_dataset_reuses_existing_dataset_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Avoid downloading the archive again when the expected train split exists.
    """
    calls: list[str] = []
    dataset_dir = tmp_path / "datasets" / "sources" / "ultralytics" / "coco128"
    (dataset_dir / "images" / "train2017").mkdir(parents=True)

    def fake_cache_download(cache_path: Path | str, url: str) -> Path:
        """
        Create only the official YAML fixture for an already-staged dataset.
        """
        destination = Path(cache_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(
                {
                    "path": "coco128",
                    "train": "images/train2017",
                    "val": "images/train2017",
                    "names": {0: "person"},
                    "download": "https://example.test/coco128.zip",
                },
                sort_keys=False,
            )
        )
        calls.append(url)
        return destination

    monkeypatch.setattr(ultralytics, "DATA_ROOT", tmp_path / "datasets")
    monkeypatch.setattr(ultralytics, "cache_download", fake_cache_download)

    ultralytics.download("coco128")

    assert calls == [ultralytics.official_dataset_yaml_url("coco128")]
