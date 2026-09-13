"""
Utilities for configuring and working with Ultralytics.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol
from zipfile import ZipFile

import yaml

from detection_common.utils.urlhelper import cache_download

from .config import DATA_ROOT, OUTPUT_ROOT

ULTRALYTICS_PRIVACY_SETTINGS: dict[str, Any] = {
    "sync": False,
    "hub": False,
    "clearml": False,
    "comet": False,
    "dvc": False,
    "mlflow": False,
    "neptune": False,
    "raytune": False,
    "tensorboard": False,
    "wandb": False,
    "vscode_msg": False,
    "openvino_msg": False,
    "datasets_dir": str(DATA_ROOT / "sources" / "ultralytics"),
}

ULTRALYTICS_DATASET_YAML_BASE_URL = (
    "https://raw.githubusercontent.com/ultralytics/ultralytics/main/"
    "ultralytics/cfg/datasets"
)


class TrainingResult(Protocol):
    """
    Runtime attributes exposed by Ultralytics training metrics.
    """

    save_dir: Path


def official_dataset_yaml_url(dataset_name: str) -> str:
    """
    Return the official Ultralytics dataset YAML URL for a dataset name.
    """
    yaml_name = dataset_name
    if not yaml_name.endswith((".yaml", ".yml")):
        yaml_name = f"{yaml_name}.yaml"

    return f"{ULTRALYTICS_DATASET_YAML_BASE_URL}/{yaml_name}"


def dataset_path(
    dataset_name: str,
    *,
    source_root: Path | str | None = None,
) -> Path:
    """
    Return the default local source path for an Ultralytics dataset.
    """
    local_name = dataset_name.removesuffix(".yaml").removesuffix(".yml")
    if source_root is None:
        source_root = DATA_ROOT / "sources" / "ultralytics"

    return Path(source_root) / local_name


def _first_split_path(dataset_dir: Path, split_value: object) -> Path:
    """
    Return the first filesystem path referenced by a dataset split value.
    """
    if isinstance(split_value, str):
        return dataset_dir / split_value

    if isinstance(split_value, list) and split_value:
        return dataset_dir / str(split_value[0])

    raise ValueError("Ultralytics dataset YAML must define a non-empty train split.")


def _extract_zip(archive_path: Path, destination: Path) -> None:
    """
    Extract a zip archive while rejecting paths outside the destination.
    """
    destination = destination.resolve()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe archive member path: {member.filename}")

        archive.extractall(destination)


def _check_data_yaml_ok(data_yaml: Path) -> bool:
    """
    Return whether a local data.yaml points to an existing train split.
    """
    try:
        with data_yaml.open() as yaml_file:
            dataset_config = yaml.safe_load(yaml_file)

        if not isinstance(dataset_config, dict):
            return False

        dataset_dir = Path(dataset_config.get("path", data_yaml.parent))
        if not dataset_dir.is_absolute():
            dataset_dir = data_yaml.parent / dataset_dir

        return _first_split_path(dataset_dir, dataset_config.get("train")).exists()
    except (FileNotFoundError, TypeError, ValueError, yaml.YAMLError):
        return False


def download(
    dataset_name: str,
    *,
    yaml_url: str | None = None,
    source_root: Path | str | None = None,
) -> Path:
    """
    Download an Ultralytics dataset into the project source dataset directory.

    Returns the local `data.yaml` path configured to read from the staged source
    directory instead of Ultralytics' global datasets directory.
    """
    dataset_dir = dataset_path(dataset_name, source_root=source_root)
    data_yaml = dataset_dir / "data.yaml"

    if _check_data_yaml_ok(data_yaml):
        return data_yaml

    if yaml_url is None:
        yaml_url = official_dataset_yaml_url(dataset_name)

    yaml_name = dataset_name.removesuffix(".yaml").removesuffix(".yml")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    official_yaml = cache_download(dataset_dir / f"{yaml_name}.official.yaml", yaml_url)
    with official_yaml.open() as yaml_file:
        dataset_config: dict[str, Any] = yaml.safe_load(yaml_file)

    train_path = _first_split_path(dataset_dir, dataset_config.get("train"))
    if not train_path.exists():
        download_url = dataset_config.get("download")
        if not isinstance(download_url, str):
            raise ValueError("Ultralytics dataset YAML must define a download URL.")

        archive_path = cache_download(
            dataset_dir.parent / f"{yaml_name}.zip",
            download_url,
        )
        _extract_zip(archive_path, dataset_dir.parent)

    dataset_config["path"] = str(dataset_dir)
    dataset_config.pop("download", None)

    with data_yaml.open("w") as yaml_file:
        yaml.safe_dump(dataset_config, yaml_file, sort_keys=False)

    return data_yaml


def configure_privacy(
    *,
    offline: bool = True,
    config_dir: Path | str | None = OUTPUT_ROOT / "ultralytics",
    settings_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Disable Ultralytics telemetry and optional experiment integrations.

    Call this before importing ``YOLO`` so Ultralytics initializes with the
    project privacy settings from the start.
    """
    if offline:
        os.environ.setdefault("YOLO_OFFLINE", "true")

    if config_dir is not None:
        config_path = Path(config_dir)
        config_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_path))

    from ultralytics import settings

    supported_keys = set(getattr(settings, "defaults", settings))
    updates = {
        key: value
        for key, value in ULTRALYTICS_PRIVACY_SETTINGS.items()
        if key in supported_keys
    }

    if settings_overrides is not None:
        unsupported_keys = set(settings_overrides) - supported_keys
        if unsupported_keys:
            unsupported_text = ", ".join(sorted(unsupported_keys))
            raise KeyError(f"Unsupported Ultralytics settings: {unsupported_text}")
        updates.update(settings_overrides)

    settings.update(updates)
    return updates
