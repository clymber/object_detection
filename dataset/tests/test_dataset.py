"""
Check versioned dataset paths and real COCO-to-YOLO conversion.
"""

import builtins
import errno
from pathlib import Path

import pytest
from detection_common.utils.digest import image_content_digest
from detection_common.utils.json_io import read_json, write_json
from PIL import Image

from dataset_builder import Dataset

SPLITS = ("train", "val", "test")


def test_summary_supports_arbitrary_versions_and_category_ids(tmp_path, monkeypatch):
    """Keep inventory generic, multiclass, read-only, and independent of Datumaro."""
    root = write_coco(tmp_path / "widgets-experimental")
    for split in SPLITS:
        path = root / "coco/annotations" / f"instances_{split}.json"
        data = read_json(path)
        data["categories"] = [{"id": 7, "name": "widget"}, {"id": 19, "name": "gizmo"}]
        for annotation in data["annotations"]:
            annotation["category_id"] = {1: 7, 2: 19}[annotation["category_id"]]
        write_json(path, data)
    before = {
        path: path.read_bytes() for path in (root / "coco/annotations").glob("*.json")
    }
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "datumaro" or name.startswith("datumaro."):
            raise AssertionError("Inventory imported a format converter")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    summary = Dataset(root).summary()
    assert summary["dataset_version"] == "widgets-experimental"
    assert summary["category_mapping"] == [
        {"source_id": 7, "model_index": 0, "name": "widget"},
        {"source_id": 19, "model_index": 1, "name": "gizmo"},
    ]
    assert summary["splits"] == {
        split: {"images": 2, "annotations": 2, "backgrounds": 1} for split in SPLITS
    }
    assert not (root / "yolo").exists()
    assert all(path.read_bytes() == content for path, content in before.items())


def test_summary_accepts_empty_targets_and_zero_category_id(tmp_path):
    """Report all-background splits without assuming a single COCO category ID."""
    root = write_coco(tmp_path / "other-data")
    for split in SPLITS:
        path = root / "coco/annotations" / f"instances_{split}.json"
        data = read_json(path)
        data["annotations"] = []
        data["categories"] = [{"id": 0, "name": "anything"}]
        write_json(path, data)
    summary = Dataset(root).summary()
    assert summary["category_mapping"] == [
        {"source_id": 0, "model_index": 0, "name": "anything"}
    ]
    assert all(
        row == {"images": 2, "annotations": 0, "backgrounds": 2}
        for row in summary["splits"].values()
    )


def write_coco(root: Path) -> Path:
    """
    Create positive and background images with digest filenames in every split.
    """
    for split_index, split in enumerate(SPLITS):
        images = []
        for index in range(2):
            directory = root / "coco" / "images" / split
            directory.mkdir(parents=True, exist_ok=True)
            image = directory / "temporary.png"
            Image.new("RGB", (20, 10), (split_index * 40, index * 60, 30)).save(image)
            destination = directory / f"{image_content_digest(image)}.png"
            image.rename(destination)
            images.append({
                "id": index + 1, "file_name": destination.name,
                "width": 20, "height": 10,
            })
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 1,
             "bbox": [2, 1, 8, 4], "area": 32, "iscrowd": 0, "segmentation": []},
            {"id": 2, "image_id": 1, "category_id": 2,
             "bbox": [6, 2, 4, 3], "area": 12, "iscrowd": 0, "segmentation": []},
        ]
        path = root / "coco" / "annotations" / f"instances_{split}.json"
        path.parent.mkdir(exist_ok=True)
        write_json(path, {
            "images": images, "annotations": annotations,
            "categories": [
                {"id": 1, "name": "football"}, {"id": 2, "name": "person"},
            ],
        })
    return root


def test_existing_paths_do_not_import_datumaro(tmp_path, monkeypatch):
    """
    Resolve existing paths without importing the optional conversion package.
    """
    root = write_coco(tmp_path / "football-v0.1")
    yolo = root / "yolo"
    yolo.mkdir()
    (yolo / "data.yaml").write_text("names: {0: football}\n", encoding="utf-8")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        """
        Fail if a path lookup attempts to load Datumaro.
        """
        if name == "datumaro" or name.startswith("datumaro."):
            raise AssertionError("Path lookup imported Datumaro")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    dataset = Dataset(str(root))
    assert isinstance(dataset.coco, Path)
    assert isinstance(dataset.yolo, Path)
    assert dataset.coco == root / "coco"
    assert dataset.yolo == yolo


def test_missing_coco_does_not_create_directories(tmp_path):
    """
    Keep construction passive and explain a missing canonical input.
    """
    root = tmp_path / "absent"
    dataset = Dataset(root)
    assert not root.exists()
    with pytest.raises(FileNotFoundError, match="COCO dataset not found"):
        _ = dataset.coco
    with pytest.raises(FileNotFoundError, match="COCO dataset not found"):
        _ = dataset.yolo
    assert not root.exists()


@pytest.mark.parametrize("copy_fallback", [False, True])
def test_conversion_preserves_content_and_reuses_export(
    tmp_path, monkeypatch, copy_fallback,
):
    """
    Preserve classes, boxes, splits, backgrounds, filenames, and source bytes.
    """
    pytest.importorskip("datumaro")
    yaml = pytest.importorskip("yaml")
    from dataset_builder import datumaro

    if copy_fallback:
        def unavailable_link(*args, **kwargs):
            """
            Simulate a source and destination on different filesystems.
            """
            raise OSError(errno.EXDEV, "Cross-device link")

        monkeypatch.setattr(datumaro.os, "link", unavailable_link)

    root = write_coco(tmp_path / "football-v0.1")
    dataset = Dataset(root)
    assert not (root / "yolo").exists()
    yolo = dataset.yolo
    config = yaml.safe_load((yolo / "data.yaml").read_text(encoding="utf-8"))
    assert config["names"] == {0: "football", 1: "person"}
    for split in SPLITS:
        coco = read_json(dataset.coco / "annotations" / f"instances_{split}.json")
        filenames = {entry["file_name"] for entry in coco["images"]}
        assert {p.name for p in (yolo / "images" / split).iterdir()} == filenames
        image_list = yolo / config[split]
        assert image_list.is_file()
        assert {Path(line).name for line in image_list.read_text().splitlines()} == (
            filenames
        )
        for entry in coco["images"]:
            source = dataset.coco / "images" / split / entry["file_name"]
            exported = yolo / "images" / split / entry["file_name"]
            assert exported.read_bytes() == source.read_bytes()
            assert exported.samefile(source) is not copy_fallback
            label = yolo / "labels" / split / f"{source.stem}.txt"
            rows = [line.split() for line in label.read_text().splitlines()]
            rows.sort(key=lambda row: int(row[0]))
            if entry["id"] == 1:
                assert [row[0] for row in rows] == ["0", "1"]
                assert [float(v) for v in rows[0][1:]] == pytest.approx(
                    [0.3, 0.3, 0.4, 0.4]
                )
                assert [float(v) for v in rows[1][1:]] == pytest.approx(
                    [0.4, 0.35, 0.2, 0.3]
                )
            else:
                assert rows == []
        assert len(coco["annotations"]) == 2

    def unexpected_conversion(*args):
        """
        Reject conversion when a complete export already exists.
        """
        raise AssertionError("Existing export was converted again")

    monkeypatch.setattr(Dataset, "_export_yolo", staticmethod(unexpected_conversion))
    assert Dataset(root).yolo == yolo
    assert not list(root.glob(".yolo-*"))


def test_failed_conversion_cleans_up_and_can_retry(tmp_path, monkeypatch):
    """
    Keep a failed export invisible and allow a later conversion to succeed.
    """
    root = write_coco(tmp_path / "football-v0.1")
    original = Dataset._export_yolo

    def interrupted_conversion(coco, destination):
        """
        Fail after creating partial files in the temporary directory.
        """
        destination.mkdir()
        (destination / "data.yaml").write_text("partial", encoding="utf-8")
        raise RuntimeError("Interrupted export")

    monkeypatch.setattr(Dataset, "_export_yolo", staticmethod(interrupted_conversion))
    with pytest.raises(RuntimeError, match="Interrupted export"):
        _ = Dataset(root).yolo
    assert not (root / "yolo").exists()
    assert not list(root.glob(".yolo-*"))
    monkeypatch.setattr(Dataset, "_export_yolo", staticmethod(original))
    pytest.importorskip("datumaro")
    assert (Dataset(root).yolo / "data.yaml").is_file()


def test_existing_incomplete_export_is_reported(tmp_path):
    """
    Leave incomplete existing exports untouched and explain the missing YAML.
    """
    root = write_coco(tmp_path / "football-v0.1")
    (root / "yolo").mkdir()
    with pytest.raises(ValueError, match="Incomplete YOLO export"):
        _ = Dataset(root).yolo
    assert not list((root / "yolo").iterdir())
