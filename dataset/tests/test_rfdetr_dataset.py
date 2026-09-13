"""
Exercise immutable RF-DETR dataset preparation without detector dependencies.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from dataset_builder.rfdetr import prepare_coco_dataset, summarize_dataset


def _fixture(source: Path, *, extra_images: bool = False) -> Path:
    """
    Create distinct positive/negative images with nested names and category 7.
    """
    (source / "annotations").mkdir(parents=True)
    for split_index, split in enumerate(("train", "val", "test")):
        images = []
        for index, name in enumerate(("nested/ball.png", "background.png")):
            image_path = source / "images" / split / name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), (split_index * 40, index * 80, 30)).save(
                image_path
            )
            images.append(
                {
                    "id": split_index * 10 + index + 1,
                    "file_name": name,
                    "width": 32,
                    "height": 24,
                }
            )
        if extra_images:
            Image.new("RGB", (32, 24), (split_index * 40, 180, 70)).save(
                source / "images" / split / "extra.png"
            )
            images.append(
                {
                    "id": split_index * 10 + 3,
                    "file_name": "extra.png",
                    "width": 32,
                    "height": 24,
                }
            )
        coco = {
            "info": {"description": "unchanged fixture metadata"},
            "licenses": [],
            "categories": [{"id": 7, "name": "basketball", "supercategory": ""}],
            "images": images,
            "annotations": [
                {
                    "id": 100,
                    "image_id": images[0]["id"],
                    "category_id": 7,
                    "bbox": [2.5, 3.5, 10.0, 12.0],
                    "area": 120.0,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            ],
        }
        (source / "annotations" / f"instances_{split}.json").write_text(
            json.dumps(coco), encoding="utf-8"
        )
    return source


def _edit_annotation(source: Path, split: str, edit) -> None:
    """
    Apply a test mutation to an original annotation document.
    """
    path = source / "annotations" / f"instances_{split}.json"
    coco = json.loads(path.read_text())
    edit(coco)
    path.write_text(json.dumps(coco))


@pytest.mark.parametrize("link_images", [True, False])
def test_preparation_preserves_negatives_ids_paths_and_boxes(
    tmp_path: Path, link_images: bool
) -> None:
    """
    Round-trip source identity and geometry with reversible category remapping.
    """
    source = _fixture(tmp_path / "source")
    destination = tmp_path / "derived"
    original = (source / "annotations" / "instances_val.json").read_bytes()
    manifest = prepare_coco_dataset(
        source, destination, category_id=1, link_images=link_images
    )

    coco = json.loads((destination / "valid" / "_annotations.coco.json").read_text())
    assert coco["info"] == {"description": "unchanged fixture metadata"}
    assert [image["id"] for image in coco["images"]] == [11, 12]
    assert coco["annotations"][0] == {
        "id": 100,
        "image_id": 11,
        "category_id": 1,
        "bbox": [2.5, 3.5, 10.0, 12.0],
        "area": 120.0,
        "iscrowd": 0,
        "segmentation": [],
    }
    assert coco["categories"][0]["id"] == 1
    assert manifest["category_mapping"]["source_to_training"] == {"7": 1}
    assert manifest["category_mapping"]["training_to_source"] == {"1": 7}
    assert manifest["splits"]["val"]["negative_images"] == 1
    assert (destination / "valid" / "background.png").is_file()
    image = destination / "valid" / "nested" / "ball.png"
    assert image.is_symlink() is link_images
    assert image.read_bytes() == (source / "images/val/nested/ball.png").read_bytes()
    assert (source / "annotations" / "instances_val.json").read_bytes() == original
    assert (
        prepare_coco_dataset(
            source, destination, category_id=1, link_images=link_images
        )
        == manifest
    )
    assert summarize_dataset(manifest)[1] == {
        "split": "val",
        "loader_split": "valid",
        "images": 2,
        "annotations": 1,
        "positive_images": 1,
        "negative_images": 1,
        "mode": "full",
    }


def test_category_preservation_and_smoke_subset(tmp_path: Path) -> None:
    """
    A deterministic smoke subset preserves labels and examples from both types.
    """
    source = _fixture(tmp_path / "source", extra_images=True)
    manifest = prepare_coco_dataset(
        source,
        tmp_path / "smoke",
        category_id=None,
        expected_counts={"train": 3, "val": 3, "test": 3},
        smoke_limits={"train": 2, "val": 2, "test": 2},
    )
    assert manifest["mode"] == "smoke"
    assert manifest["category_mapping"]["training_to_source"] == {"7": 7}
    for split in ("train", "val", "test"):
        assert manifest["splits"][split]["images"] == 2
        assert manifest["splits"][split]["negative_images"] == 1
        assert manifest["splits"][split]["source_counts"]["images"] == 3
    with pytest.raises(ValueError, match="at least 2"):
        prepare_coco_dataset(
            source,
            tmp_path / "bad-smoke",
            smoke_limits={"train": 1, "val": 1, "test": 1},
        )


@pytest.mark.parametrize("change", ["annotation", "source_image", "derived_image"])
def test_stale_cache_is_rejected_without_overwrite(tmp_path: Path, change: str) -> None:
    """
    Changing source metadata or either image copy invalidates the cache.
    """
    source = _fixture(tmp_path / "source")
    destination = tmp_path / "derived"
    prepare_coco_dataset(source, destination, link_images=False)
    saved_manifest = (destination / "manifest.json").read_bytes()
    if change == "annotation":
        _edit_annotation(source, "train", lambda coco: coco["info"].update(extra=True))
    else:
        image_path = (
            source / "images/train/background.png"
            if change == "source_image"
            else destination / "train/background.png"
        )
        Image.new("RGB", (32, 24), (1, 2, 3)).save(image_path)
    with pytest.raises(ValueError, match="Stale derived dataset"):
        prepare_coco_dataset(source, destination, link_images=False)
    assert (destination / "manifest.json").read_bytes() == saved_manifest


def test_refuse_unknown_destination_and_changed_derived_annotation(
    tmp_path: Path,
) -> None:
    """
    Refuse existing unmanaged directories and corrupt managed annotations.
    """
    source = _fixture(tmp_path / "source")
    destination = tmp_path / "derived"
    destination.mkdir()
    (destination / "keep.txt").write_text("valuable")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_coco_dataset(source, destination)
    assert (destination / "keep.txt").read_text() == "valuable"
    prepared = tmp_path / "prepared"
    prepare_coco_dataset(source, prepared)
    (prepared / "test/_annotations.coco.json").write_text("{}")
    with pytest.raises(ValueError, match="test annotations changed"):
        prepare_coco_dataset(source, prepared)


def test_cross_split_duplicates_require_acknowledgement(tmp_path: Path) -> None:
    """
    Report exact duplicate contents without deleting or relocating source examples.
    """
    source = _fixture(tmp_path / "source")
    (source / "images/test/background.png").write_bytes(
        (source / "images/train/background.png").read_bytes()
    )
    with pytest.raises(ValueError, match="including 1 across splits"):
        prepare_coco_dataset(source, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()
    with pytest.warns(UserWarning, match="including 1 across splits"):
        manifest = prepare_coco_dataset(
            source, tmp_path / "derived", allow_cross_split_duplicates=True
        )
    assert manifest["duplicate_image_groups"][0]["cross_split"] is True
    assert manifest["splits"]["test"]["images"] == 2


def test_within_split_duplicates_warn_and_preserve_membership(tmp_path: Path) -> None:
    """
    Existing repeated training images need no override that also permits leakage.
    """
    source = _fixture(tmp_path / "source", extra_images=True)
    (source / "images/train/extra.png").write_bytes(
        (source / "images/train/background.png").read_bytes()
    )
    with pytest.warns(UserWarning, match="including 0 across splits"):
        manifest = prepare_coco_dataset(source, tmp_path / "derived")
    assert manifest["splits"]["train"]["images"] == 3
    assert manifest["duplicate_image_groups"][0]["cross_split"] is False
    assert manifest["allow_cross_split_duplicates"] is False


def test_symlink_cannot_place_derived_dataset_inside_source(tmp_path: Path) -> None:
    """
    Resolving parent links keeps source-dataset protection effective for aliases.
    """
    source = _fixture(tmp_path / "source")
    alias = tmp_path / "source-alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the source"):
        prepare_coco_dataset(source, alias / "derived")
    assert not (source / "derived").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("bbox", [0, 0, 0, 1], "invalid/out-of-image"),
        ("bbox", [30, 0, 8, 8], "invalid/out-of-image"),
        ("bbox", [0, 0, float("nan"), 8], "nonfinite"),
        ("image_id", 999, "unknown image"),
        ("category_id", 99, "unknown category"),
    ],
)
def test_invalid_annotations_are_rejected(
    tmp_path: Path, field: str, value, message: str
) -> None:
    """
    Catch broken annotation geometry and references before writing derived data.
    """
    source = _fixture(tmp_path / "source")
    _edit_annotation(
        source, "train", lambda coco: coco["annotations"][0].update({field: value})
    )
    with pytest.raises(ValueError, match=message):
        prepare_coco_dataset(source, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_missing_test_and_image_paths_fail_explicitly(tmp_path: Path) -> None:
    """
    The real held-out test annotations and safe source image paths are mandatory.
    """
    source = _fixture(tmp_path / "source")
    path = source / "annotations/instances_test.json"
    saved = path.read_bytes()
    path.unlink()
    with pytest.raises(FileNotFoundError, match="instances_test.json"):
        prepare_coco_dataset(source, tmp_path / "derived")
    path.write_bytes(saved)
    _edit_annotation(
        source, "train", lambda coco: coco["images"][0].update(file_name="../ball.png")
    )
    with pytest.raises(ValueError, match="inside its split"):
        prepare_coco_dataset(source, tmp_path / "derived")


def test_expected_source_counts_and_complete_smoke_limits(tmp_path: Path) -> None:
    """
    Catch incorrect frozen dataset selection or an accidentally unbounded smoke run.
    """
    source = _fixture(tmp_path / "source")
    with pytest.raises(ValueError, match="expected 3 source images"):
        prepare_coco_dataset(
            source,
            tmp_path / "derived",
            expected_counts={"train": 3, "val": 2, "test": 2},
        )
    with pytest.raises(ValueError, match="explicitly bound"):
        prepare_coco_dataset(source, tmp_path / "smoke", smoke_limits={"train": 2})


def test_dataset_import_does_not_require_datumaro() -> None:
    """
    A clean interpreter can load the RF-DETR adapter without preprocessing extras.
    """
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = '''
import importlib.abc
import sys

class BlockDatumaro(importlib.abc.MetaPathFinder):
    """
    Make accidental optional-dependency imports fail in this child interpreter.
    """

    def find_spec(self, fullname, path=None, target=None):
        """
        Reject any attempt to import the optional Datumaro package.
        """
        if fullname == "datumaro" or fullname.startswith("datumaro."):
            raise ModuleNotFoundError("Datumaro is intentionally unavailable")
        return None

sys.meta_path.insert(0, BlockDatumaro())
sys.path.insert(0, sys.argv[1])
from dataset_builder.rfdetr import prepare_coco_dataset
assert "datumaro" not in sys.modules
'''
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(source_root)],
        check=True,
        capture_output=True,
        text=True,
    )
