import json
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("datumaro")

from dataset_builder._sport_object import build_sport_object_dataset

SPLITS = ("train", "val", "test")


def write_source(
    root: Path,
    *,
    category_name: str,
    category_id: int,
    color: tuple[int, int, int],
    image_id: int = 1,
    duplicate_from: Path | None = None,
) -> None:
    """
    Write a one-class COCO source with an annotated and background image per split.
    """
    for split_index, split in enumerate(SPLITS):
        image_dir = root / "images" / split
        image_dir.mkdir(parents=True)
        annotated_path = image_dir / f"{category_name}_{split}.png"
        if duplicate_from is None:
            Image.new(
                "RGB",
                (20, 10),
                tuple(channel + split_index for channel in color),
            ).save(annotated_path)
        else:
            annotated_path.write_bytes(duplicate_from.read_bytes())
        background_path = image_dir / f"{category_name}_{split}_background.png"
        Image.new(
            "RGB",
            (20, 10),
            (color[0], 250, split_index),
        ).save(background_path)
        payload = {
            "categories": [{"id": category_id, "name": category_name}],
            "images": [
                {
                    "id": image_id,
                    "file_name": annotated_path.name,
                    "width": 20,
                    "height": 10,
                },
                {
                    "id": image_id + 1,
                    "file_name": background_path.name,
                    "width": 20,
                    "height": 10,
                },
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [2, 1, 8, 4],
                    "area": 32,
                    "iscrowd": 0,
                }
            ],
        }
        annotation_path = root / "annotations" / f"instances_{split}.json"
        annotation_path.parent.mkdir(exist_ok=True)
        annotation_path.write_text(json.dumps(payload), encoding="utf-8")


def build_fixture_dataset(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """
    Create two valid source datasets and their requested artifact paths.
    """
    basketball_root = tmp_path / "coco_basketball"
    football_root = tmp_path / "coco_football"
    write_source(
        basketball_root,
        category_name="basketball",
        category_id=29,
        color=(20, 30, 40),
    )
    write_source(
        football_root,
        category_name="football",
        category_id=7,
        color=(50, 60, 70),
    )
    return (
        basketball_root,
        football_root,
        tmp_path / "coco_sport_object",
        tmp_path / "yolo_sport_object",
        tmp_path / "sport_object_manifest.csv",
    )


def test_build_sport_object_dataset_exports_matching_formats(tmp_path: Path) -> None:
    """
    Compose differing source category and image IDs while retaining backgrounds.
    """
    paths = build_fixture_dataset(tmp_path)
    manifest = build_sport_object_dataset(
        basketball_root=paths[0],
        football_root=paths[1],
        coco_root=paths[2],
        yolo_root=paths[3],
        manifest_path=paths[4],
    )

    assert len(manifest) == 12
    assert manifest.groupby(["source_dataset", "split"]).size().eq(2).all()
    assert manifest["is_background"].sum() == 6
    assert manifest["annotation_count"].sum() == 6
    assert manifest["coco_output_path"].map(Path).map(Path.is_file).all()
    assert manifest["yolo_output_path"].map(Path).map(Path.is_file).all()
    assert paths[4].is_file()
    for split in SPLITS:
        payload = json.loads(
            (paths[2] / "annotations" / f"instances_{split}.json").read_text(
                encoding="utf-8"
            )
        )
        assert [category["name"] for category in payload["categories"]] == [
            "football",
            "basketball",
        ]
        assert len(payload["images"]) == 4
        assert len(payload["annotations"]) == 2
        label_paths = list((paths[3] / "labels" / split).glob("*.txt"))
        assert len(label_paths) == 4
        assert sum(not path.read_text(encoding="utf-8") for path in label_paths) == 2


def test_build_sport_object_dataset_rejects_duplicate_pixels(tmp_path: Path) -> None:
    """
    Stop before export when source images share decoded pixels.
    """
    paths = build_fixture_dataset(tmp_path)
    duplicate_target = paths[0] / "images" / "train" / "basketball_train.png"
    duplicate_path = paths[1] / "images" / "train" / "football_train.png"
    duplicate_path.write_bytes(duplicate_target.read_bytes())

    with pytest.raises(ValueError, match="Duplicate image pixels"):
        build_sport_object_dataset(
            basketball_root=paths[0],
            football_root=paths[1],
            coco_root=paths[2],
            yolo_root=paths[3],
            manifest_path=paths[4],
        )
    assert not paths[2].exists()
    assert not paths[3].exists()


def test_build_sport_object_dataset_rejects_cross_split_duplicates(
    tmp_path: Path,
) -> None:
    """
    Reject repeated pixels even when the records belong to different splits.
    """
    paths = build_fixture_dataset(tmp_path)
    source_path = paths[0] / "images" / "train" / "basketball_train.png"
    duplicate_path = paths[0] / "images" / "val" / "basketball_val.png"
    duplicate_path.write_bytes(source_path.read_bytes())

    with pytest.raises(ValueError, match="Duplicate image pixels"):
        build_sport_object_dataset(
            basketball_root=paths[0],
            football_root=paths[1],
            coco_root=paths[2],
            yolo_root=paths[3],
            manifest_path=paths[4],
        )


def test_build_sport_object_dataset_rejects_missing_inputs(tmp_path: Path) -> None:
    """
    Explain which upstream source must be built when it is missing.
    """
    paths = build_fixture_dataset(tmp_path)

    with pytest.raises(FileNotFoundError, match="football input"):
        build_sport_object_dataset(
            basketball_root=paths[0],
            football_root=tmp_path / "missing_football",
            coco_root=paths[2],
            yolo_root=paths[3],
            manifest_path=paths[4],
        )


def test_build_sport_object_dataset_rejects_invalid_annotation(tmp_path: Path) -> None:
    """
    Stop when a source box extends outside its declared media dimensions.
    """
    paths = build_fixture_dataset(tmp_path)
    annotation_path = paths[0] / "annotations" / "instances_train.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["annotations"][0]["bbox"] = [15, 1, 8, 4]
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Out-of-bounds bounding box"):
        build_sport_object_dataset(
            basketball_root=paths[0],
            football_root=paths[1],
            coco_root=paths[2],
            yolo_root=paths[3],
            manifest_path=paths[4],
        )


def test_build_sport_object_dataset_refuses_existing_destinations(tmp_path: Path) -> None:
    """
    Refuse to produce a partial replacement when either destination exists.
    """
    paths = build_fixture_dataset(tmp_path)
    paths[2].mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_sport_object_dataset(
            basketball_root=paths[0],
            football_root=paths[1],
            coco_root=paths[2],
            yolo_root=paths[3],
            manifest_path=paths[4],
        )