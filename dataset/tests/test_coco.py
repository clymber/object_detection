from pathlib import Path
from typing import Any

from detection_common.utils.json_io import write_json

from dataset_builder import (
    filter_coco_annotation_by_labels,
    summarize_coco_datasets,
)


def coco_fixture() -> dict[str, Any]:
    """
    Return a tiny COCO fixture with mixed-label and background images.
    """
    return {
        "info": {"description": "fixture"},
        "licenses": [{"id": 1, "name": "fixture"}],
        "categories": [
            {"id": 1, "name": "ball", "supercategory": "object"},
            {"id": 2, "name": "hoop", "supercategory": "object"},
            {"id": 3, "name": "person", "supercategory": "object"},
        ],
        "images": [
            {"id": 10, "file_name": "ball.jpg"},
            {"id": 11, "file_name": "ball_person.jpg"},
            {"id": 12, "file_name": "hoop.jpg"},
            {"id": 13, "file_name": "background.jpg"},
        ],
        "annotations": [
            {"id": 100, "image_id": 10, "category_id": 1},
            {"id": 101, "image_id": 11, "category_id": 1},
            {"id": 102, "image_id": 11, "category_id": 3},
            {"id": 103, "image_id": 12, "category_id": 2},
        ],
        "extra": "preserved",
    }


def category_names(coco: dict[str, Any]) -> list[str]:
    """
    Return category names from a COCO annotation dictionary.
    """
    return [category["name"] for category in coco["categories"]]


def image_ids(coco: dict[str, Any]) -> list[int]:
    """
    Return image IDs from a COCO annotation dictionary.
    """
    return [image["id"] for image in coco["images"]]


def annotation_ids(coco: dict[str, Any]) -> list[int]:
    """
    Return annotation IDs from a COCO annotation dictionary.
    """
    return [annotation["id"] for annotation in coco["annotations"]]


def write_coco_split(dataset_path: Path, split: str) -> None:
    """
    Write the COCO fixture as one dataset split.
    """
    annotation_path = dataset_path / split / "_annotations.coco.json"
    annotation_path.parent.mkdir(parents=True)
    write_json(annotation_path, coco_fixture())


def test_filter_coco_annotation_applies_include_and_exclude_labels() -> None:
    """
    Keep included labels while removing images that contain excluded labels.
    """
    trimmed = filter_coco_annotation_by_labels(
        coco_fixture(),
        labels_include={"ball", "hoop"},
        labels_exclude={"person"},
    )

    assert trimmed["info"] == {"description": "fixture"}
    assert trimmed["licenses"] == [{"id": 1, "name": "fixture"}]
    assert trimmed["extra"] == "preserved"
    assert category_names(trimmed) == ["ball", "hoop"]
    assert image_ids(trimmed) == [10, 12]
    assert annotation_ids(trimmed) == [100, 103]


def test_filter_coco_annotation_keeps_background_images_for_exclude_only() -> None:
    """
    Keep background images when only excluded labels are removed.
    """
    trimmed = filter_coco_annotation_by_labels(
        coco_fixture(),
        labels_exclude={"person"},
    )

    assert category_names(trimmed) == ["ball", "hoop"]
    assert image_ids(trimmed) == [10, 12, 13]
    assert annotation_ids(trimmed) == [100, 103]


def test_summarize_coco_datasets_combines_dataset_summaries(tmp_path: Path) -> None:
    """
    Combine summaries from an iterable of COCO dataset paths.
    """
    first_path = tmp_path / "dataset_b" / "v1"
    second_path = tmp_path / "dataset_a" / "v1"
    write_coco_split(first_path, "train")
    write_coco_split(second_path, "valid")

    summary = summarize_coco_datasets(path for path in [first_path, second_path])

    assert summary.to_dict(orient="records") == [
        {
            "dataset_id": "dataset_a",
            "train": 0,
            "val": 4,
            "test": 0,
            "classes": 3,
            "labels": "ball, hoop, person",
            "annotations": 4,
        },
        {
            "dataset_id": "dataset_b",
            "train": 4,
            "val": 0,
            "test": 0,
            "classes": 3,
            "labels": "ball, hoop, person",
            "annotations": 4,
        },
    ]


def test_summarize_coco_datasets_accepts_empty_iterable() -> None:
    """
    Return an empty summary with the standard columns for no datasets.
    """
    summary = summarize_coco_datasets([])

    assert summary.empty
    assert summary.columns.tolist() == [
        "dataset_id",
        "train",
        "val",
        "test",
        "classes",
        "labels",
        "annotations",
    ]
