"""
Private helpers for composing the sport object dataset.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import datumaro as dm
import pandas as pd
from detection_common.utils.image import image_content_digest
from PIL import Image

from .datumaro import ExportFormat, prefer_hardlinked_datumaro_media

SPLITS = ("train", "val", "test")
CLASS_NAMES = ("football", "basketball")


@dataclass(frozen=True)
class SourceImage:
    """
    Describe one validated image from a source COCO dataset.
    """

    source_dataset: str
    split: str
    image_id: int
    source_path: Path
    pixel_sha256: str
    annotations: tuple[dict, ...]


def _annotation_path(dataset_root: Path, split: str) -> Path:
    """
    Return the conventional COCO annotation path for a split.
    """
    return dataset_root / "annotations" / f"instances_{split}.json"


def _load_source(
    dataset_root: Path,
    *,
    source_dataset: str,
    expected_category: str,
) -> list[SourceImage]:
    """
    Load and validate one single-class COCO source dataset.
    """
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Missing {source_dataset} input at {dataset_root}. "
            "Run its upstream dataset builder first."
        )

    records: list[SourceImage] = []
    for split in SPLITS:
        annotation_path = _annotation_path(dataset_root, split)
        image_dir = dataset_root / "images" / split
        if not annotation_path.is_file() or not image_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {source_dataset} {split} split. "
                "Run its upstream dataset builder first."
            )
        with annotation_path.open(encoding="utf-8") as file:
            coco = json.load(file)

        categories = coco.get("categories", [])
        category_ids = {
            category.get("id")
            for category in categories
            if category.get("name") == expected_category
        }
        category_names = {category.get("name") for category in categories}
        if category_names != {expected_category} or len(category_ids) != 1:
            raise ValueError(
                f"{source_dataset} {split} categories must be exactly "
                f"{{{expected_category!r}}}, found {sorted(category_names)!r}"
            )
        category_id = next(iter(category_ids))

        images = coco.get("images", [])
        image_ids = [image.get("id") for image in images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"{source_dataset} {split} has duplicate image IDs")
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            image_id = annotation.get("image_id")
            if image_id not in image_ids or annotation.get("category_id") != category_id:
                raise ValueError(
                    f"{source_dataset} {split} has an invalid annotation reference"
                )
            annotations_by_image[image_id].append(annotation)

        for image in images:
            image_id = image["id"]
            source_path = image_dir / image["file_name"]
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Missing {source_dataset} {split} image: {source_path}"
                )
            with Image.open(source_path) as media:
                width, height = media.size
            if (image.get("width"), image.get("height")) != (width, height):
                raise ValueError(
                    f"{source_dataset} {split} dimensions disagree for {source_path}"
                )
            for annotation in annotations_by_image[image_id]:
                bbox = annotation.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Invalid bounding box for {source_path}")
                x, y, box_width, box_height = map(float, bbox)
                if (
                    x < 0
                    or y < 0
                    or box_width <= 0
                    or box_height <= 0
                    or x + box_width > width
                    or y + box_height > height
                ):
                    raise ValueError(f"Out-of-bounds bounding box for {source_path}")
            records.append(
                SourceImage(
                    source_dataset=source_dataset,
                    split=split,
                    image_id=image_id,
                    source_path=source_path.resolve(),
                    pixel_sha256=image_content_digest(source_path),
                    annotations=tuple(annotations_by_image[image_id]),
                )
            )
    return records


def _reject_duplicates(records: list[SourceImage]) -> None:
    """
    Reject exact pixel duplicates across either source and every split.
    """
    by_digest: dict[str, list[SourceImage]] = defaultdict(list)
    for record in records:
        by_digest[record.pixel_sha256].append(record)
    duplicates = [group for group in by_digest.values() if len(group) > 1]
    if duplicates:
        details = "; ".join(
            ", ".join(
                f"{record.source_dataset}/{record.split}/{record.image_id}"
                for record in group
            )
            for group in duplicates
        )
        raise ValueError(f"Duplicate image pixels detected: {details}")


def _validate_outputs(
    records: list[SourceImage],
    coco_root: Path,
    yolo_root: Path,
) -> None:
    """
    Validate exported split membership, classes, and labels.
    """
    expected_by_split = {
        split: [record for record in records if record.split == split]
        for split in SPLITS
    }
    label_by_source = {"football": 0, "basketball": 1}
    for split, expected in expected_by_split.items():
        if not expected:
            raise ValueError(f"The {split} split is empty")
        expected_classes = {
            record.source_dataset for record in expected if record.annotations
        }
        if expected_classes != set(CLASS_NAMES):
            raise ValueError(f"The {split} split does not contain both classes")

        with _annotation_path(coco_root, split).open(encoding="utf-8") as file:
            coco = json.load(file)
        if [category["name"] for category in coco["categories"]] != list(CLASS_NAMES):
            raise ValueError("Exported COCO category order is incorrect")
        image_by_id = {image["id"]: image for image in coco["images"]}
        category_by_id = {
            category["id"]: category["name"] for category in coco["categories"]
        }
        if not all(
            annotation["image_id"] in image_by_id
            and annotation["category_id"] in category_by_id
            for annotation in coco["annotations"]
        ):
            raise ValueError("Exported COCO references are invalid")

        expected_stems = {record.pixel_sha256 for record in expected}
        coco_stems = {
            Path(image["file_name"]).stem for image in image_by_id.values()
        }
        if coco_stems != expected_stems:
            raise ValueError(f"Exported {split} COCO image files disagree")
        expected_by_stem = {record.pixel_sha256: record for record in expected}
        coco_boxes: dict[str, Counter] = defaultdict(Counter)
        for annotation in coco["annotations"]:
            stem = Path(image_by_id[annotation["image_id"]]["file_name"]).stem
            coco_boxes[stem][
                (
                    category_by_id[annotation["category_id"]],
                    *(float(value) for value in annotation["bbox"]),
                )
            ] += 1
        for stem, record in expected_by_stem.items():
            expected_boxes = Counter(
                (record.source_dataset, *(float(value) for value in annotation["bbox"]))
                for annotation in record.annotations
            )
            if coco_boxes[stem] != expected_boxes:
                raise ValueError(f"Exported COCO boxes disagree for {stem}")

        image_paths = list((yolo_root / "images" / split).iterdir())
        label_paths = list((yolo_root / "labels" / split).glob("*.txt"))
        if {path.stem for path in image_paths} != expected_stems or {
            path.stem for path in label_paths
        } != expected_stems:
            raise ValueError(f"Exported {split} image and label files disagree")
        for label_path in label_paths:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) != 5 or fields[0] not in {"0", "1"}:
                    raise ValueError(f"Invalid YOLO label in {label_path}")
                if not all(0 <= float(value) <= 1 for value in fields[1:]):
                    raise ValueError(f"Out-of-range YOLO label in {label_path}")
            record = expected_by_stem[label_path.stem]
            with Image.open(record.source_path) as media:
                width, height = media.size
            expected_boxes = sorted(
                (
                    label_by_source[record.source_dataset],
                    (float(annotation["bbox"][0]) + float(annotation["bbox"][2]) / 2)
                    / width,
                    (float(annotation["bbox"][1]) + float(annotation["bbox"][3]) / 2)
                    / height,
                    float(annotation["bbox"][2]) / width,
                    float(annotation["bbox"][3]) / height,
                )
                for annotation in record.annotations
            )
            actual_boxes = sorted(
                (int(fields[0]), *(float(value) for value in fields[1:]))
                for fields in (
                    line.split()
                    for line in label_path.read_text(encoding="utf-8").splitlines()
                )
            )
            if len(actual_boxes) != len(expected_boxes) or any(
                actual[0] != expected_box[0]
                or any(
                    abs(actual_value - expected_value) > 1e-6
                    for actual_value, expected_value in zip(
                        actual[1:], expected_box[1:], strict=True
                    )
                )
                for actual, expected_box in zip(actual_boxes, expected_boxes, strict=True)
            ):
                raise ValueError(f"Exported YOLO boxes disagree for {label_path.stem}")
    data_yaml = (yolo_root / "data.yaml").read_text(encoding="utf-8")
    if "football" not in data_yaml or "basketball" not in data_yaml:
        raise ValueError("Exported YOLO data.yaml is missing class names")


def build_sport_object_dataset(
    *,
    basketball_root: Path,
    football_root: Path,
    coco_root: Path,
    yolo_root: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    """
    Compose validated basketball and football COCO exports into two formats.
    """
    existing = [path for path in (coco_root, yolo_root) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite output(s): {existing}")

    records = _load_source(
        football_root,
        source_dataset="football",
        expected_category="football",
    ) + _load_source(
        basketball_root,
        source_dataset="basketball",
        expected_category="basketball",
    )
    _reject_duplicates(records)

    label_by_source = {"football": 0, "basketball": 1}
    items = [
        dm.DatasetItem(
            id=record.pixel_sha256,
            subset=record.split,
            media=dm.Image.from_file(str(record.source_path)),
            annotations=[
                dm.Bbox(
                    *annotation["bbox"],
                    label=label_by_source[record.source_dataset],
                    attributes={"is_crowd": bool(annotation.get("iscrowd", 0))},
                )
                for annotation in record.annotations
            ],
        )
        for record in records
    ]
    dataset = dm.Dataset.from_iterable(items, categories=list(CLASS_NAMES))
    with prefer_hardlinked_datumaro_media():
        dataset.export(
            str(coco_root),
            format=ExportFormat.COCO_INSTANCES,
            save_media=True,
            reindex=True,
        )
        dataset.export(
            str(yolo_root),
            format=ExportFormat.YOLO_ULTRALYTICS,
            save_media=True,
        )

    _validate_outputs(records, coco_root, yolo_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        {
            "source_dataset": record.source_dataset,
            "split": record.split,
            "source_image_id": record.image_id,
            "source_path": str(record.source_path),
            "pixel_sha256": record.pixel_sha256,
            "annotation_count": len(record.annotations),
            "is_background": not record.annotations,
            "coco_output_path": str(
                coco_root
                / "images"
                / record.split
                / f"{record.pixel_sha256}{record.source_path.suffix}"
            ),
            "yolo_output_path": str(
                yolo_root
                / "images"
                / record.split
                / f"{record.pixel_sha256}{record.source_path.suffix}"
            ),
        }
        for record in records
    )
    manifest.to_csv(manifest_path, index=False)
    return manifest