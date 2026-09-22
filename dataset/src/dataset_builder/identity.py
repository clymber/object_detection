"""
Fingerprint COCO sources and validate framework-neutral dataset loader layouts.

The canonical source identity deliberately excludes filesystem roots and source
metadata. It records only split membership, COCO detection content, category
mapping, relative image names, and image bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import DATA_ROOT
from .rfdetr import (
    SPLIT_NAMES,
    _file_hash,
    _json_bytes,
    _read_split,
    _smoke_subset,
    _validate_derived,
    prepare_coco_dataset,
)

IDENTITY_FILE_NAME = "dataset_identity.json"
SMOKE_LIMITS = {"train": 16, "val": 8, "test": 8}


def _fingerprint(payload: Mapping[str, Any]) -> str:
    """
    Return the SHA-256 fingerprint of a normalized JSON-compatible payload.
    """
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _canonical_categories(coco: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Return categories in a deterministic form meaningful to detection loaders.
    """
    return [
        {
            "id": category["id"],
            "name": category["name"],
            "supercategory": category.get("supercategory", ""),
        }
        for category in sorted(coco["categories"], key=lambda item: item["id"])
    ]


def _canonical_annotations(coco: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Return all COCO annotations independent of document ordering.
    """
    return sorted(deepcopy(coco["annotations"]), key=lambda item: item["id"])


def _canonical_split(coco: Mapping[str, Any], identities: list[dict[str, Any]]) -> dict:
    """
    Combine validated image bytes and annotations into one canonical split payload.
    """
    identity_by_id = {entry["image_id"]: entry for entry in identities}
    return {
        "images": [
            identity_by_id[image["id"]]
            for image in sorted(coco["images"], key=lambda item: item["id"])
        ],
        "annotations": _canonical_annotations(coco),
    }


def canonical_coco_identity(source_dir: str | Path) -> dict[str, Any]:
    """
    Build a root-independent identity for a train/val/test COCO source.

    All source image bytes, including images without annotations, participate in
    the fingerprint. COCO document ordering and informational metadata do not.
    """
    source = Path(source_dir).expanduser().resolve()
    canonical_splits = {}
    category_mapping = None
    for split in SPLIT_NAMES:
        coco, identities = _read_split(source, split)
        categories = _canonical_categories(coco)
        if category_mapping is None:
            category_mapping = categories
        elif category_mapping != categories:
            raise ValueError("COCO category mapping differs between source splits")
        canonical_splits[split] = _canonical_split(coco, identities)
    canonical = {
        "schema_version": 1,
        "format": "coco_source",
        "category_mapping": category_mapping,
        "splits": canonical_splits,
    }
    return {
        "schema_version": 1,
        "canonical_source": canonical,
        "source_fingerprint": _fingerprint(canonical),
    }


def _source_images(identity: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    """
    Return canonical source image entries for one logical split.
    """
    return identity["canonical_source"]["splits"][split]["images"]


def _source_annotations(
    identity: Mapping[str, Any], split: str
) -> list[dict[str, Any]]:
    """
    Return canonical source annotation entries for one logical split.
    """
    return identity["canonical_source"]["splits"][split]["annotations"]


def _normalized_boxes(
    image: Mapping[str, Any],
    annotations: list[Mapping[str, Any]],
    class_ids: Mapping[int, int],
) -> list[tuple[int, float, float, float, float]]:
    """
    Convert COCO xywh annotations into sorted YOLO normalized boxes.
    """
    width = image["width"]
    height = image["height"]
    boxes = []
    for annotation in annotations:
        x, y, box_width, box_height = annotation["bbox"]
        boxes.append(
            (
                class_ids[annotation["category_id"]],
                (x + box_width / 2) / width,
                (y + box_height / 2) / height,
                box_width / width,
                box_height / height,
            )
        )
    return sorted(boxes)


def _split_annotations_by_image(
    identity: Mapping[str, Any], split: str
) -> dict[int, list[dict[str, Any]]]:
    """
    Group canonical source annotations by their source image ID.
    """
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in _source_annotations(identity, split):
        result[annotation["image_id"]].append(annotation)
    return result


def _strip_yaml_comment(line: str) -> str:
    """
    Remove comments from the constrained, scalar YAML accepted by this module.
    """
    return line.split("#", maxsplit=1)[0].rstrip()


def _yaml_scalar(value: str) -> str:
    """
    Decode an unquoted or simply quoted scalar from a dataset YAML file.
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_yolo_yaml(yaml_path: Path) -> dict[str, Any]:
    """
    Read the narrow data.yaml subset required for train/val/test YOLO layouts.
    """
    values: dict[str, str] = {}
    names: dict[int, str] = {}
    in_names = False
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = _strip_yaml_comment(raw_line)
        if not line.strip():
            continue
        if line[0].isspace():
            if not in_names or ":" not in line:
                raise ValueError(f"Unsupported YAML entry in {yaml_path}: {raw_line!r}")
            key, value = line.strip().split(":", maxsplit=1)
            if not key.isdigit():
                raise ValueError(f"Invalid YOLO class index in {yaml_path}: {key!r}")
            names[int(key)] = _yaml_scalar(value)
            continue
        in_names = line.startswith("names:")
        if in_names:
            value = _yaml_scalar(line.split(":", maxsplit=1)[1])
            if value:
                if not (value.startswith("[") and value.endswith("]")):
                    raise ValueError(
                        f"Unsupported names value in {yaml_path}: {value!r}"
                    )
                names = {
                    index: _yaml_scalar(name)
                    for index, name in enumerate(value[1:-1].split(","))
                    if name.strip()
                }
            continue
        if ":" not in line:
            raise ValueError(f"Invalid YAML entry in {yaml_path}: {raw_line!r}")
        key, value = line.split(":", maxsplit=1)
        values[key.strip()] = _yaml_scalar(value)
    if set(values) & {"train", "val", "test"} != {"train", "val", "test"}:
        raise ValueError(f"{yaml_path}: train, val, and test paths are required")
    if not names:
        raise ValueError(f"{yaml_path}: class names are required")
    return {"path": values.get("path", "."), "splits": values, "names": names}


def _resolve_yaml_path(root: Path, value: str) -> Path:
    """
    Resolve a data.yaml path relative to its dataset root when needed.
    """
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _yolo_split_images(
    root: Path, split: str, split_path: Path
) -> tuple[Path, list[Path], str | None]:
    """
    Read a split directory or a YOLO image list and return its image directory.
    """
    if split_path.is_dir():
        return (
            split_path,
            sorted(path for path in split_path.rglob("*") if path.is_file()),
            None,
        )
    if not split_path.is_file() or split_path.suffix != ".txt":
        raise ValueError(f"YOLO {split} path is not an image directory or list")
    try:
        split_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"YOLO {split} list must stay below {root}") from error
    image_dir = root / "images" / split
    image_paths = []
    for line in split_path.read_text(encoding="utf-8").strip().splitlines():
        if line.startswith("./"):
            image_path = (split_path.parent / line).resolve()
        elif Path(line).is_absolute():
            image_path = Path(line).resolve()
        else:
            raise ValueError(
                f"YOLO {split} list entries must start with ./ or be absolute"
            )
        if not image_path.is_file() or not image_path.is_relative_to(image_dir):
            raise ValueError(f"YOLO {split} list has an invalid image: {line}")
        image_paths.append(image_path)
    directory_images = {
        path for path in image_dir.rglob("*") if path.is_file()
    }
    if (
        len(image_paths) != len(set(image_paths))
        or set(image_paths) != directory_images
    ):
        raise ValueError(f"YOLO {split} list membership differs from images")
    return image_dir, sorted(image_paths), _file_hash(split_path)


def _read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    Parse normalized class/xywh rows, treating a missing negative label as empty.
    """
    if not label_path.is_file():
        return []
    boxes = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or not fields[0].isdigit():
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
        values = tuple(float(value) for value in fields[1:])
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite YOLO label at {label_path}:{line_number}")
        x, y, width, height = values
        if (
            x < 0
            or x > 1
            or y < 0
            or y > 1
            or width <= 0
            or width > 1
            or height <= 0
            or height > 1
            or x - width / 2 < -1e-6
            or y - height / 2 < -1e-6
            or x + width / 2 > 1 + 1e-6
            or y + height / 2 > 1 + 1e-6
        ):
            raise ValueError(f"Out-of-range YOLO label at {label_path}:{line_number}")
        boxes.append((int(fields[0]), *values))
    return sorted(boxes)


def _boxes_match(
    actual: list[tuple[int, float, float, float, float]],
    expected: list[tuple[int, float, float, float, float]],
) -> bool:
    """
    Compare normalized YOLO boxes with enough tolerance for text formatting.
    """
    if len(actual) != len(expected):
        return False
    return all(
        actual_box[0] == expected_box[0]
        and all(
            abs(actual_value - expected_value) <= 1e-6
            for actual_value, expected_value in zip(
                actual_box[1:], expected_box[1:], strict=True
            )
        )
        for actual_box, expected_box in zip(actual, expected, strict=True)
    )


def validate_yolo_layout(
    source_dir: str | Path,
    yolo_dir: str | Path,
    *,
    yaml_name: str = "data.yaml",
) -> dict[str, Any]:
    """
    Validate a YOLO layout against canonical COCO membership, classes, and boxes.

    The returned loader fingerprint includes image and label file hashes plus a
    normalized split configuration. The canonical source fingerprint remains
    independent of all roots and loader-specific filenames.
    """
    source_identity = canonical_coco_identity(source_dir)
    root = Path(yolo_dir).expanduser().resolve()
    yaml_path = root / yaml_name
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing YOLO dataset YAML: {yaml_path}")
    config = _read_yolo_yaml(yaml_path)
    yaml_root = _resolve_yaml_path(yaml_path.parent, config["path"])
    if yaml_root != root:
        raise ValueError("YOLO data.yaml path must resolve to its dataset root")
    categories = source_identity["canonical_source"]["category_mapping"]
    expected_names = [category["name"] for category in categories]
    actual_names = [config["names"][index] for index in sorted(config["names"])]
    if (
        actual_names != expected_names
        or list(config["names"]) != list(range(len(categories)))
    ):
        raise ValueError("YOLO class names do not match the COCO category mapping")
    class_ids = {category["id"]: index for index, category in enumerate(categories)}
    payload_splits = {}
    image_list_hashes = {}
    for split in SPLIT_NAMES:
        split_path = _resolve_yaml_path(yaml_root, config["splits"][split])
        image_dir, image_paths, list_hash = _yolo_split_images(root, split, split_path)
        try:
            image_relative = image_dir.relative_to(root)
        except ValueError as error:
            raise ValueError(f"YOLO {split} images must stay below {root}") from error
        if len(image_relative.parts) < 2 or image_relative.parts[0] != "images":
            raise ValueError(f"YOLO {split} path must be below images/")
        label_dir = root / "labels" / Path(*image_relative.parts[1:])
        if list_hash is not None:
            image_list_hashes[split] = list_hash
        label_paths = sorted(
            path for path in label_dir.rglob("*.txt") if path.is_file()
        )
        if not image_paths:
            raise ValueError(f"YOLO {split} image split is empty")
        expected_images = _source_images(source_identity, split)
        expected_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for image in expected_images:
            expected_by_hash[image["sha256"]].append(image)
        actual_by_hash: dict[str, list[Path]] = defaultdict(list)
        for image_path in image_paths:
            actual_by_hash[_file_hash(image_path)].append(image_path)
        if Counter(
            {digest: len(paths) for digest, paths in actual_by_hash.items()}
        ) != Counter(image["sha256"] for image in expected_images):
            raise ValueError(f"YOLO {split} image membership differs from COCO")
        annotations_by_image = _split_annotations_by_image(source_identity, split)
        payload_images = []
        for digest, paths in sorted(actual_by_hash.items()):
            source_images = expected_by_hash[digest]
            if len(paths) != len(source_images):
                raise ValueError(f"YOLO {split} image multiplicity differs from COCO")
            expected_boxes = [
                (
                    source_image,
                    _normalized_boxes(
                        source_image,
                        annotations_by_image[source_image["image_id"]],
                        class_ids,
                    ),
                )
                for source_image in source_images
            ]
            for image_path in paths:
                relative_image = image_path.relative_to(image_dir)
                label_path = (label_dir / relative_image).with_suffix(".txt")
                actual_boxes = _read_yolo_labels(label_path)
                match = next(
                    (
                        index
                        for index, (_, candidate_boxes) in enumerate(expected_boxes)
                        if _boxes_match(actual_boxes, candidate_boxes)
                    ),
                    None,
                )
                if match is None:
                    raise ValueError(f"YOLO {split} labels disagree for {image_path}")
                _, matched_boxes = expected_boxes.pop(match)
                payload_images.append(
                    {
                        "image": image_path.relative_to(root).as_posix(),
                        "image_sha256": digest,
                        "label": label_path.relative_to(root).as_posix(),
                        "label_sha256": (
                            _file_hash(label_path) if label_path.is_file() else None
                        ),
                        "boxes": matched_boxes,
                    }
                )
        expected_labels = {
            (label_dir / image.relative_to(image_dir)).with_suffix(".txt")
            for image in image_paths
        }
        if not set(label_paths).issubset(expected_labels):
            raise ValueError(f"YOLO {split} label membership differs from images")
        payload_splits[split] = sorted(payload_images, key=lambda item: item["image"])
    loader = {
        "schema_version": 1,
        "format": "yolo",
        "category_mapping": {
            str(index): category["id"]
            for index, category in enumerate(categories)
        },
        "class_names": expected_names,
        "data_yaml_sha256": _file_hash(yaml_path),
        "split_configuration": {
            split: _resolve_yaml_path(yaml_root, config["splits"][split])
            .relative_to(root)
            .as_posix()
            for split in SPLIT_NAMES
        },
        "splits": payload_splits,
    }
    if image_list_hashes:
        loader["image_list_sha256"] = image_list_hashes
    return {
        "source_fingerprint": source_identity["source_fingerprint"],
        "loader": loader,
        "loader_fingerprint": _fingerprint(loader),
    }


def validate_rfdetr_layout(
    source_dir: str | Path,
    rfdetr_dir: str | Path,
) -> dict[str, Any]:
    """
    Validate an RF-DETR train/valid/test layout against its COCO source identity.
    """
    source_identity = canonical_coco_identity(source_dir)
    root = Path(rfdetr_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing RF-DETR layout manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_derived(root, manifest)
    mapping = manifest.get("category_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("RF-DETR manifest has no category mapping")
    categories = source_identity["canonical_source"]["category_mapping"]
    source_to_training = mapping.get("source_to_training", {})
    training_to_source = mapping.get("training_to_source", {})
    if set(source_to_training) != {str(category["id"]) for category in categories}:
        raise ValueError("RF-DETR category mapping differs from COCO")
    if len(training_to_source) != len(categories):
        raise ValueError("RF-DETR category mapping is not bijective")
    for category in categories:
        training_id = source_to_training[str(category["id"])]
        if (
            type(training_id) is not int
            or training_to_source.get(str(training_id)) != category["id"]
        ):
            raise ValueError(
                "RF-DETR category mapping does not round-trip the COCO class"
            )
    expected_predictions = {
        str(index): training_to_source[str(training_id)]
        for index, training_id in enumerate(
            sorted(int(value) for value in training_to_source)
        )
    }
    if mapping.get("prediction_to_source") != expected_predictions:
        raise ValueError("RF-DETR prediction mapping differs from training labels")
    expected_categories = deepcopy(categories)
    for category in expected_categories:
        category["id"] = source_to_training[str(category["id"])]
    if mapping.get("class_names") != [category["name"] for category in categories]:
        raise ValueError("RF-DETR class names differ from COCO")
    payload_splits = {}
    for split, loader_split in SPLIT_NAMES.items():
        annotation_path = root / loader_split / "_annotations.coco.json"
        derived = json.loads(annotation_path.read_text(encoding="utf-8"))
        if _canonical_categories(derived) != sorted(
            expected_categories, key=lambda item: item["id"]
        ):
            raise ValueError("RF-DETR derived categories differ from COCO")
        source_images = _source_images(source_identity, split)
        actual_images = [
            {
                "id": image.get("id"),
                "file_name": image.get("file_name"),
                "width": image.get("width"),
                "height": image.get("height"),
            }
            for image in derived.get("images", [])
        ]
        if sorted(actual_images, key=lambda item: item["id"]) != [
            {
                "id": image["image_id"],
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
            }
            for image in source_images
        ]:
            raise ValueError(f"RF-DETR {split} image membership differs from COCO")
        expected_annotations = deepcopy(_source_annotations(source_identity, split))
        for annotation in expected_annotations:
            annotation["category_id"] = source_to_training[
                str(annotation["category_id"])
            ]
        actual_annotations = [
            {
                key: annotation.get(key, [] if key == "segmentation" else 0)
                for key in (
                    "id",
                    "image_id",
                    "category_id",
                    "bbox",
                    "area",
                    "iscrowd",
                    "segmentation",
                )
            }
            for annotation in derived.get("annotations", [])
        ]
        if (
            sorted(actual_annotations, key=lambda item: item["id"])
            != expected_annotations
        ):
            raise ValueError(f"RF-DETR {split} annotations differ from COCO")
        payload_splits[split] = {
            "annotation_sha256": _file_hash(annotation_path),
            "images": [
                {
                    "file_name": image["file_name"],
                    "sha256": _file_hash(root / loader_split / image["file_name"]),
                }
                for image in source_images
            ],
        }
    loader = {
        "schema_version": 1,
        "format": "rfdetr",
        "category_mapping": mapping,
        "splits": payload_splits,
    }
    return {
        "source_fingerprint": source_identity["source_fingerprint"],
        "loader": loader,
        "loader_fingerprint": _fingerprint(loader),
    }


def capture_dataset_identity(
    identity_path: str | Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Create an immutable dataset identity record or verify an existing equivalent one.
    """
    path = Path(identity_path).expanduser().resolve()
    record = json.loads(_json_bytes(dict(identity)))
    if path.exists():
        return verify_dataset_identity(path, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_json_bytes(record))
        try:
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, path)
        except FileExistsError:
            return verify_dataset_identity(path, record)
        finally:
            if temporary.exists():
                temporary.unlink()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return record


def verify_dataset_identity(
    identity_path: str | Path,
    actual_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Verify an identity record without replacing it when current content differs.
    """
    path = Path(identity_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset identity record: {path}")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    actual = json.loads(_json_bytes(dict(actual_identity)))
    if recorded != actual:
        raise ValueError(
            f"Dataset identity mismatch at {path}; the original record was preserved"
        )
    return recorded


def _copy_or_link(source: Path, destination: Path, link_images: bool) -> None:
    """
    Materialize one source image without modifying the source dataset.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_images:
        try:
            destination.symlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _write_yolo_subset(
    source_identity: Mapping[str, Any],
    destination: Path,
    source_images_root: Path,
    dataset_root: Path,
    link_images: bool,
) -> None:
    """
    Write a YOLO layout with labels generated directly from selected COCO boxes.
    """
    categories = source_identity["canonical_source"]["category_mapping"]
    class_ids = {category["id"]: index for index, category in enumerate(categories)}
    for split in SPLIT_NAMES:
        annotations_by_image = _split_annotations_by_image(source_identity, split)
        for image in _source_images(source_identity, split):
            relative = Path(image["file_name"])
            _copy_or_link(
                source_images_root / split / relative,
                destination / "images" / split / relative,
                link_images,
            )
            label_path = (destination / "labels" / split / relative).with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            boxes = _normalized_boxes(
                image,
                annotations_by_image[image["image_id"]],
                class_ids,
            )
            label_path.write_text(
                "".join(
                    f"{class_id} {x:.10g} {y:.10g} {width:.10g} {height:.10g}\n"
                    for class_id, x, y, width, height in boxes
                ),
                encoding="utf-8",
            )
    names = "\n".join(
        f"  {index}: {category['name']}" for index, category in enumerate(categories)
    )
    (destination / "data.yaml").write_text(
        f"path: {dataset_root}\n"
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def _write_coco_subset(
    source_dir: Path,
    destination: Path,
    limits: Mapping[str, int],
    link_images: bool,
) -> dict[str, Any]:
    """
    Write an immutable selected COCO source layout while preserving original IDs.
    """
    (destination / "annotations").mkdir(parents=True)
    selected_ids = {}
    for split in SPLIT_NAMES:
        coco, _ = _read_split(source_dir, split)
        if len(coco["images"]) < limits[split]:
            raise ValueError(
                f"{split} has {len(coco['images'])} images; smoke requires "
                f"{limits[split]}"
            )
        subset = _smoke_subset(coco, limits[split])
        selected_ids[split] = sorted(image["id"] for image in subset["images"])
        (destination / "annotations" / f"instances_{split}.json").write_bytes(
            _json_bytes(subset)
        )
        for image in subset["images"]:
            relative = Path(image["file_name"])
            _copy_or_link(
                source_dir / "images" / split / relative,
                destination / "images" / split / relative,
                link_images,
            )
    return selected_ids


def smoke_layout_directory(source_dir: str | Path) -> Path:
    """
    Select a shared smoke directory bound to the current source content.
    """
    source = Path(source_dir).expanduser().resolve()
    fingerprint = canonical_coco_identity(source)["source_fingerprint"]
    return DATA_ROOT / "processed" / "smoke" / source.name / fingerprint


def prepare_smoke_layouts(
    source_dir: str | Path,
    destination_dir: str | Path | None = None,
    *,
    dataset_name: str | None = None,
    limits: Mapping[str, int] = SMOKE_LIMITS,
    link_images: bool = True,
) -> dict[str, Any]:
    """
    Prepare equivalent 16/8/8 COCO, YOLO, and RF-DETR smoke layouts.

    By default output is created below ``DATA_ROOT/processed``. Existing output
    is never replaced: its identity record and all three layouts must still
    validate before it can be reused.
    """
    if set(limits) != set(SPLIT_NAMES):
        raise ValueError("Smoke limits must specify train, val, and test")
    source = Path(source_dir).expanduser().resolve()
    target_name = dataset_name or f"{source.name}_smoke"
    destination = (
        (
            smoke_layout_directory(source)
            if dataset_name is None
            else DATA_ROOT / "processed" / target_name
        )
        if destination_dir is None
        else Path(destination_dir).expanduser().resolve()
    )
    if destination.exists() or destination.is_symlink():
        prepare_coco_dataset(
            destination / "coco", destination / "rfdetr", link_images=False
        )
        coco_identity = canonical_coco_identity(destination / "coco")
        yolo_identity = validate_yolo_layout(destination / "coco", destination / "yolo")
        rfdetr_identity = validate_rfdetr_layout(
            destination / "coco", destination / "rfdetr"
        )
        expected = {
            "schema_version": 1,
            "kind": "smoke_dataset",
            "smoke_limits": dict(limits),
            "parent_source_fingerprint": canonical_coco_identity(source)[
                "source_fingerprint"
            ],
            "selected_ids": {
                split: [
                    image["image_id"]
                    for image in _source_images(coco_identity, split)
                ]
                for split in SPLIT_NAMES
            },
            "subset_source_fingerprint": coco_identity["source_fingerprint"],
            "loader_fingerprints": {
                "coco": coco_identity["source_fingerprint"],
                "yolo": yolo_identity["loader_fingerprint"],
                "rfdetr": rfdetr_identity["loader_fingerprint"],
            },
        }
        return verify_dataset_identity(destination / IDENTITY_FILE_NAME, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".smoke-", dir=destination.parent))
    promoted = False
    try:
        selected_ids = _write_coco_subset(source, staging / "coco", limits, link_images)
        subset_identity = canonical_coco_identity(staging / "coco")
        _write_yolo_subset(
            subset_identity,
            staging / "yolo",
            source / "images",
            destination / "yolo",
            link_images,
        )
        manifest = prepare_coco_dataset(
            staging / "coco", staging / "rfdetr", link_images=False
        )
        # The loader manifest survives promotion, so persist its final roots.
        # Roots are intentionally excluded from the content fingerprint.
        manifest["source_dir"] = str(destination / "coco")
        manifest["dataset_dir"] = str(destination / "rfdetr")
        (staging / "rfdetr" / "manifest.json").write_bytes(_json_bytes(manifest))
        staging.rename(destination)
        promoted = True
        subset_identity = canonical_coco_identity(destination / "coco")
        yolo_identity = validate_yolo_layout(destination / "coco", destination / "yolo")
        rfdetr_identity = validate_rfdetr_layout(destination / "coco", destination / "rfdetr")
        record = {
            "schema_version": 1,
            "kind": "smoke_dataset",
            "smoke_limits": dict(limits),
            "parent_source_fingerprint": canonical_coco_identity(source)[
                "source_fingerprint"
            ],
            "selected_ids": selected_ids,
            "subset_source_fingerprint": subset_identity["source_fingerprint"],
            "loader_fingerprints": {
                "coco": subset_identity["source_fingerprint"],
                "yolo": yolo_identity["loader_fingerprint"],
                "rfdetr": rfdetr_identity["loader_fingerprint"],
            },
        }
        capture_dataset_identity(destination / IDENTITY_FILE_NAME, record)
        return record
    except Exception:
        if promoted and destination.exists():
            shutil.rmtree(destination)
        raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)