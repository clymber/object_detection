"""
Validate frozen COCO splits and prepare an auditable RF-DETR directory layout.

This module deliberately has no detector or torch dependency. Source annotations
and image contents are never modified; derived caches are immutable once built.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

SPLIT_NAMES = {"train": "train", "val": "valid", "test": "test"}
MANIFEST_NAME = "manifest.json"


def _json_bytes(value: Any) -> bytes:
    """
    Serialize a manifest or annotation deterministically.
    """
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _file_hash(path: Path) -> str:
    """
    Hash the complete file without loading large image files into memory.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    """
    Require a real integer rather than a bool, float, or numeric string.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}; got {value!r}")
    return value


def _relative_image_path(file_name: Any) -> Path:
    """
    Reject image names that escape the split or collide with generated files.
    """
    if not isinstance(file_name, str) or not file_name or "\\" in file_name:
        raise ValueError(f"Invalid COCO file_name: {file_name!r}")
    path = PurePosixPath(file_name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"COCO file_name must stay inside its split: {file_name!r}")
    if path.as_posix() != file_name or path.parts[0] == "_annotations.coco.json":
        raise ValueError(f"Noncanonical or reserved COCO file_name: {file_name!r}")
    return Path(*path.parts)


def _read_split(source_dir: Path, split: str) -> tuple[dict, list[dict]]:
    """
    Validate image files, dimensions, categories, IDs, boxes, and references.
    """
    annotation_path = source_dir / "annotations" / f"instances_{split}.json"
    with annotation_path.open(encoding="utf-8") as stream:
        coco = json.load(stream)
    if not isinstance(coco, dict):
        raise ValueError(f"{annotation_path}: expected a COCO object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(coco.get(key), list):
            raise ValueError(f"{annotation_path}: {key} must be a list")
    if len(coco["categories"]) != 1:
        raise ValueError(f"{annotation_path}: expected exactly one basketball class")
    category = coco["categories"][0]
    category_id = _integer(category.get("id"), f"{split} category ID")
    if category.get("name") != "basketball":
        raise ValueError(f"{annotation_path}: expected category name 'basketball'")

    images = {}
    file_names = set()
    identities = []
    for entry in coco["images"]:
        image_id = _integer(entry.get("id"), f"{split} image ID")
        if image_id in images:
            raise ValueError(f"{split}: duplicate image ID {image_id}")
        file_name = entry.get("file_name")
        relative_path = _relative_image_path(file_name)
        if file_name in file_names:
            raise ValueError(f"{split}: duplicate image file_name {file_name!r}")
        file_names.add(file_name)
        width = _integer(entry.get("width"), f"{split}/{file_name} width", minimum=1)
        height = _integer(entry.get("height"), f"{split}/{file_name} height", minimum=1)
        image_path = source_dir / "images" / split / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing {split} image: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"{split}/{file_name}: COCO size {(width, height)} does not "
                    f"match image size {image.size}"
                )
        images[image_id] = entry
        identities.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "sha256": _file_hash(image_path),
                "width": width,
                "height": height,
            }
        )

    annotation_ids = set()
    for entry in coco["annotations"]:
        annotation_id = _integer(entry.get("id"), f"{split} annotation ID")
        if annotation_id in annotation_ids:
            raise ValueError(f"{split}: duplicate annotation ID {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = _integer(entry.get("image_id"), f"{split} annotation image ID")
        if image_id not in images:
            raise ValueError(
                f"{split}: annotation {annotation_id} has unknown image ID"
            )
        if entry.get("category_id") != category_id:
            raise ValueError(
                f"{split}: annotation {annotation_id} has unknown category"
            )
        bbox = entry.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{split}: annotation {annotation_id} needs an xywh bbox")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            for value in bbox
        ):
            raise ValueError(f"{split}: annotation {annotation_id} has nonfinite bbox")
        x, y, width, height = bbox
        image = images[image_id]
        if (
            x < 0
            or y < 0
            or width <= 0
            or height <= 0
            or x + width > image["width"] + 1e-6
            or y + height > image["height"] + 1e-6
        ):
            raise ValueError(
                f"{split}: annotation {annotation_id} has invalid/out-of-image "
                f"bbox {bbox}; source boxes will not be silently clipped"
            )
        area = entry.get("area")
        if (
            isinstance(area, bool)
            or not isinstance(area, (int, float))
            or not math.isfinite(area)
            or area <= 0
        ):
            raise ValueError(f"{split}: annotation {annotation_id} needs positive area")
        if entry.get("iscrowd", 0) not in (0, 1):
            raise ValueError(f"{split}: annotation {annotation_id} has invalid iscrowd")
    return coco, identities


def _counts(coco: dict) -> dict[str, int]:
    """
    Count images with and without annotations as well as total boxes.
    """
    positive_ids = {entry["image_id"] for entry in coco["annotations"]}
    return {
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "positive_images": len(positive_ids),
        "negative_images": len(coco["images"]) - len(positive_ids),
    }


def _smoke_subset(coco: dict, limit: int) -> dict:
    """
    Select deterministic source IDs, retaining a positive and negative if present.
    """
    _integer(limit, "smoke image limit", minimum=1)
    if len(coco["images"]) <= limit:
        return deepcopy(coco)
    positives = {entry["image_id"] for entry in coco["annotations"]}
    images = sorted(coco["images"], key=lambda item: item["id"])
    positive = next((entry for entry in images if entry["id"] in positives), None)
    negative = next((entry for entry in images if entry["id"] not in positives), None)
    if limit < 2 and positive is not None and negative is not None:
        raise ValueError(
            "Mixed positive/negative smoke splits require at least 2 images"
        )
    selected = {entry["id"] for entry in (positive, negative) if entry is not None}
    for entry in images:
        if len(selected) >= limit:
            break
        selected.add(entry["id"])
    result = deepcopy(coco)
    result["images"] = [entry for entry in coco["images"] if entry["id"] in selected]
    result["annotations"] = [
        entry for entry in coco["annotations"] if entry["image_id"] in selected
    ]
    return result


def _duplicate_images(identities: dict[str, list[dict]]) -> list[dict]:
    """
    Report identical image bytes within or across original splits.
    """
    by_hash = defaultdict(list)
    for split, entries in identities.items():
        for entry in entries:
            by_hash[entry["sha256"]].append(
                {
                    "split": split,
                    "image_id": entry["image_id"],
                    "file_name": entry["file_name"],
                }
            )
    return [
        {
            "sha256": digest,
            "cross_split": len({entry["split"] for entry in entries}) > 1,
            "images": entries,
        }
        for digest, entries in sorted(by_hash.items())
        if len(entries) > 1
    ]


def _validate_derived(destination_dir: Path, manifest: dict) -> None:
    """
    Reject incomplete or edited derived annotations and changed image contents.
    """
    for split, summary in manifest["splits"].items():
        split_dir = destination_dir / summary["loader_split"]
        annotations = split_dir / "_annotations.coco.json"
        if (
            not annotations.is_file()
            or _file_hash(annotations) != summary["derived_annotation_sha256"]
        ):
            raise ValueError(f"Stale derived dataset: {split} annotations changed")
        for entry in summary["images_identity"]:
            image = split_dir / entry["file_name"]
            if not image.is_file() or _file_hash(image) != entry["sha256"]:
                raise ValueError(f"Stale derived dataset: {split}/{entry['file_name']}")


def prepare_coco_dataset(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    expected_counts: Mapping[str, int] | None = None,
    category_id: int | None = None,
    smoke_limits: Mapping[str, int] | None = None,
    link_images: bool = True,
    allow_cross_split_duplicates: bool = False,
) -> dict[str, Any]:
    """
    Build or verify RF-DETR train/valid/test splits from frozen COCO annotations.

    ``category_id`` selects the training label ID; ``None`` preserves source IDs.
    A reversible mapping always records the original category. ``smoke_limits``
    must bound all three splits explicitly; it creates a labeled subset without
    resplitting or moving images between splits. Count assertions always apply
    to the original full dataset. Existing caches are validated, never replaced.
    Duplicates within a split remain unchanged and emit a warning. Identical
    image bytes across splits are rejected unless explicitly acknowledged.
    Every duplicate group is recorded in the manifest.
    """
    source_dir = Path(source_dir).expanduser().resolve()
    destination_dir = Path(destination_dir).expanduser().resolve()
    if destination_dir == source_dir or source_dir in destination_dir.parents:
        raise ValueError(
            "The derived dataset must be stored outside the source dataset"
        )
    if category_id is not None:
        _integer(category_id, "training category ID")
    if expected_counts is not None and set(expected_counts) != set(SPLIT_NAMES):
        raise ValueError("expected_counts must specify train, val, and test")
    if smoke_limits is not None and set(smoke_limits) != set(SPLIT_NAMES):
        raise ValueError("smoke_limits must explicitly bound train, val, and test")

    source_coco = {}
    source_identities = {}
    source_category_id = None
    for split in SPLIT_NAMES:
        coco, identities = _read_split(source_dir, split)
        current_category_id = coco["categories"][0]["id"]
        if source_category_id is not None and current_category_id != source_category_id:
            raise ValueError("Basketball category IDs differ between source splits")
        source_category_id = current_category_id
        if (
            expected_counts is not None
            and len(coco["images"]) != expected_counts[split]
        ):
            raise ValueError(
                f"{split}: expected {expected_counts[split]} source images, "
                f"found {len(coco['images'])}"
            )
        source_coco[split] = coco
        source_identities[split] = identities

    duplicates = _duplicate_images(source_identities)
    if duplicates:
        cross_split = sum(group["cross_split"] for group in duplicates)
        message = (
            f"Found {len(duplicates)} duplicate image-content groups, "
            f"including {cross_split} across splits. "
            "The frozen split membership has not been changed. "
            f"First group: {duplicates[0]['images']}"
        )
        if cross_split and not allow_cross_split_duplicates:
            raise ValueError(
                message + "; set allow_cross_split_duplicates=True to acknowledge"
            )
        warnings.warn(message, UserWarning, stacklevel=2)

    training_category_id = source_category_id if category_id is None else category_id
    manifest = {
        "schema_version": 1,
        "source_dir": str(source_dir),
        "dataset_dir": str(destination_dir),
        "mode": "smoke" if smoke_limits is not None else "full",
        "smoke_limits": dict(smoke_limits) if smoke_limits is not None else None,
        "category_mapping": {
            "source_category_id": source_category_id,
            "source_to_training": {str(source_category_id): training_category_id},
            "training_to_source": {str(training_category_id): source_category_id},
            "prediction_to_source": {"0": source_category_id},
            "class_names": ["basketball"],
        },
        "duplicate_image_groups": duplicates,
        "allow_cross_split_duplicates": allow_cross_split_duplicates,
        "splits": {},
    }
    adapted = {}
    for split, coco in source_coco.items():
        current = (
            _smoke_subset(coco, smoke_limits[split])
            if smoke_limits is not None
            else deepcopy(coco)
        )
        current["categories"][0]["id"] = training_category_id
        for annotation in current["annotations"]:
            annotation["category_id"] = training_category_id
        selected_ids = {entry["id"] for entry in current["images"]}
        adapted[split] = _json_bytes(current)
        annotation_path = source_dir / "annotations" / f"instances_{split}.json"
        manifest["splits"][split] = {
            "loader_split": SPLIT_NAMES[split],
            **_counts(current),
            "source_counts": _counts(coco),
            "source_annotation_sha256": _file_hash(annotation_path),
            "derived_annotation_sha256": hashlib.sha256(adapted[split]).hexdigest(),
            "source_images_identity": source_identities[split],
            "images_identity": [
                entry
                for entry in source_identities[split]
                if entry["image_id"] in selected_ids
            ],
        }
    fingerprint_data = {
        key: value
        for key, value in manifest.items()
        if key not in {"source_dir", "dataset_dir"}
    }
    manifest["fingerprint"] = hashlib.sha256(_json_bytes(fingerprint_data)).hexdigest()

    if destination_dir.exists() or destination_dir.is_symlink():
        manifest_path = destination_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileExistsError(
                f"Refusing to overwrite {destination_dir}: no dataset manifest"
            )
        with manifest_path.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous != manifest:
            raise ValueError(
                f"Stale derived dataset at {destination_dir}: source data or "
                "preparation settings changed; choose a new destination explicitly"
            )
        _validate_derived(destination_dir, manifest)
        return manifest

    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rfdetr-", dir=destination_dir.parent))
    try:
        for split, summary in manifest["splits"].items():
            split_dir = staging / summary["loader_split"]
            split_dir.mkdir()
            for entry in summary["images_identity"]:
                source = source_dir / "images" / split / entry["file_name"]
                target = split_dir / entry["file_name"]
                target.parent.mkdir(parents=True, exist_ok=True)
                if link_images:
                    try:
                        target.symlink_to(source)
                        continue
                    except OSError:
                        pass
                shutil.copy2(source, target)
            (split_dir / "_annotations.coco.json").write_bytes(adapted[split])
        (staging / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
        if destination_dir.exists() or destination_dir.is_symlink():
            raise FileExistsError(
                f"Destination appeared during preparation: {destination_dir}"
            )
        staging.rename(destination_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def summarize_dataset(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """
    Return concise split statistics suitable for a notebook dataframe.
    """
    return [
        {
            "split": split,
            "loader_split": summary["loader_split"],
            "images": summary["images"],
            "annotations": summary["annotations"],
            "positive_images": summary["positive_images"],
            "negative_images": summary["negative_images"],
            "mode": manifest["mode"],
        }
        for split, summary in manifest["splits"].items()
    ]
