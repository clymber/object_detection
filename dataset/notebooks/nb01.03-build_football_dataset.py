# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     formats: ipynb,py:percent
#     notebook_metadata_filter: kernelspec,jupytext,title,authors,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (Object Detection Dataset)
#     language: python
#     name: object-detection-dataset
# ---

# %% [markdown]
# # Build the combined football dataset
#
# This notebook combines the selected Roboflow datasets into a single-class
# football detection dataset. It records the provenance of every source image,
# applies the agreed class rules, removes exact duplicate pixels, and assigns each
# raw dataset to exactly one of the train, validation, or test splits.
#
# The final retained images are exported in matching COCO and Ultralytics YOLO
# formats. Both exports use SHA-256 image names and contain one class named
# `football`. A CSV manifest connects each output image to its source and records
# every exclusion decision.
#
# This preprocessing stage handles exact duplicates only. Visually similar images,
# adjacent video frames, and grouping by match or video remain outside its scope.

# %%
"""
Build a deduplicated, grouped football detection dataset.
"""

# %load_ext autoreload
# %autoreload 2
# %aimport -csv, -textwrap, -functools, -IPython, -ultralytics, -pandas, -datumaro
# %aimport -random

from dataset_builder.config import DATA_ROOT, OUTPUT_ROOT, WORKSPACE_ROOT
from detection_common import configure_stdio_relative_path

configure_stdio_relative_path(WORKSPACE_ROOT)

# %%
import hashlib
import json
import math
import os
import struct
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import datumaro as dm
import pandas as pd
from dataset_builder import (
    prefer_hardlinked_datumaro_media,
    summarize_coco_datasets,
)
from dataset_builder import roboflow as rf_platform
from dataset_builder.datumaro import ExportFormat as DatumaroExpFmt
from dataset_builder.roboflow import RoboflowFormat as rf_format
from IPython.display import display
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

# %% [markdown]
# ## 1. Download the source datasets
#
# The URLs below are the canonical inputs for this build. The Roboflow helper stores
# each COCO dataset under `datasets/sources/roboflow/` and reuses an existing local
# download. Later stages read the returned locations without changing any source
# image or annotation file.

# %%
URLS = (
    "https://universe.roboflow.com/coffee-dcy4j/football-5beob/dataset/1",
    "https://universe.roboflow.com/ahmad-wjdkl/football-gsgma/dataset/1",
    "https://universe.roboflow.com/uc-berkely-w210-tracer/tracer-football/dataset/2",
    "https://universe.roboflow.com/amar-gpnyu/football-detection-gjrtq/dataset/1",
    "https://universe.roboflow.com/fish-ibgxf/football-2k4ty/dataset/1",
    "https://universe.roboflow.com/football-wekgj/football-ywyvk/dataset/2",
    "https://universe.roboflow.com/szeivan2729-gmail-com/football-rmhvy/dataset/19",
    "https://universe.roboflow.com/maxim-xjjwj/football-ablqb/dataset/2",
    "https://universe.roboflow.com/mbs4542-h9fkh/football-tnb17/dataset/6",
    "https://universe.roboflow.com/lwl/football-csdy0/dataset/10",
    "https://universe.roboflow.com/via-i65sm/football-wsef2/dataset/1",
    (
        "https://universe.roboflow.com/comsats-university-lahore/"
        "football-detection-ftt4q/dataset/1"
    ),
)

datasets = []
for url in URLS:
    print(f"Processing dataset from URL: {url}")
    dataset = rf_platform.download_by_url(url, rf_format.COCO)
    datasets.append(dataset)

# %% [markdown]
# ### 1.1 Review the raw dataset totals
#
# Before applying any rule, summarize image counts by original split and annotation
# counts by source. The totals establish the complete input population against which
# all later retained and excluded counts can be reconciled.

# %%
summary = summarize_coco_datasets([dataset.location for dataset in datasets])
display(summary.style.hide(axis="index"))

total_annotations = summary["annotations"].sum()
total_images = summary["train"].sum() + summary["val"].sum()
total_images += summary["test"].sum()
print(f"Total images: {total_images}, total annotations: {total_annotations}")

# %% [markdown]
# ### 1.2 Define the source-specific class rules
#
# Each `SrcCfg` names the raw dataset version and separates its labels into
# three explicit actions:
#
# - `map_labels`: convert these annotations to the target `football` class;
# - `remove_labels`: discard these annotations while retaining a mixed image when a
#   mapped football box remains; and
# - `exclude_image_labels`: exclude the complete image when any of these ambiguous
#   labels occurs.
#
# The table below exposes every mapping decision before the inventory is filtered.

# %%


class SrcCfg(NamedTuple):
    """
    Define the label rules for one raw source dataset.
    """

    dataset_id: str
    version: int
    map_labels: set[str]
    remove_labels: set[str]
    exclude_image_labels: set[str]


SOURCE_CONFIG = (
    SrcCfg("football-5beob", 1, {"ball"}, {"person"}, set(),),
    SrcCfg("football-gsgma", 1, {"football", "soccer", "soccer ball", "soccer_ball"}, {"football2"}, set(),),
    SrcCfg("tracer-football", 2, {"football", "soccer", "Soccer ball"}, {"ball", "other", "others"}, set(),),
    SrcCfg("football-detection-ftt4q", 1, {"Football"}, set(), set(),),
    SrcCfg("football-detection-gjrtq", 1, {"ball", "Football"}, {"big", "Player", "referee"}, set(),),
    SrcCfg("football-2k4ty", 1, {"football"}, set(), set()),
    SrcCfg("football-ywyvk", 2, {"Football"}, set(), set()),
    SrcCfg("football-rmhvy", 19, {"Football", "football_1"}, {"hball", "Not football"}, set(),),
    SrcCfg("football-ablqb", 2, {"football"}, set(), set()),
    SrcCfg("football-tnb17", 6, {"football"}, set(), {"other"},),
    SrcCfg("football-csdy0", 10, {"football"}, set(), {"otherball"}),
    SrcCfg("football-wsef2", 1, {"football"}, set(), set()),
)

SPLIT_NAMES = {"train": "train", "valid": "val", "test": "test"}
BUILD_OUTPUT_ROOT = OUTPUT_ROOT / "dataset" / "build_football_dataset"
COMPOSED_ROOT = DATA_ROOT / "composed"
RANDOM_STATE = 42
N_SPLIT_CANDIDATES = 100

source_configs = []
downloaded_sources = {
    Path(dataset.location).parent.name: (url, dataset)
    for url, dataset in zip(URLS, datasets, strict=True)
}
for source_order, config in enumerate(SOURCE_CONFIG):
    if config.dataset_id not in downloaded_sources:
        raise FileNotFoundError(
            f"Downloaded source is missing: {config.dataset_id}/{config.version}"
        )
    url, dataset = downloaded_sources[config.dataset_id]
    source_root = Path(dataset.location)
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset cannot be loaded: {source_root}")
    source_configs.append(
        {
            "url": url,
            "source_order": source_order,
            "dataset_id": config.dataset_id,
            "version": config.version,
            "source_root": source_root,
            "map_labels": config.map_labels,
            "remove_labels": config.remove_labels,
            "exclude_image_labels": config.exclude_image_labels,
        }
    )

config_table = pd.DataFrame(
    {
        "dataset_id": config["dataset_id"],
        "version": config["version"],
        "map_to_football": ", ".join(sorted(config["map_labels"])),
        "remove": ", ".join(sorted(config["remove_labels"])),
        "exclude_whole_image": ", ".join(sorted(config["exclude_image_labels"])),
    }
    for config in source_configs
)
display(config_table.style.hide(axis="index"))

# %% [markdown]
# ## 2. Inventory every image, category, and annotation
#
# Read the COCO files directly into image, category, and annotation tables. Each
# image retains its source URL, source order, raw `dataset_id`, version, original
# split, source ID, path, and declared dimensions. Annotation rows retain their
# category and geometry.
#
# The raw `dataset_id` is also recorded as `group_id`; it becomes the indivisible
# unit during splitting. The displayed inventory summarizes the unfiltered image,
# annotation, class, and label counts for each source.

# %%
image_rows = []
category_rows = []
annotation_rows = []
issues = []

for config in source_configs:
    annotation_paths = sorted(config["source_root"].glob("*/_annotations.coco.json"))
    if not annotation_paths:
        raise FileNotFoundError("No COCO annotations found for " + config["dataset_id"])

    for annotation_path in annotation_paths:
        original_split = SPLIT_NAMES.get(
            annotation_path.parent.name,
            annotation_path.parent.name,
        )
        try:
            with annotation_path.open(encoding="utf-8") as file:
                coco = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"COCO annotation file cannot be loaded: {annotation_path}"
            ) from error

        categories = {
            category["id"]: category["name"] for category in coco["categories"]
        }
        for category in coco["categories"]:
            category_rows.append(
                {
                    "dataset_id": config["dataset_id"],
                    "version": config["version"],
                    "original_split": original_split,
                    "annotation_json_path": str(annotation_path),
                    "category_id": category["id"],
                    "category_name": category["name"],
                }
            )

        images_by_id = {}
        for image in coco["images"]:
            record_key = (
                f"{config['dataset_id']}:{config['version']}:"
                f"{original_split}:{image['id']}"
            )
            image_path = annotation_path.parent / image["file_name"]
            row = {
                "record_key": record_key,
                "source_url": config["url"],
                "source_order": config["source_order"],
                "dataset_id": config["dataset_id"],
                "group_id": config["dataset_id"],
                "version": config["version"],
                "original_split": original_split,
                "annotation_json_path": str(annotation_path),
                "source_image_id": image["id"],
                "original_name": image["file_name"],
                "source_path": str(image_path),
                "declared_width": image.get("width"),
                "declared_height": image.get("height"),
                "original_annotation_count": 0,
                "raw_annotations": [],
                "mapped_annotations": [],
                "status": "pending",
                "exclusion_reason": "",
                "pixel_sha256": "",
                "final_split": "",
                "coco_output_path": "",
                "yolo_output_path": "",
                "coco_media_method": "",
                "yolo_media_method": "",
            }
            images_by_id[image["id"]] = row
            image_rows.append(row)

        for annotation in coco["annotations"]:
            image = images_by_id.get(annotation.get("image_id"))
            category_name = categories.get(annotation.get("category_id"))
            bbox = annotation.get("bbox")
            bbox_valid = (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(value, (int, float)) for value in bbox)
                and all(math.isfinite(value) for value in bbox)
                and bbox[2] > 0
                and bbox[3] > 0
            )
            valid = image is not None and category_name is not None and bbox_valid
            issue = ""
            if image is None:
                issue = "annotation references a missing image"
            elif category_name is None:
                issue = "annotation references a missing category"
            elif not bbox_valid:
                issue = "annotation has an invalid bbox"

            row = {
                "dataset_id": config["dataset_id"],
                "version": config["version"],
                "original_split": original_split,
                "annotation_json_path": str(annotation_path),
                "annotation_id": annotation.get("id"),
                "source_image_id": annotation.get("image_id"),
                "category_id": annotation.get("category_id"),
                "category_name": category_name,
                "bbox": bbox,
                "area": annotation.get("area"),
                "iscrowd": annotation.get("iscrowd", 0),
                "segmentation": annotation.get("segmentation"),
                "valid": valid,
                "issue": issue,
            }
            annotation_rows.append(row)
            if image is not None:
                image["raw_annotations"].append(row)
                image["original_annotation_count"] += 1
            if issue:
                issues.append(
                    {
                        "record_key": image["record_key"] if image else "",
                        "dataset_id": config["dataset_id"],
                        "issue": issue,
                        "path": str(annotation_path),
                    }
                )

images_df = pd.DataFrame(image_rows)
categories_df = pd.DataFrame(category_rows)
annotations_df = pd.DataFrame(annotation_rows)
inventory_summary = (
    images_df.groupby("dataset_id", sort=False)
    .agg(
        images=("record_key", "size"),
        annotations=("original_annotation_count", "sum"),
    )
    .reset_index()
)
inventory_summary["classes"] = inventory_summary["dataset_id"].map(
    categories_df.groupby("dataset_id")["category_name"].nunique()
)
inventory_summary["labels"] = inventory_summary["dataset_id"].map(
    categories_df.groupby("dataset_id")["category_name"].agg(
        lambda labels: ", ".join(sorted(set(labels)))
    )
)
display(inventory_summary.style.hide(axis="index"))
print(
    f"Inventoried {len(images_df):,} images, {len(annotations_df):,} "
    f"annotations, and {len(categories_df):,} split-level category records."
)

# %% [markdown]
# ### 2.1 Validate the inventory
#
# Decode every image and compare its dimensions with the COCO metadata. An unreadable
# image, a dimension mismatch, or an invalid or unmapped annotation excludes the
# complete image. Removing only a bad annotation could leave a visible football
# without a target box.
#
# Recoverable problems are collected in one issues table. Supported image modes are
# accepted here and converted to RGB later only for exact-pixel hashing.

# %%
config_by_dataset = {config["dataset_id"]: config for config in source_configs}
for image in image_rows:
    try:
        with Image.open(image["source_path"]) as decoded:
            decoded.load()
            actual_width, actual_height = decoded.size
    except (OSError, ValueError) as error:
        image["status"] = "unreadable_image"
        image["exclusion_reason"] = str(error)
        issues.append(
            {
                "record_key": image["record_key"],
                "dataset_id": image["dataset_id"],
                "issue": "unreadable image",
                "path": image["source_path"],
            }
        )
        continue

    image["width"] = actual_width
    image["height"] = actual_height
    dimensions_differ = image["declared_width"] != actual_width
    dimensions_differ |= image["declared_height"] != actual_height
    if dimensions_differ:
        image["status"] = "invalid_image_dimensions"
        image["exclusion_reason"] = "decoded dimensions differ from COCO metadata"
        issues.append(
            {
                "record_key": image["record_key"],
                "dataset_id": image["dataset_id"],
                "issue": image["exclusion_reason"],
                "path": image["source_path"],
            }
        )
        continue

    if any(not annotation["valid"] for annotation in image["raw_annotations"]):
        image["status"] = "invalid_annotation"
        image["exclusion_reason"] = "image contains an invalid annotation"
        continue

    config = config_by_dataset[image["dataset_id"]]
    known_labels = (
        config["map_labels"] | config["remove_labels"] | config["exclude_image_labels"]
    )
    unmapped_labels = {
        annotation["category_name"]
        for annotation in image["raw_annotations"]
        if annotation["category_name"] not in known_labels
    }
    if unmapped_labels:
        image["status"] = "unmapped_label"
        image["exclusion_reason"] = "image contains unmapped labels: " + ", ".join(
            sorted(unmapped_labels)
        )
        issues.append(
            {
                "record_key": image["record_key"],
                "dataset_id": image["dataset_id"],
                "issue": image["exclusion_reason"],
                "path": image["source_path"],
            }
        )

issues_df = pd.DataFrame(issues)
if issues_df.empty:
    print("No input validation issues found.")
else:
    issue_summary = issues_df.groupby(["dataset_id", "issue"]).size()
    display(issue_summary.reset_index(name="count").style.hide(axis="index"))

# %% [markdown]
# ## 3. Normalize the target class
#
# Apply the source-specific rules and rename every mapped target box to `football`.
# Images containing `football-csdy0: otherball` or `football-tnb17: other` are
# excluded completely because those labels mix target and non-target balls.
#
# Ordinary removal labels delete only their boxes. A mixed image remains when a
# mapped football box survives. An annotated image made empty by filtering is
# excluded so it cannot become a false background; an image that was originally
# annotation-free remains as a genuine background. The table reports every outcome
# by source dataset.

# %%
for image in image_rows:
    if image["status"] != "pending":
        continue
    config = config_by_dataset[image["dataset_id"]]
    raw_labels = {
        annotation["category_name"] for annotation in image["raw_annotations"]
    }
    ambiguous_labels = raw_labels & config["exclude_image_labels"]
    if ambiguous_labels:
        image["status"] = "excluded_ambiguous"
        image["exclusion_reason"] = "image contains ambiguous labels: " + ", ".join(
            sorted(ambiguous_labels)
        )
        continue

    image["mapped_annotations"] = [
        {
            "bbox": [float(value) for value in annotation["bbox"]],
            "iscrowd": int(annotation["iscrowd"]),
        }
        for annotation in image["raw_annotations"]
        if annotation["category_name"] in config["map_labels"]
    ]
    if image["original_annotation_count"] and not image["mapped_annotations"]:
        image["status"] = "empty_after_filtering"
        image["exclusion_reason"] = "all original annotations were removed"
    else:
        image["status"] = "ready_for_hashing"

filter_summary = (
    pd.DataFrame(image_rows)
    .groupby(["dataset_id", "status"], sort=False)
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
display(filter_summary.style.hide(axis="index"))

# %% [markdown]
# ## 4. Remove exact duplicates
#
# Hash each candidate from its decoded RGB width and height, encoded as fixed-width
# integers, followed by all RGB pixel bytes. This identifies equal displayed pixels
# even when JPEG metadata or compression bytes differ. No EXIF rotation, resize,
# crop, or other image transformation is applied.
#
# Within an equal-pixel group, prefer an annotated record over an annotation-free
# copy. When mapped box sets agree, deterministic source order selects the keeper.
# When nonempty box sets disagree, exclude and report the whole group rather than
# silently choosing or merging annotations. The summary distinguishes retained
# images, removed copies, preferred annotated records, and annotation conflicts.


# %%
def decoded_pixel_sha256(image: Image.Image) -> str:
    """
    Hash image dimensions and decoded RGB pixels with SHA-256.
    """
    rgb = image.convert("RGB")
    header = struct.pack(">II", rgb.width, rgb.height)
    return hashlib.sha256(header + rgb.tobytes()).hexdigest()


def hash_image_file(path: str | Path) -> str:
    """
    Return the dimension-aware decoded-pixel hash for an image file.
    """
    with Image.open(path) as image:
        image.load()
        return decoded_pixel_sha256(image)


def annotation_signature(image: dict) -> tuple:
    """
    Return a stable comparison key for an image's mapped boxes.
    """
    boxes = (
        tuple(round(value, 6) for value in annotation["bbox"])
        + (annotation["iscrowd"],)
        for annotation in image["mapped_annotations"]
    )
    return tuple(sorted(boxes))


def duplicate_sort_key(image: dict) -> tuple:
    """
    Return the deterministic precedence key for a duplicate record.
    """
    split_order = {"train": 0, "val": 1, "test": 2}
    source_root = config_by_dataset[image["dataset_id"]]["source_root"]
    relative_path = Path(image["source_path"]).relative_to(source_root)
    return (
        image["source_order"],
        split_order.get(image["original_split"], 99),
        str(relative_path),
        str(image["source_image_id"]),
    )


def choose_duplicate_keeper(records: list[dict]) -> tuple[dict | None, str]:
    """
    Choose an annotation-aware keeper or report conflicting boxes.
    """
    ordered = sorted(records, key=duplicate_sort_key)
    nonempty_signatures = {
        annotation_signature(record)
        for record in ordered
        if record["mapped_annotations"]
    }
    if len(nonempty_signatures) > 1:
        return None, "nonempty mapped annotations disagree"
    annotated = [record for record in ordered if record["mapped_annotations"]]
    return (annotated[0] if annotated else ordered[0]), ""


hashable_images = [
    image for image in image_rows if image["status"] == "ready_for_hashing"
]
for image in hashable_images:
    image["pixel_sha256"] = hash_image_file(image["source_path"])

duplicate_groups = defaultdict(list)
for image in hashable_images:
    duplicate_groups[image["pixel_sha256"]].append(image)

preferred_annotated_records = 0
conflicting_duplicate_groups = 0
for pixel_sha256, records in duplicate_groups.items():
    if len(records) > 1:
        with Image.open(records[0]["source_path"]) as reference:
            reference_rgb = reference.convert("RGB")
            reference_pixels = reference_rgb.tobytes()
            reference_size = reference_rgb.size
        for record in records[1:]:
            with Image.open(record["source_path"]) as candidate:
                candidate_rgb = candidate.convert("RGB")
                unequal = candidate_rgb.size != reference_size
                unequal |= candidate_rgb.tobytes() != reference_pixels
                if unequal:
                    raise RuntimeError(
                        "One pixel SHA-256 digest corresponds to unequal decoded "
                        f"pixels: {pixel_sha256}"
                    )

    keeper, conflict_reason = choose_duplicate_keeper(records)
    if keeper is None:
        conflicting_duplicate_groups += 1
        for record in records:
            record["status"] = "duplicate_annotation_conflict"
            record["exclusion_reason"] = conflict_reason
        continue

    keeper["status"] = "retained"
    first_record = sorted(records, key=duplicate_sort_key)[0]
    if len(records) > 1 and first_record is not keeper:
        preferred_annotated_records += 1
    for record in records:
        if record is not keeper:
            record["status"] = "exact_duplicate"
            record["exclusion_reason"] = f"kept {keeper['record_key']}"

retained_images = [image for image in image_rows if image["status"] == "retained"]
deduplication_summary = pd.DataFrame(
    {
        "metric": [
            "hash candidates",
            "unique retained images",
            "removed exact copies",
            "preferred annotated records",
            "excluded conflicting groups",
        ],
        "count": [
            len(hashable_images),
            len(retained_images),
            sum(image["status"] == "exact_duplicate" for image in image_rows),
            preferred_annotated_records,
            conflicting_duplicate_groups,
        ],
    }
)
display(deduplication_summary.style.hide(axis="index"))

# %% [markdown]
# ## 5. Assign grouped train, validation, and test splits
#
# Pass the raw `dataset_id` values to `GroupShuffleSplit` as groups, ensuring that
# every source dataset appears in exactly one final split. Because split sizes refer
# to group counts rather than image counts, a single draw can be far from the target
# ratio when sources have different sizes.
#
# Generate 100 deterministic train/held-out candidates and select the first one
# closest to 70%/30% by retained-image count. Repeat within the held-out records to
# approach 15% validation and 15% test. Both searches use `random_state=42`. The
# displayed tables report the achieved image ratios and the final group assignment.


# %%
def select_group_split(
    frame: pd.DataFrame,
    test_size: float,
    total_image_count: int,
    left_target: float,
    right_target: float,
) -> tuple[list[int], list[int], float, int]:
    """
    Select the grouped candidate closest to target image ratios.
    """
    splitter = GroupShuffleSplit(
        n_splits=N_SPLIT_CANDIDATES,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )
    best = None
    for number, (left, right) in enumerate(
        splitter.split(frame, groups=frame["group_id"])
    ):
        score = abs(len(left) / total_image_count - left_target)
        score += abs(len(right) / total_image_count - right_target)
        candidate = (score, number, left.tolist(), right.tolist())
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("GroupShuffleSplit did not generate a candidate.")
    score, number, left, right = best
    return left, right, score, number


retained_df = pd.DataFrame(retained_images).reset_index(drop=True)
if retained_df["group_id"].nunique() < 4:
    raise RuntimeError(
        "At least four nonempty dataset_id groups are required for grouped splits."
    )

total_retained = len(retained_df)
train_positions, heldout_positions, train_score, train_candidate = select_group_split(
    retained_df, 0.30, total_retained, 0.70, 0.30
)
heldout_df = retained_df.iloc[heldout_positions].reset_index(drop=True)
val_positions, test_positions, heldout_score, heldout_candidate = select_group_split(
    heldout_df, 0.50, total_retained, 0.15, 0.15
)

train_keys = set(retained_df.iloc[train_positions]["record_key"])
val_keys = set(heldout_df.iloc[val_positions]["record_key"])
test_keys = set(heldout_df.iloc[test_positions]["record_key"])
for image in retained_images:
    if image["record_key"] in train_keys:
        image["final_split"] = "train"
    elif image["record_key"] in val_keys:
        image["final_split"] = "val"
    elif image["record_key"] in test_keys:
        image["final_split"] = "test"

split_df = pd.DataFrame(retained_images)
split_summary = (
    split_df.groupby("final_split")
    .agg(images=("record_key", "size"), groups=("group_id", "nunique"))
    .reindex(["train", "val", "test"])
    .reset_index()
)
split_summary["image_ratio"] = split_summary["images"] / total_retained
display(split_summary.style.hide(axis="index").format({"image_ratio": "{:.2%}"}))

group_assignment = (
    split_df[["group_id", "final_split"]]
    .drop_duplicates()
    .sort_values(["final_split", "group_id"])
)
display(group_assignment.style.hide(axis="index"))
print(
    f"Selected train candidate {train_candidate}, score {train_score:.6f}; "
    f"held-out candidate {heldout_candidate}, score {heldout_score:.6f}."
)

# %% [markdown]
# ### 5.1 Run focused logic checks
#
# These assertions cover the error-prone decisions without duplicating the complete
# notebook: dimension-aware hashing, preference for an annotated duplicate,
# rejection of conflicting annotations, reproducible candidate selection, and group
# isolation. The earlier inventory summaries exercise the class rules on real data.

# %%
assert decoded_pixel_sha256(Image.new("RGB", (1, 2))) != decoded_pixel_sha256(
    Image.new("RGB", (2, 1))
)

test_background = {
    **retained_images[0],
    "mapped_annotations": [],
    "source_order": 0,
}
test_annotated = {
    **retained_images[0],
    "mapped_annotations": [{"bbox": [1.0, 2.0, 3.0, 4.0], "iscrowd": 0}],
    "source_order": 1,
}
keeper, reason = choose_duplicate_keeper([test_background, test_annotated])
assert keeper is test_annotated and not reason
test_conflict = {
    **test_annotated,
    "mapped_annotations": [{"bbox": [5.0, 6.0, 7.0, 8.0], "iscrowd": 0}],
}
keeper, reason = choose_duplicate_keeper([test_annotated, test_conflict])
assert keeper is None and reason

repeat_split = select_group_split(
    retained_df,
    0.30,
    total_retained,
    0.70,
    0.30,
)
assert repeat_split == (
    train_positions,
    heldout_positions,
    train_score,
    train_candidate,
)
assert group_assignment.groupby("group_id")["final_split"].nunique().max() == 1
print("Focused logic checks passed.")

# %% [markdown]
# ## 6. Export matching COCO and YOLO datasets
#
# Build one Datumaro dataset so COCO and YOLO receive the same images, boxes, class,
# and split assignments. The pixel digest is the item ID and therefore the image
# filename in both formats. Output directories encode the final counts as
# `<format>_football_<train>_<val>_<test>`.
#
# Media export attempts a hardlink first to avoid duplicating image bytes and falls
# back to a byte-identical copy when linking is unavailable. Existing destinations
# stop the build to prevent accidental replacement or mixed outputs. Displayed paths
# are absolute so external DATA_ROOT and OUTPUT_ROOT overrides remain valid.

# %%
n_train = len(split_df[split_df["final_split"] == "train"])
n_val = len(split_df[split_df["final_split"] == "val"])
n_test = len(split_df[split_df["final_split"] == "test"])
dataset_suffix = f"football_{n_train}_{n_val}_{n_test}"
coco_dataset_dir = COMPOSED_ROOT / f"coco_{dataset_suffix}"
yolo_dataset_dir = COMPOSED_ROOT / f"yolo_{dataset_suffix}"

existing = [path for path in (coco_dataset_dir, yolo_dataset_dir) if path.exists()]
if existing:
    paths = ", ".join(str(path) for path in existing)
    raise FileExistsError(f"Export destination already exists: {paths}")

datumaro_items = []
for image in retained_images:
    annotations = [
        dm.Bbox(
            *annotation["bbox"],
            label=0,
            attributes={"is_crowd": bool(annotation["iscrowd"])},
        )
        for annotation in image["mapped_annotations"]
    ]
    datumaro_items.append(
        dm.DatasetItem(
            id=image["pixel_sha256"],
            subset=image["final_split"],
            media=dm.Image.from_file(image["source_path"]),
            annotations=annotations,
        )
    )

football_dataset = dm.Dataset.from_iterable(
    datumaro_items,
    categories=["football"],
)
with prefer_hardlinked_datumaro_media():
    football_dataset.export(
        str(coco_dataset_dir),
        format=DatumaroExpFmt.COCO_INSTANCES,
        save_media=True,
        reindex=True,
    )
    football_dataset.export(
        str(yolo_dataset_dir),
        format=DatumaroExpFmt.YOLO_ULTRALYTICS,
        save_media=True,
    )

print(f"COCO export: {coco_dataset_dir}")
print(f"YOLO export: {yolo_dataset_dir}")

# %% [markdown]
# ### 6.1 Save the source manifest
#
# Write one CSV row for every inventoried source image, including images excluded
# before hashing and copies removed during deduplication. Each row records its
# pixel hash when available, source identity, final status and reason, split, output
# paths, and whether exported media was hardlinked or copied. This makes the final
# dataset traceable without creating several intermediate reports.


# %%
def exported_media_method(source: Path, destination: Path) -> str:
    """
    Identify exported media as a hardlink or byte-identical copy.
    """
    if os.path.samefile(source, destination):
        return "hardlink"
    if source.read_bytes() == destination.read_bytes():
        return "copy"
    return "different"


for image in retained_images:
    filename = image["pixel_sha256"] + Path(image["source_path"]).suffix
    coco_path = coco_dataset_dir / "images" / image["final_split"] / filename
    yolo_path = yolo_dataset_dir / "images" / image["final_split"] / filename
    image["coco_output_path"] = str(coco_path)
    image["yolo_output_path"] = str(yolo_path)
    image["coco_media_method"] = exported_media_method(
        Path(image["source_path"]),
        coco_path,
    )
    image["yolo_media_method"] = exported_media_method(
        Path(image["source_path"]),
        yolo_path,
    )

manifest_columns = [
    "pixel_sha256",
    "source_path",
    "original_name",
    "dataset_id",
    "group_id",
    "original_split",
    "final_split",
    "status",
    "exclusion_reason",
    "coco_output_path",
    "yolo_output_path",
    "coco_media_method",
    "yolo_media_method",
]
manifest_df = pd.DataFrame(image_rows)[manifest_columns]
BUILD_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
manifest_path = BUILD_OUTPUT_ROOT / "football_dataset_manifest.csv"
manifest_df.to_csv(manifest_path, index=False)
print(f"Manifest: {manifest_path}")

# %% [markdown]
# ## 7. Validate the final exports
#
# Validate the generated artifacts rather than relying only on the in-memory
# dataset. The checks require:
#
# - one COCO category named `football` and YOLO class index `0`;
# - unique retained pixel hashes and one final split per `dataset_id`;
# - matching image and label filenames in both formats;
# - valid COCO image/category references and normalized YOLO boxes;
# - image counts that match the output directory name; and
# - hardlinked or byte-identical media files.
#
# The final table reconciles the source, exclusion, duplicate, and retained counts
# and reports all artifact locations relative to the project root. Any failed check
# stops the build because the exported dataset would not be trustworthy.

# %%
assert len({image["pixel_sha256"] for image in retained_images}) == total_retained
assert {image["final_split"] for image in retained_images} == {
    "train",
    "val",
    "test",
}
assert n_train + n_val + n_test == total_retained
assert dataset_suffix == f"football_{n_train}_{n_val}_{n_test}"

expected_names = {
    split: {
        image["pixel_sha256"] + Path(image["source_path"]).suffix
        for image in retained_images
        if image["final_split"] == split
    }
    for split in ("train", "val", "test")
}
for split in ("train", "val", "test"):
    annotation_path = coco_dataset_dir / "annotations" / f"instances_{split}.json"
    with annotation_path.open(encoding="utf-8") as file:
        exported_coco = json.load(file)
    assert exported_coco["categories"] == [
        {"id": 1, "name": "football", "supercategory": ""}
    ]
    image_ids = {image["id"] for image in exported_coco["images"]}
    assert all(
        annotation["image_id"] in image_ids and annotation["category_id"] == 1
        for annotation in exported_coco["annotations"]
    )
    coco_names = {path.name for path in (coco_dataset_dir / "images" / split).iterdir()}
    yolo_names = {path.name for path in (yolo_dataset_dir / "images" / split).iterdir()}
    assert coco_names == expected_names[split]
    assert yolo_names == expected_names[split]

    label_paths = list((yolo_dataset_dir / "labels" / split).glob("*.txt"))
    assert {path.stem for path in label_paths} == {
        Path(name).stem for name in expected_names[split]
    }
    for label_path in label_paths:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            assert len(fields) == 5 and fields[0] == "0"
            assert all(0 <= float(value) <= 1 for value in fields[1:])

assert "football" in (yolo_dataset_dir / "data.yaml").read_text(encoding="utf-8")
assert group_assignment.groupby("group_id")["final_split"].nunique().max() == 1
assert set(manifest_df["coco_media_method"]) <= {"", "hardlink", "copy"}
assert set(manifest_df["yolo_media_method"]) <= {"", "hardlink", "copy"}

final_summary = pd.DataFrame(
    {
        "metric": [
            "source images",
            "retained images",
            "excluded ambiguous images",
            "removed exact copies",
            "duplicate annotation conflicts",
            "COCO export",
            "YOLO export",
            "manifest",
        ],
        "result": [
            len(image_rows),
            total_retained,
            sum(image["status"] == "excluded_ambiguous" for image in image_rows),
            sum(image["status"] == "exact_duplicate" for image in image_rows),
            sum(
                image["status"] == "duplicate_annotation_conflict"
                for image in image_rows
            ),
            str(coco_dataset_dir),
            str(yolo_dataset_dir),
            str(manifest_path),
        ],
    }
)
display(final_summary.style.hide(axis="index"))
print("Final COCO and YOLO validation passed.")
