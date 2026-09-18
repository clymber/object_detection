# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,title,authors,-kernelspec,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %%
"""
Builds a larger basketball dataset. This dataset uses datasets that other than
basketball, such as tennis, as background images.
"""
# %load_ext autoreload
# %autoreload 2
# %aimport -csv, -textwrap, -functools, -IPython, -ultralytics, -pandas, -datumaro
# %aimport -random

from detection_common import configure_stdio_relative_path

from dataset_builder.config import DATA_ROOT, OUTPUT_ROOT, WORKSPACE_ROOT

# Display project paths relative to project root directory for consistency.
configure_stdio_relative_path(WORKSPACE_ROOT)

# %%
import hashlib
import math
from collections import defaultdict
from itertools import islice
from pathlib import Path

import datumaro as dm
import pandas as pd
from detection_common.utils import json_io
from detection_common.utils.filepath import dir_tree
from detection_common.utils.image import image_content_digest
from IPython.display import display

from dataset_builder import (
    prefer_hardlinked_datumaro_media,
    summarize_coco_datasets,
    summarize_datumaro_label_counts,
)
from dataset_builder import roboflow as rf_platform
from dataset_builder.datumaro import ExportFormat as DatumaroExpFmt
from dataset_builder.datumaro import ImportFormat as DatumaroImpFmt
from dataset_builder.roboflow import RoboflowFormat as rf_format

# %% [markdown]
# ## 1. Download source datasets
#
# The datasets downloaded from Roboflow have data leakage issues, such as basketball
# in similar scenes existing in all training, validation and test splits. So our
# target dataset will be a combination of 3 different basketball datasets, and use a
# tennis dataset as background images. The target dataset will be exported to both
# COCO instance and YOLO format. 
#

# %%
trn_set_url = "https://universe.roboflow.com/cricket-qnb5l/basketball-xil7x/dataset/1"
val_set_url = "https://universe.roboflow.com/aidatasets-qwszk/ball-tracker-jpjs8/dataset/4"
tst_set_url = "https://universe.roboflow.com/detectron2-z9hbp/basketball-lokg2/dataset/1"
bg_tnns_url = "https://universe.roboflow.com/test-06r5e/tennis-racket-r6mgq/dataset/4"

train_rf_dataset = rf_platform.download_by_url(trn_set_url, rf_format.COCO)
val_rf_dataset = rf_platform.download_by_url(val_set_url, rf_format.COCO)
test_rf_dataset = rf_platform.download_by_url(tst_set_url, rf_format.COCO)
backg_rf_dataset = rf_platform.download_by_url(bg_tnns_url, rf_format.COCO)

# %% [markdown]
# ### 1.1 Source dataset structure

# %%
print(f"Basketball dataset: {train_rf_dataset.location}")
print(dir_tree(train_rf_dataset.location))

# %%
print(f"tennis dataset: {backg_rf_dataset.location}")
print(dir_tree(backg_rf_dataset.location))

# %% [markdown]
# ### 1.2 Splits and annotations of source dataset

# %%
summary = summarize_coco_datasets(
    [
        train_rf_dataset.location,
        val_rf_dataset.location,
        test_rf_dataset.location,
        backg_rf_dataset.location,
    ]
)
display(summary.style.hide(axis="index"))

# %% [markdown]
#
# The COCO files in the original `ball-tracker-jpjs8` dataset have the same category
# name twice:
# ```json
#   "categories": [{
#       "id": 0,
#       "name": "ball",
#       "supercategory": "none"
#   }, {
#       "id": 1,
#       "name": "ball",
#       "supercategory": "ball"
#   }]
# ```
#
# But Datumaro requires unique categrory names, so we'll have to remove the duplicate
# category name when we import the dataset into Datumaro. Fortunately, all annotations
# reference category ID 1; none reference ID 0, so it's save to remove category ID 0.

# %%
for annotation_path in Path(val_rf_dataset.location).rglob("*.coco.json"):
    coco_data = json_io.read_json(annotation_path)

    if not any(category["id"] == 0 for category in coco_data["categories"]):
        continue

    coco_data["categories"] = [
        category
        for category in coco_data["categories"]
        if category["id"] != 0
    ]

    json_io.write_json(annotation_path, coco_data)


# %%
import_format = DatumaroImpFmt.ROBOFLOW_COCO
train_dm_dataset = dm.Dataset.import_from(train_rf_dataset.location, import_format)
test_dm_dataset = dm.Dataset.import_from(test_rf_dataset.location, import_format)
val_dm_dataset = dm.Dataset.import_from(val_rf_dataset.location, import_format)
backg_dm_dataset = dm.Dataset.import_from(backg_rf_dataset.location, import_format)

# %% [markdown]
# Roboflow uses category ID 0, `ball-person-rim`, as an umbrella category for the
# actual `ball`, `human`, and `rim` categories. Datumaro reserves ID 0 for the
# background class, so it warns that an annotation using that ID would have no label.
# None of this dataset's annotations use ID 0, however, so the warning does not
# indicate any lost labels and can safely be ignored.
#

# %% [markdown]
# ## 2. Build a new dataset
#
# The new dtaset will be a combination of the 3 basketball datasets, and will include
# background images from the tennis dataset if the percentages of background images are
# less than 30%.
#
# The new dataset will contain basketball images with basketball annotations, and the
# background images without basketball annotations. The other annotations will be
# removed. Eventually, the new dataset will be exported to both coco instance and
# ultrlytics YOLO format.

# %% [markdown]
# ### 2.1 Remove non-basketball annotations from basketball dataset
#
# Remap the labels to keep only the `basketball` label, and remove all the others.

# %%
train_dm_dataset = train_dm_dataset.transform(
    "remap_labels",
    mapping={"ball": "basketball"},
    default="delete",
)

val_dm_dataset = val_dm_dataset.transform(
    "remap_labels",
    mapping={"ball": "basketball"},
    default="delete",
)

test_dm_dataset = test_dm_dataset.transform(
    "remap_labels",
    mapping={"basketball": "basketball"},
    default="delete",
)

print("After removing non-basketball annotations:")
label_counts = pd.concat(
    [
        summarize_datumaro_label_counts(dataset).assign(dataset=dataset_name)
        for dataset_name, dataset in (
            ("train", train_dm_dataset),
            ("validation", val_dm_dataset),
            ("test", test_dm_dataset),
        )
    ],
    ignore_index=True,
)
display(
    label_counts[["dataset", "label", "annotation_count"]].style.hide(axis="index")
)

# %% [markdown]
# ### 2.2 Combine a new basketball dataset
#
# New training set: combine train, val and test splits of `basketball-xil7x`.
# New validation set: combine train, val and test splits of `ball-tracker-jpjs8`.
# New test set: combine train, val and test splits of `basketball-lokg2`.

# %%
def hash_item_id(source: str, item: dm.DatasetItem) -> str:
    """
    Return a temporary flat ID derived from an item's source identity.
    """
    source_key = f"{source}:{item.subset}:{item.id}"
    return hashlib.sha1(
        source_key.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]


def source_record(
    item: dm.DatasetItem,
    *,
    source_url: str,
    source_order: int,
    source_root: str | Path,
    target_split: str,
    annotations: list | None = None,
) -> dict:
    """Preserve source identity and the target split for one candidate image."""
    source_root = Path(source_root).resolve()
    source_path = Path(item.media.path).resolve()
    relative_path = source_path.relative_to(source_root)
    candidate = item.wrap(
        id=hash_item_id(source_url, item),
        subset=target_split,
        annotations=item.annotations if annotations is None else annotations,
    )
    return {
        "record_key": f"{source_url}:{item.subset}:{item.id}",
        "source_url": source_url,
        "source_order": source_order,
        "source_subset": item.subset,
        "source_item_id": str(item.id),
        "source_path": str(source_path),
        "source_relative_path": str(relative_path),
        "target_split": target_split,
        "item": candidate,
        "pixel_sha256": "",
        "annotation_signature": (),
        "status": "pending",
        "keeper_record_key": "",
        "resolution_reason": "",
    }


source_specs = (
    ("train", trn_set_url, 0, train_rf_dataset.location, train_dm_dataset),
    ("val", val_set_url, 1, val_rf_dataset.location, val_dm_dataset),
    ("test", tst_set_url, 2, test_rf_dataset.location, test_dm_dataset),
)
candidate_records = [
    source_record(
        item,
        source_url=source_url,
        source_order=source_order,
        source_root=source_root,
        target_split=target_split,
    )
    for target_split, source_url, source_order, source_root, dataset in source_specs
    for item in dataset
]
basketball_dm_dataset = dm.Dataset.from_iterable(
    [record["item"] for record in candidate_records],
    categories=train_dm_dataset.categories(),
)


# %% [markdown]
# ### 2.3 Annotated images and background images

# %%
def summary_annotated_vs_background(dataset: dm.Dataset) -> pd.DataFrame:
    """
    Summarize annotated and background image counts by subset.
    """
    counts_by_subset = {}
    for item in dataset:
        counts = counts_by_subset.setdefault(
            item.subset,
            {"annotated_images": 0, "background_images": 0},
        )
        image_type = "annotated_images" if item.annotations else "background_images"
        counts[image_type] += 1

    statistics = pd.DataFrame.from_dict(
        counts_by_subset,
        orient="index",
    ).rename_axis("subset")
    statistics["total_images"] = statistics.sum(axis="columns")
    statistics.loc["total"] = statistics.sum(axis="index")
    statistics["background_percentage"] = (
        100 * statistics["background_images"] / statistics["total_images"]
    ).round(2)
    return statistics


image_statistics = summary_annotated_vs_background(basketball_dm_dataset)
display(
    image_statistics.reset_index().style.hide(axis="index").format(
        {
            "background_percentage": "{:.2f}%",
        }
    )
)

# %% [markdown]
# ### 2.4 Collect background images from tennis dataset
#
# Target percentages of background images in training, validation and test splits are
# 30% each. Extract the images from tennis dataset, clear the annotations, and add them
# to each split of the basketball dataset.

# %%
background_ratio = 0.3
subset_statistics = image_statistics.drop(index="total")
backgrounds_needed_by_subset = {}
for subset, statistics in subset_statistics.iterrows():
    background_deficit = (
        background_ratio * statistics["total_images"]
        - statistics["background_images"]
    )
    backgrounds_needed_by_subset[subset] = max(
        0,
        math.ceil(background_deficit / (1 - background_ratio)),
    )

total_backgrounds_needed = sum(backgrounds_needed_by_subset.values())
if total_backgrounds_needed > len(backg_dm_dataset):
    raise ValueError("Not sufficient background images from tennis dataset.")

background_records = []
tennis_items = iter(backg_dm_dataset)
for target_subset, count in backgrounds_needed_by_subset.items():
    for item in islice(tennis_items, count):
        background_records.append(
            source_record(
                item,
                source_url=bg_tnns_url,
                source_order=3,
                source_root=backg_rf_dataset.location,
                target_split=target_subset,
                annotations=[],
            )
        )

candidate_records.extend(background_records)
basketball_dm_dataset.update(record["item"] for record in background_records)

image_statistics = summary_annotated_vs_background(basketball_dm_dataset)
display(
    image_statistics.reset_index().style.hide(axis="index").format(
        {"background_percentage": "{:.2f}%"}
    )
)

# %% [markdown]
# ## 3. Remove exact decoded-pixel duplicates
#
# Hash EXIF-normalized RGBA pixels and dimensions so metadata or compression changes
# cannot hide duplicate content. Prefer annotated records, then test over validation
# over training for equivalent cross-split records. Annotation coordinates within
# 0.5 pixel are treated as equivalent; larger disagreements stop the build.

# %%
ANNOTATION_TOLERANCE = 0.5
EXPECTED_CANDIDATE_COUNTS = {"train": 11501, "val": 1156, "test": 1395}
EXPECTED_RETAINED_COUNTS = {"train": 11394, "val": 1155, "test": 1395}
EXPECTED_DUPLICATES_REMOVED = 108


def annotation_signature(item: dm.DatasetItem) -> tuple:
    """Return a stable signature for the basketball boxes on one image."""
    signature = []
    for annotation in item.annotations:
        if not isinstance(annotation, dm.Bbox):
            raise TypeError(
                f"Unsupported annotation type on {item.id}: "
                f"{type(annotation).__name__}"
            )
        signature.append(
            (
                float(annotation.x),
                float(annotation.y),
                float(annotation.w),
                float(annotation.h),
                int(annotation.label),
                bool(annotation.attributes.get("is_crowd", False)),
            )
        )
    return tuple(sorted(signature))


def annotations_equivalent(
    left: tuple,
    right: tuple,
    *,
    tolerance: float = ANNOTATION_TOLERANCE,
) -> bool:
    """Compare equally ordered box signatures with a coordinate tolerance."""
    if len(left) != len(right):
        return False
    for left_box, right_box in zip(left, right, strict=True):
        if left_box[4:] != right_box[4:]:
            return False
        if any(
            abs(left_value - right_value) > tolerance
            for left_value, right_value in zip(
                left_box[:4],
                right_box[:4],
                strict=True,
            )
        ):
            return False
    return True


def duplicate_sort_key(record: dict) -> tuple:
    """Return the deterministic precedence key for equivalent records."""
    split_order = {"test": 0, "val": 1, "train": 2}
    source_subset_order = {"train": 0, "val": 1, "valid": 1, "test": 2}
    return (
        split_order[record["target_split"]],
        record["source_order"],
        source_subset_order.get(record["source_subset"], 99),
        record["source_relative_path"],
        record["source_item_id"],
    )


def choose_duplicate_keeper(records: list[dict]) -> dict:
    """Choose one safe keeper from an equal-pixel group."""
    annotated = [record for record in records if record["annotation_signature"]]
    candidates = annotated or records
    reference = candidates[0]["annotation_signature"]
    disagreements = [
        record
        for record in candidates[1:]
        if not annotations_equivalent(reference, record["annotation_signature"])
    ]
    if disagreements:
        record_keys = ", ".join(record["record_key"] for record in candidates)
        raise ValueError(
            "Equal-pixel records have annotations that differ by more than "
            f"{ANNOTATION_TOLERANCE} pixel: {record_keys}"
        )
    return min(candidates, key=duplicate_sort_key)


def duplicate_resolution_reason(records: list[dict], keeper: dict) -> str:
    """Describe the precedence rule that selected a duplicate keeper."""
    annotation_states = {bool(record["annotation_signature"]) for record in records}
    if len(annotation_states) > 1:
        return "preferred annotated record"
    if len({record["target_split"] for record in records}) > 1:
        return f"preferred {keeper['target_split']} split"
    if len({record["annotation_signature"] for record in records}) > 1:
        return f"annotations equivalent within {ANNOTATION_TOLERANCE} pixel"
    return "deterministic source identity order"


candidate_counts = {
    split: sum(record["target_split"] == split for record in candidate_records)
    for split in ("train", "val", "test")
}
assert candidate_counts == EXPECTED_CANDIDATE_COUNTS

for record in candidate_records:
    record["pixel_sha256"] = image_content_digest(record["source_path"])
    record["annotation_signature"] = annotation_signature(record["item"])

duplicate_groups = defaultdict(list)
for record in candidate_records:
    duplicate_groups[record["pixel_sha256"]].append(record)

duplicate_group_count = 0
for records in duplicate_groups.values():
    keeper = choose_duplicate_keeper(records)
    keeper["status"] = "retained"
    if len(records) == 1:
        continue

    duplicate_group_count += 1
    resolution_reason = duplicate_resolution_reason(records, keeper)
    keeper["resolution_reason"] = resolution_reason
    for record in records:
        if record is keeper:
            continue
        record["status"] = "exact_duplicate"
        record["keeper_record_key"] = keeper["record_key"]
        record["resolution_reason"] = resolution_reason

retained_records = [
    record for record in candidate_records if record["status"] == "retained"
]
removed_duplicate_count = sum(
    record["status"] == "exact_duplicate" for record in candidate_records
)
retained_counts = {
    split: sum(record["target_split"] == split for record in retained_records)
    for split in ("train", "val", "test")
}
assert duplicate_group_count == EXPECTED_DUPLICATES_REMOVED
assert removed_duplicate_count == EXPECTED_DUPLICATES_REMOVED
assert retained_counts == EXPECTED_RETAINED_COUNTS
assert len({record["pixel_sha256"] for record in retained_records}) == len(
    retained_records
)

basketball_dm_dataset = dm.Dataset.from_iterable(
    [
        record["item"].wrap(
            id=record["pixel_sha256"],
            subset=record["target_split"],
        )
        for record in retained_records
    ],
    categories=train_dm_dataset.categories(),
)

deduplication_summary = pd.DataFrame(
    {
        "metric": [
            "candidate images",
            "duplicate groups",
            "removed duplicate records",
            "retained images",
        ],
        "count": [
            len(candidate_records),
            duplicate_group_count,
            removed_duplicate_count,
            len(retained_records),
        ],
    }
)
display(deduplication_summary.style.hide(axis="index"))


# %% [markdown]
# ### 3.1 Run focused keeper-policy checks

# %%
assert annotations_equivalent(
    ((0.0, 0.0, 10.0, 10.0, 0, False),),
    ((0.0, 0.0, 10.5, 9.5, 0, False),),
)
assert not annotations_equivalent(
    ((0.0, 0.0, 10.0, 10.0, 0, False),),
    ((0.0, 0.0, 10.51, 10.0, 0, False),),
)
assert not annotations_equivalent(
    ((0.0, 0.0, 10.0, 10.0, 0, False),),
    (),
)

test_record_base = {
    "record_key": "base",
    "source_order": 0,
    "source_subset": "train",
    "source_item_id": "1",
    "source_relative_path": "b.jpg",
    "target_split": "train",
    "annotation_signature": (),
}
test_background = dict(test_record_base)
test_annotated = {
    **test_record_base,
    "record_key": "annotated",
    "source_relative_path": "c.jpg",
    "annotation_signature": ((0.0, 0.0, 10.0, 10.0, 0, False),),
}
assert choose_duplicate_keeper([test_background, test_annotated]) is test_annotated

test_held_out = {
    **test_record_base,
    "record_key": "test",
    "target_split": "test",
}
assert choose_duplicate_keeper([test_background, test_held_out]) is test_held_out

test_lexical_first = {
    **test_record_base,
    "record_key": "first",
    "source_relative_path": "a.jpg",
}
assert (
    choose_duplicate_keeper([test_background, test_lexical_first]) is test_lexical_first
)

test_conflict = {
    **test_annotated,
    "record_key": "conflict",
    "annotation_signature": ((0.0, 0.0, 10.51, 10.0, 0, False),),
}
try:
    choose_duplicate_keeper([test_annotated, test_conflict])
except ValueError:
    pass
else:
    raise AssertionError("Annotation conflicts beyond the tolerance must fail")

print("Focused keeper-policy checks passed.")


# %% [markdown]
# ## 4. Export the canonical basketball dataset
#
# Export matching COCO and YOLO datasets under stable names. Existing destinations
# stop the build so stale artifacts cannot be mistaken for a successful rebuild.


# %%
n_train = len(basketball_dm_dataset.get_subset("train"))
n_val = len(basketball_dm_dataset.get_subset("val"))
n_test = len(basketball_dm_dataset.get_subset("test"))
export_counts = {"train": n_train, "val": n_val, "test": n_test}
print(f"Reconstructed split counts: {export_counts}")
assert export_counts == EXPECTED_RETAINED_COUNTS, (
    f"Unexpected reconstructed split counts: {export_counts}"
)

coco_dataset_dir = DATA_ROOT / "composed" / "coco_basketball"
yolo_dataset_dir = DATA_ROOT / "composed" / "yolo_basketball"
existing_destinations = [
    path for path in (coco_dataset_dir, yolo_dataset_dir) if path.exists()
]
if existing_destinations:
    paths = ", ".join(str(path) for path in existing_destinations)
    raise FileExistsError(f"Export destination already exists: {paths}")

with prefer_hardlinked_datumaro_media():
    basketball_dm_dataset.export(
        str(coco_dataset_dir),
        format=DatumaroExpFmt.COCO_INSTANCES,
        save_media=True,
        reindex=True,
    )
    basketball_dm_dataset.export(
        str(yolo_dataset_dir),
        format=DatumaroExpFmt.YOLO_ULTRALYTICS,
        save_media=True,
    )

print(f"COCO dataset directory: {coco_dataset_dir}")
print(f"YOLO dataset directory: {yolo_dataset_dir}")
print(dir_tree(coco_dataset_dir))


# %% [markdown]
# ### 4.1 Save the source audit manifest
#
# Record every source candidate, including excluded copies, so each exported image
# and every duplicate decision can be traced back to its source.

# %%
manifest_columns = [
    "pixel_sha256",
    "record_key",
    "source_url",
    "source_subset",
    "source_item_id",
    "source_path",
    "source_relative_path",
    "target_split",
    "annotation_signature",
    "status",
    "keeper_record_key",
    "resolution_reason",
]
manifest_df = pd.DataFrame(candidate_records)[manifest_columns]
build_output_root = OUTPUT_ROOT / "dataset" / "build_large_basketball_dataset"
build_output_root.mkdir(parents=True, exist_ok=True)
manifest_path = build_output_root / "basketball_dataset_manifest.csv"
manifest_df.to_csv(manifest_path, index=False)
print(f"Manifest: {manifest_path}")


# %% [markdown]
# ## 5. Validate the canonical exports
#
# Check the generated COCO and YOLO artifacts, their split counts, annotations,
# filenames, class mapping, normalized YOLO boxes, and manifest reconciliation.

# %%
expected_names = {
    split: {
        record["pixel_sha256"] + Path(record["source_path"]).suffix
        for record in retained_records
        if record["target_split"] == split
    }
    for split in ("train", "val", "test")
}
for split in ("train", "val", "test"):
    annotation_path = coco_dataset_dir / "annotations" / f"instances_{split}.json"
    exported_coco = json_io.read_json(annotation_path)
    assert [category["name"] for category in exported_coco["categories"]] == [
        "basketball"
    ]
    category_ids = {
        category["id"] for category in exported_coco["categories"]
    }
    image_ids = {image["id"] for image in exported_coco["images"]}
    assert all(
        annotation["image_id"] in image_ids
        and annotation["category_id"] in category_ids
        for annotation in exported_coco["annotations"]
    )

    coco_names = {
        path.name for path in (coco_dataset_dir / "images" / split).iterdir()
    }
    yolo_names = {
        path.name for path in (yolo_dataset_dir / "images" / split).iterdir()
    }
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

assert "basketball" in (
    yolo_dataset_dir / "data.yaml"
).read_text(encoding="utf-8")
assert len(manifest_df) == sum(EXPECTED_CANDIDATE_COUNTS.values())
assert (manifest_df["status"] == "retained").sum() == sum(
    EXPECTED_RETAINED_COUNTS.values()
)
assert (manifest_df["status"] == "exact_duplicate").sum() == (
    EXPECTED_DUPLICATES_REMOVED
)
assert not manifest_df["status"].eq("pending").any()

final_summary = pd.DataFrame(
    {
        "metric": [
            "source candidates",
            "retained train images",
            "retained validation images",
            "retained test images",
            "removed duplicate records",
            "COCO export",
            "YOLO export",
            "manifest",
        ],
        "result": [
            len(candidate_records),
            n_train,
            n_val,
            n_test,
            removed_duplicate_count,
            str(coco_dataset_dir),
            str(yolo_dataset_dir),
            str(manifest_path),
        ],
    }
)
display(final_summary.style.hide(axis="index"))
print("Canonical COCO and YOLO validation passed.")
