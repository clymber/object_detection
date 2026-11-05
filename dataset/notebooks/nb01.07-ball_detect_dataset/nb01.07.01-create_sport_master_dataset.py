# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     notebook_metadata_filter: jupytext,title,authors,-kernelspec,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Build a Master Dataset for Sport Objects
#
# Import multiple source datasets into the master dataset and cache normalized
# pixel digests.

# %%
# %load_ext autoreload
# %autoreload 2
# %aimport -fiftyone -pathlib

from detection_common import configure_stdio_relative_path
from detection_common.utils.digest import image_content_digest
from detection_common.utils.json_io import print_json

from dataset_builder.config import DATA_ROOT, WORKSPACE_ROOT

# Display project paths relatively for consistent output across environments.
configure_stdio_relative_path(WORKSPACE_ROOT)

# %%
from pathlib import Path

import fiftyone as fo
from fiftyone.types import COCODetectionDataset

# Name of target master dataset to be created/imported
MASTER_DATASET_NAME = "sports_objects_master"

# %%
# REBUILD_MASTER_DATASET = True
REBUILD_MASTER_DATASET = False
if REBUILD_MASTER_DATASET and fo.dataset_exists(MASTER_DATASET_NAME):
    fo.delete_dataset(MASTER_DATASET_NAME)


# %%
def import_coco_split(
    dataset: fo.Dataset,
    split_dir: Path,
    label_path: Path,
    source_name: str,
    split: str,
) -> list[str]:
    """
    Merge a COCO split, cache pixel digests, and return newly added sample IDs.

    Preserve existing samples and fill missing digests after interrupted imports.
    """
    existing_ids = dataset.values("id")

    dataset.merge_dir(
        dataset_type=COCODetectionDataset,
        data_path=split_dir.as_posix(),
        labels_path=label_path.as_posix(),
        label_types=["detections"],
        label_field="original_labels",
        tags=[f"source:{source_name}"],
        key_field="filepath",
        skip_existing=True,
    )

    new_ids = dataset.exclude(existing_ids).values("id")
    if new_ids:
        dataset.set_values(
            "source_dataset",
            {sid: source_name for sid in new_ids},
            key_field="id",
        )
        dataset.set_values(
            "original_split",
            {sid: split for sid in new_ids},
            key_field="id",
        )

    unhashed_ids, unhashed_paths = dataset.exists("pixel_digest", False).values(
        ["id", "filepath"]
    )
    if unhashed_ids:
        with fo.ProgressBar(total=len(unhashed_ids)) as progress:
            digests = {
                sid: image_content_digest(path)
                for sid, path in progress(zip(unhashed_ids, unhashed_paths))
            }
        dataset.set_values("pixel_digest", digests, key_field="id")
    return new_ids

# %%
if fo.dataset_exists(MASTER_DATASET_NAME):
    master = fo.load_dataset(MASTER_DATASET_NAME)
else:
    master = fo.Dataset(MASTER_DATASET_NAME)
master.persistent = True
if not master.has_sample_field("pixel_digest"):
    master.add_sample_field("pixel_digest", fo.StringField)

# %% [markdown]
# Each sample's `pixel_digest` is the SHA-256 of its dimensions and RGBA pixels
# after EXIF orientation normalization. Compute it once during import and reuse
# it for duplicate detection. Recompute the field if the source media changes.
# Run `nb01.07.02` after importing to tag exact pixel duplicates for review.

# %% [markdown]
# ## Import gaelic football dataset
#
# Preserve provenance: `source_dataset` records the original dataset name, and
# `original_split` records "train", "val"/"valid", or "test" when available.

# %%
splits = ("train", "val", "valid", "test")
source_dir = DATA_ROOT / "composed" / "coco_gaelic_football"

for split in splits:
    split_dir = source_dir / "images" / split
    label_path = source_dir / "annotations" / f"instances_{split}.json"

    if not label_path.is_file():
        continue

    import_coco_split(
        dataset=master,
        split_dir=split_dir,
        label_path=label_path,
        source_name=source_dir.name,
        split=split,
    )

print("Total images:", master.count())
print_json(master.count_sample_tags())

# %% [markdown]
# ## Import Roboflow source datasets
#
# Discover COCO exports under `<dataset>/<version>/<split>`. Preserve the
# source dataset name and original split, and skip existing image filepaths.

# %%
roboflow_root = DATA_ROOT / "sources" / "roboflow"

if not roboflow_root.is_dir():
    raise FileNotFoundError(f"Roboflow source directory not found: {roboflow_root}")

for label_path in sorted(roboflow_root.glob("*/*/*/_annotations.coco.json")):
    split_dir = label_path.parent
    split = split_dir.name
    if split not in splits:
        continue

    source_dir = split_dir.parent.parent
    new_ids = import_coco_split(
        dataset=master,
        split_dir=split_dir,
        label_path=label_path,
        source_name=source_dir.name,
        split=split,
    )
    print(f"{source_dir.name}/{split_dir.parent.name}/{split}: {len(new_ids)} added")

# %%
print("Total images:", master.count())
print_json(master.count_sample_tags())
