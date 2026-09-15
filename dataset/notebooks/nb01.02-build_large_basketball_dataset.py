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

from dataset_builder.config import DATA_ROOT, WORKSPACE_ROOT
from detection_common import configure_stdio_relative_path

# Display project paths relative to project root directory for consistency.
configure_stdio_relative_path(WORKSPACE_ROOT)

# %%
import hashlib
import math
from itertools import islice
from pathlib import Path

import datumaro as dm
import pandas as pd
from dataset_builder import (
    prefer_hardlinked_datumaro_media,
    summarize_coco_datasets,
    summarize_datumaro_label_counts,
)
from dataset_builder import roboflow as rf_platform
from dataset_builder.datumaro import ExportFormat as DatumaroExpFmt
from dataset_builder.datumaro import ImportFormat as DatumaroImpFmt
from dataset_builder.roboflow import RoboflowFormat as rf_format
from detection_common.utils import json_io
from detection_common.utils.filepath import dir_tree, ensure_dir
from IPython.display import display

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
    Return a deterministic, flat item ID derived from its source identity.
    """
    source_key = f"{source}:{item.subset}:{item.id}"
    return hashlib.sha1(
        source_key.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:16]


basketball_dm_dataset = dm.Dataset.from_iterable(
    (
        item.wrap(
            id=hash_item_id(source, item),
            subset=target_subset,
        )
        for target_subset, source, dataset in (
            ("train", trn_set_url, train_dm_dataset),
            ("val", val_set_url, val_dm_dataset),
            ("test", tst_set_url, test_dm_dataset),
        )
        for item in dataset
    ),
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
# ### 2.3 Collect background images from tennis dataset
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

background_items = []
tennis_items = iter(backg_dm_dataset)
for target_subset, count in backgrounds_needed_by_subset.items():
    background_items.extend(
        item.wrap(
            id=hash_item_id(bg_tnns_url, item),
            subset=target_subset,
            annotations=[],
        )
        for item in islice(tennis_items, count)
    )

basketball_dm_dataset.update(background_items)

image_statistics = summary_annotated_vs_background(basketball_dm_dataset)
display(
    image_statistics.reset_index().style.hide(axis="index").format(
        {"background_percentage": "{:.2f}%"}
    )
)

# %% [markdown]
# ## 3. Export the new basketball dataset
#
# Export the sampled dataset into both COCO and YOLO formats, using hardlinks for media
# files when possible to save disk space.


# %%
DATASET_ROOT = DATA_ROOT
n_train = len(basketball_dm_dataset.get_subset("train"))
n_val = len(basketball_dm_dataset.get_subset("val"))
n_test = len(basketball_dm_dataset.get_subset("test"))
ds_name_suffix = f"basketball_{n_train}_{n_val}_{n_test}"
coco_dataset_dir = DATASET_ROOT / "composed" / f"coco_{ds_name_suffix}"
yolo_dataset_dir = DATASET_ROOT / "composed" / f"yolo_{ds_name_suffix}"

with prefer_hardlinked_datumaro_media():
    if not coco_dataset_dir.exists():
        ensure_dir(coco_dataset_dir)
        basketball_dm_dataset.export(
            str(coco_dataset_dir),
            format=DatumaroExpFmt.COCO_INSTANCES,
            save_media=True,
            reindex=True,
        )
    else:
        print(f"COCO dataset already exists at {coco_dataset_dir}")

    if not yolo_dataset_dir.exists():
        ensure_dir(yolo_dataset_dir)
        basketball_dm_dataset.export(
            str(yolo_dataset_dir),
            format=DatumaroExpFmt.YOLO_ULTRALYTICS,
            save_media=True,
        )
    else:
        print(f"YOLO dataset already exists at {yolo_dataset_dir}")

# %%
print(f"COCO dataset directory: {coco_dataset_dir}")
print(dir_tree(coco_dataset_dir))

# %% [markdown]
#
