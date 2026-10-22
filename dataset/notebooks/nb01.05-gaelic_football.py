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
# # Dataset Cleaning for a Gaelic Football Dataset
#
# This notebook uses FiftyOne to identify and review exact and near-duplicate images.
# Selected duplicates are removed, and the cleaned dataset is exported to the processed
# data directory.

# %%
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.brain as fob
from detection_common.utils import json_io
from detection_common.utils.filepath import dir_tree

from dataset_builder.config import DATA_ROOT, WORKSPACE_ROOT

# %%
GAELIC_FOOTBALL_PATH = DATA_ROOT / "sources" / "downloads" / "gaelic-football"

dataset = fo.Dataset.from_dir(
    dataset_dir=GAELIC_FOOTBALL_PATH,
    dataset_type=fo.types.ImageDirectory
)

# %%
exect_duplicates = fob.compute_exact_duplicates(dataset)
if exect_duplicates:
    print("Exact duplicated images:")
    json_io.print_json(exect_duplicates)
else:
    print("No exact duplicate found.")

# %%
index = fob.compute_near_duplicates(dataset, threshold=0.2)

# %%
for threshold in [0.10, 0.15, 0.20, 0.25]:
    index.find_duplicates(thresh=threshold)
    print(f"Number of duplicates with threshold {threshold}: {len(index.duplicate_ids)}")
    if index.duplicate_ids:
        print("Near duplicate ids:")
        json_io.print_json(index.duplicate_ids)

# %%
if index.neighbors_map:
    print("Length of neighbors map:", len(index.neighbors_map))

# %%
session = fo.launch_app(index.duplicates_view())

# %%
DUPLICATE_THRESHOLD = 0.8

index.find_duplicates(thresh=DUPLICATE_THRESHOLD)

duplicates_view = index.duplicates_view(
    type_field="duplicate_type",
    id_field="duplicate_group",
    dist_field="duplicate_distance",
)

print(f"Potential duplicate images: {len(duplicates_view)}")
session.view = duplicates_view

# %%
if session.selected:
    print("Selected duplicates:")
    json_io.print_json(session.selected)

    for dup_id in session.selected:
        print(Path(dataset[dup_id].filepath).relative_to(DATA_ROOT))


# %%
dataset.delete_samples(session.selected)
session.clear_selected()
session.refresh()

# %%
CLEAN_PATH = DATA_ROOT / "processed" / "gaelic-football"

if CLEAN_PATH.exists():
    print(f"Removing old directory: {CLEAN_PATH.relative_to(WORKSPACE_ROOT).as_posix()}")
    shutil.rmtree(CLEAN_PATH)

dataset.export(
    export_dir=CLEAN_PATH.as_posix(),
    dataset_type=fo.types.ImageDirectory,
    export_media=True
)

# %%
session.freeze()

# %%
print(dir_tree(CLEAN_PATH.parent))

# %%
session.show()
