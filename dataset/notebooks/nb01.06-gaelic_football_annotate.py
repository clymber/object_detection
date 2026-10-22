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
# # Annotate A Gaelic Football Dataset Manually

# %%
import fiftyone as fo
import fiftyone.brain as fob
from detection_common.utils import json_io
from detection_common.utils.filepath import dir_tree

from dataset_builder.config import DATA_ROOT, WORKSPACE_ROOT

# %%
print("FiftyOne: ", fo.__version__)

# %%
SOURCE_DIR = DATA_ROOT / "processed" / "gaelic-football"
print(dir_tree(SOURCE_DIR.parent))

# %% [markdown]
# Import images from directory

# %%
dataset = fo.Dataset.from_dir(
    dataset_dir=SOURCE_DIR,
    dataset_type=fo.types.ImageDirectory
)

# Keep dataset metadata and annotations across sessions
dataset.persistent = True

# %% [markdown]
# Open FiftyOne UI

# %%
session = fo.launch_app(dataset)

# %%
session.show()

# %%
from pathlib import Path

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
import shutil

CLEAN_PATH = DATA_ROOT / "processed" / "gaelic-football2"

if CLEAN_PATH.exists():
    print(f"Removing old directory: {CLEAN_PATH.relative_to(WORKSPACE_ROOT).as_posix()}")
    shutil.rmtree(CLEAN_PATH)

dataset.export(
    export_dir=CLEAN_PATH.as_posix(),
    dataset_type=fo.types.ImageDirectory,
    export_media=True
)

# %%
session.show()

# %% [markdown]
# Save annotations in the FiftyOne UI before exporting. Rerun this export cell
# after further edits; rerunning the image import creates a separate dataset.
#
# Include samples with `ground_truth` values or the sample tag `background`.
# Negative images export without boxes; no `background` object class is needed.
# An empty `fo.Detections(detections=[])` can explicitly mark a reviewed negative.
# Untagged samples without `ground_truth` remain excluded.

# %%
ANNOTATED_DIR = DATA_ROOT / "processed" / "gaelic-football-annotated"

# Refresh the notebook's schema and samples after changes made by the UI.
dataset.reload()
export_view = dataset.match(
    fo.ViewField("ground_truth").exists()
    | fo.ViewField("tags").contains("background")
)

export_view.export(
    export_dir=ANNOTATED_DIR.as_posix(),
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth",
    export_media=True,
    overwrite=True,
)

# %% [markdown]
# ## Split the reviewed images into COCO training, validation, and test sets
#
# Split labeled images and negative images separately with the same 70/15/15
# fractions, rounded to whole images. A fixed seed and filepath order make the
# assignments reproducible for the same reviewed images. Samples with boxes count
# as labeled even if they also carry the `background` tag.
#
# The export uses `images/<split>/` and `annotations/instances_<split>.json`.
# Rerunning the export replaces these split outputs. Source annotations and sample
# tags are preserved.

# %%
RESULT_DIR = DATA_ROOT / "composed" / "coco_gaelic_football"

SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
SPLIT_TAGS = tuple(SPLITS)
SPLIT_SEED = 42

# %%
import fiftyone.utils.random as four
import pandas as pd
from IPython.display import display

# Reload both the schema and the view after any further edits in the UI.
export_view.reload()
if not len(export_view):
    raise ValueError("Annotate images or tag reviewed negatives before splitting.")

has_detections = (
    fo.ViewField("ground_truth.detections").if_null([]).length() > 0
)
split_ids = {split: [] for split in SPLIT_TAGS}
for group_view in (
    export_view.match(has_detections),
    export_view.match(~has_detections),
):
    group_splits = four.random_split(
        group_view.sort_by("filepath"), list(SPLITS.values()), seed=SPLIT_SEED
    )
    for split, group_split in zip(SPLIT_TAGS, group_splits):
        split_ids[split].extend(group_split.values("id"))

assigned_ids = [sample_id for ids in split_ids.values() for sample_id in ids]
assert len(assigned_ids) == len(set(assigned_ids)) == len(export_view)
split_views = {
    split: export_view.select(ids) for split, ids in split_ids.items()
}

split_summary = pd.DataFrame(
    {
        "split": split,
        "images": len(split_view),
        "labeled": len(split_view.match(has_detections)),
        "background": len(split_view.match(~has_detections)),
        "annotations": split_view.count("ground_truth.detections"),
    }
    for split, split_view in split_views.items()
)
display(split_summary.set_index("split"))

# %%
# Use the complete class list so category IDs agree across every split.
class_names = sorted(export_view.distinct("ground_truth.detections.label"))
for split, split_view in split_views.items():
    image_dir = RESULT_DIR / "images" / split
    labels_path = RESULT_DIR / "annotations" / f"instances_{split}.json"
    split_view.export(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=image_dir.as_posix(),
        labels_path=labels_path.as_posix(),
        label_field="ground_truth",
        classes=class_names,
        export_media=True,
        overwrite=True,
    )

    # FiftyOne can omit this list when a split has only negatives or is empty.
    coco_split = json_io.read_json(labels_path)
    coco_split.setdefault("annotations", [])
    json_io.write_json(labels_path, coco_split)
    assert len(coco_split["images"]) == len(split_view)
    assert all(
        (image_dir / image["file_name"]).is_file()
        for image in coco_split["images"]
    )

print(f"COCO splits exported to: {RESULT_DIR}")

# %%
session.freeze()
session.close()
