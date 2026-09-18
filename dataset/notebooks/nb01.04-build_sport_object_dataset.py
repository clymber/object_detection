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

# %% [markdown]
# # Build the two-class sport object dataset
#
# Combine the canonical football and basketball COCO exports without changing their
# train, validation, or test assignments. The build retains background images,
# refuses duplicate decoded pixels, and exports matching COCO and Ultralytics YOLO
# datasets with `football` at class index 0 and `basketball` at class index 1.

# %%
"""
Build the two-class sport object dataset.
"""

# %load_ext autoreload
# %autoreload 2
# %aimport -csv, -textwrap, -functools, -IPython, -ultralytics, -pandas, -datumaro

from detection_common import configure_stdio_relative_path

from dataset_builder.config import DATA_ROOT, OUTPUT_ROOT, WORKSPACE_ROOT

configure_stdio_relative_path(WORKSPACE_ROOT)

# %%
from IPython.display import display

from dataset_builder._sport_object import build_sport_object_dataset

COMPOSED_ROOT = DATA_ROOT / "composed"
BASKETBALL_ROOT = COMPOSED_ROOT / "coco_basketball"
FOOTBALL_ROOT = COMPOSED_ROOT / "coco_football"
COCO_ROOT = COMPOSED_ROOT / "coco_sport_object"
YOLO_ROOT = COMPOSED_ROOT / "yolo_sport_object"
MANIFEST_PATH = (
    OUTPUT_ROOT
    / "dataset"
    / "build_sport_object_dataset"
    / "sport_object_dataset_manifest.csv"
)

# %% [markdown]
# ## 1. Compose and export
#
# Both source inputs must contain `train`, `val`, and `test` COCO splits. The build
# validates media, dimensions, categories, and boxes before it writes either output.
# Existing output directories or any repeated pixel hash stop the build.

# %%
manifest_df = build_sport_object_dataset(
    basketball_root=BASKETBALL_ROOT,
    football_root=FOOTBALL_ROOT,
    coco_root=COCO_ROOT,
    yolo_root=YOLO_ROOT,
    manifest_path=MANIFEST_PATH,
)

# %% [markdown]
# ## 2. Verify provenance totals
#
# The helper has already validated generated COCO and YOLO content. This summary
# reconciles the provenance manifest by source and preserved split.

# %%
summary = (
    manifest_df.groupby(["source_dataset", "split"], sort=True)
    .agg(
        images=("pixel_sha256", "size"),
        backgrounds=("is_background", "sum"),
        annotations=("annotation_count", "sum"),
    )
    .reset_index()
)
display(summary.style.hide(axis="index"))
print(f"COCO export: {COCO_ROOT}")
print(f"YOLO export: {YOLO_ROOT}")
print(f"Provenance manifest: {MANIFEST_PATH}")
