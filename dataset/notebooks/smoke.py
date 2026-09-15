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
from dataset_builder import summarize_coco_datasets
from dataset_builder.config import (
    DATA_ROOT,
    OUTPUT_ROOT,
    SUBPROJECT_ROOT,
    WORKSPACE_ROOT,
)

assert SUBPROJECT_ROOT.is_absolute() and WORKSPACE_ROOT.is_absolute()
assert DATA_ROOT.is_absolute() and OUTPUT_ROOT.is_absolute()
assert summarize_coco_datasets([]).empty
