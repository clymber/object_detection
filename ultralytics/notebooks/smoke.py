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
#     display_name: Python (Object Detection Ultralytics)
#     language: python
#     name: object-detection-ultralytics
# ---

# %%
from ultralytics_pipeline import ultralytics as ultralytics_helpers
from ultralytics_pipeline.config import (
    DATA_ROOT,
    OUTPUT_ROOT,
    SUBPROJECT_ROOT,
    WORKSPACE_ROOT,
)

assert SUBPROJECT_ROOT.is_absolute() and WORKSPACE_ROOT.is_absolute()
assert DATA_ROOT.is_absolute() and OUTPUT_ROOT.is_absolute()
assert callable(ultralytics_helpers.configure_privacy)
