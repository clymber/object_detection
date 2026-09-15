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
from detection_evaluation import read_prediction_artifact
from detection_evaluation.config import (
    DATA_ROOT,
    OUTPUT_ROOT,
    SUBPROJECT_ROOT,
    WORKSPACE_ROOT,
)

assert SUBPROJECT_ROOT.is_absolute() and WORKSPACE_ROOT.is_absolute()
assert DATA_ROOT.is_absolute() and OUTPUT_ROOT.is_absolute()
assert callable(read_prediction_artifact)
