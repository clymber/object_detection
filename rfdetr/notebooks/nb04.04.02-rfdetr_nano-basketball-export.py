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
# # rfdetr_nano — export and runtime validation
#
# Reload a completed training run in a fresh kernel. Keep one export set under
# `models/exports/basketball/<model>/<run>/{onnx,coreml}/`, without an attempt-ID
# directory. Reports use the matching `outputs/exports/` path. The run's
# `weights/` and `export_validation` links expose those same artifacts/results.
# Training inputs stay unchanged. These are host measurements, not iPhone results.
# Run with the `object-detection-rfdetr` kernel.

# %%
from __future__ import annotations

import platform
from pathlib import Path

import pandas as pd
from IPython.display import display
from detection_common import configure_stdio_relative_path
from detection_common.utils.image import display as display_img
from detection_evaluation.runtime_workflow import require_success
from rfdetr_pipeline.config import DATA_ROOT, OUTPUT_ROOT, WORKSPACE_ROOT
from rfdetr_pipeline.export_validation import export_and_validate

configure_stdio_relative_path(WORKSPACE_ROOT)

# %% [markdown]
# ## Settings
#
# Set RUN_DIR to an absolute completed-run directory to pin an exact checkpoint.
# None selects the latest eligible rfdetr_nano run using modification time.
# Archived Linux dataset paths are resolved against DATASET_DIR without rewriting
# the saved provenance. CPU/FP32 is the checkpoint/ONNX reference configuration.

# %%
RUNS_DIR = OUTPUT_ROOT / "runs" / "basketball"
RUN_DIR = None
DATASET_DIR = DATA_ROOT / "composed" / "basketball-v0.1" / "coco"
EXPORT_FORMATS = ("onnx", "coreml") if platform.system() == "Darwin" else ("onnx",)
CHECKPOINT_DEVICE = "cpu"
CPU_THREADS = 4
COREML_PRECISION = "float16"
COREML_COMPUTE_UNITS = "all"

# %% [markdown]
# ## Export, parity, complete validation/test evaluation and host timing
#
# The fixed development set uses the first 20 positive and five negative validation
# IDs. It is distinct from the frozen mobile parity set. Confidence is inclusive
# 0.25, top 100; coordinate tolerance is max(2 px, 0.002 × source dimension),
# score tolerance 0.02. Full accuracy uses score floor 0.001 and COCO maxDets.
# Host timing uses 30 warmups and three repetitions of 300 predecoded test images.
# Format failures are independent; successful results survive later failures.

# %%
result = export_and_validate(
    RUN_DIR, DATASET_DIR,
    model="rfdetr_nano",
    runs_dir=RUNS_DIR,
    formats=EXPORT_FORMATS,
    device=CHECKPOINT_DEVICE,
    cpu_threads=CPU_THREADS,
    coreml_precision=COREML_PRECISION,
    compute_units=COREML_COMPUTE_UNITS,
)

# %%
print(f"Manifest: {result['manifest_path']}")
print(f"Exports: {result['export_dir']}")
print(f"Validation: {result['report_dir']}")
statuses = pd.DataFrame.from_dict(result["formats"], orient="index")
for column, key in (("runtime_precision", "precision"),
                    ("device", "device"), ("compute_units", "compute_units")):
    statuses[column] = pd.Series({
        name: outcome.get("contract", {}).get(key)
        for name, outcome in result["formats"].items()
    })
display(statuses.reindex(columns=[
    "status", "parity_status", "accepted", "runtime_precision", "device",
    "compute_units", "failed_stage", "error"
]))
metric_rows = [
    {"runtime": name, "split": split,
     "accepted": outcome.get("accepted", False),
     "parity_status": outcome.get("parity_status"), **{
        key: metrics[key] for key in ("ap50", "ap50_95", "precision", "recall", "f1")
    }}
    for name, outcome in result["formats"].items()
    for split, metrics in outcome.get("metrics", {}).items()
]
display(pd.DataFrame(metric_rows))

# %% [markdown]
# ## Qualitative comparison
#
# Each runtime uses the same saved source-image selection. Green is ground truth;
# red is that runtime's detections at confidence >= 0.25. Complete grids are saved
# with the prediction artifacts, including after failed development parity.

# %%
for name in result["formats"]:
    for split in ("val", "test"):
        grid = Path(result["report_dir"]) / name / f"{split}_comparison.jpg"
        if grid.is_file():
            print(f"{name}: {split}")
            display_img(grid, width=1000)

# %% [markdown]
# Requested conversion/evaluation failures or failed parity make execution fail
# after saving all available diagnostic evidence and showing per-format outcomes.

# %%
require_success(result)
