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

import os
import platform
from pathlib import Path

# Set before importing PyTorch through a model backend.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd
from detection_common import configure_stdio_relative_path
from detection_common.utils.image import display as display_img
from detection_common.utils.json_io import read_json
from detection_evaluation.runtime_workflow import require_success
from IPython.display import display

from rfdetr_pipeline.config import DATA_ROOT, OUTPUT_ROOT, WORKSPACE_ROOT
from rfdetr_pipeline.export_validation import (
    benchmark_host,
    check_parity,
    evaluate_accuracy,
    export_models,
    load_export_set,
    prepare_export,
    read_results,
)

configure_stdio_relative_path(WORKSPACE_ROOT)

# %% [markdown]
# ## 1. Configure
#
# - `RUN_DIR=None` selects the latest completed `rfdetr_nano` run in `RUNS_DIR`.
# - Set `RUN_DIR` to an absolute or workspace-relative directory to pin an exact
#   run, for example `Path("outputs/runs/basketball/<run>")`.
# - Relative `RUN_DIR`, `RUNS_DIR`, and `DATASET_DIR` paths resolve from
#   `WORKSPACE_ROOT`, independently of the notebook's current directory.
# - `DATA_ROOT` and `OUTPUT_ROOT` default to the workspace's `data/` and `outputs/`.
#   Their environment overrides must be absolute paths.
# - `DATASET_DIR` locates the local dataset for archived runs while preserving
#   saved training provenance.
# - CPU/FP32 is the default reference. Set `CHECKPOINT_DEVICE="mps"` to use
#   Apple GPU inference with CPU fallback for unsupported PyTorch operations.
#   ONNX remains CPU; Core ML already uses all available compute units.

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
# ## 2. Prepare and inspect
#
# Verify the selected run, saved settings and local dataset. This step loads no
# detector and writes no artifacts.

# %%
plan = prepare_export(
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
print(f"Run: {plan.run_dir}")
display(pd.Series({
    "model": plan.model,
    "checkpoint": str(plan.run_dir / "checkpoint_best_total.pth"),
    "resolution": plan.configuration["resolution"],
    "category_ids": plan.configuration["category_ids"],
    "classes": [category["name"] for category in plan.protocol[
        "dataset_identity"]["canonical"]["canonical_source"]["category_mapping"]],
    "coreml_precision": plan.configuration["coreml_precision"],
    "compute_units": plan.configuration["compute_units"],
    "formats": plan.formats,
    "device": plan.configuration["checkpoint_device"],
    "cpu_threads": plan.configuration["cpu_threads"],
}))

# %% [markdown]
# ## 3. Export
#
# Convert each requested format using fresh checkpoint state. Only rerunning this
# step replaces exported models and starts a new export generation.

# %%
export_set = export_models(plan)

# %%
export_result = read_results(export_set)
print(f"Manifest: {export_result['manifest_path']}")
print(f"Exports: {export_result['export_dir']}")
display(pd.DataFrame.from_dict(export_result["exports"], orient="index"))

# %% [markdown]
# ## 4. Check parity
#
# Compare the first 20 positive and five negative validation IDs against a fresh
# checkpoint reference. This development set is distinct from frozen mobile parity.
# Confidence is inclusive 0.25, top 100; coordinate tolerance is
# max(2 px, 0.002 × source dimension), and score tolerance is 0.02.
#
# After a kernel restart, run imports and resume with
# `export_set = load_export_set(Path("<saved manifest path>"))`, skipping steps 2–3.
# Every validation cell executes only its named task and keeps model artifacts.
# Rerunning parity keeps accuracy reports and clears prior host timing.

# %%
parity_result = check_parity(export_set)

# %%
display(pd.DataFrame.from_dict({
    name: {"parity_status": outcome.get("parity_status"),
           **outcome.get("stages", {}).get("parity", {})}
    for name, outcome in parity_result["formats"].items()
}, orient="index"))

# %% [markdown]
# ## 5. Evaluate accuracy
#
# Evaluate every test image at score floor 0.001 with COCO maxDets.
# Validation keeps its separate 25-image parity check.
# Failed parity keeps these results diagnostic and unaccepted. Rerunning accuracy
# keeps parity reports and clears prior host timing. Format failures are independent.

# %%
accuracy_result = evaluate_accuracy(export_set)

# %%
metric_rows = [
    {"runtime": name, "split": split,
     "parity_status": outcome.get("parity_status"), **{
        key: metrics[key] for key in ("ap50", "ap50_95", "precision", "recall", "f1")
    }}
    for name, outcome in accuracy_result["formats"].items()
    for split, metrics in outcome.get("metrics", {}).items()
]
display(pd.DataFrame(metric_rows))

# %% [markdown]
# ## 6. Measure host timing
#
# Measure eligible runtimes with five untimed warmups, then 100 predecoded test
# images once. Warmups let initialization settle before measuring. This cell
# performs no conversion, parity or full-split evaluation. Saved timing reports
# record the exact protocol; compare latency only within matching timing groups.

# %%
timing_result = benchmark_host(export_set)

# %%
timing_rows = []
for name, outcome in timing_result["formats"].items():
    row = {"runtime": name, **outcome.get("stages", {}).get("timing", {})}
    if "timing" in outcome:
        timing = read_json(outcome["timing"]["path"])
        row.update({key: timing[key] for key in ("mean_ms", "p50_ms", "p95_ms")})
    timing_rows.append(row)
display(pd.DataFrame(timing_rows))

# %% [markdown]
# ## 7. Inspect saved comparisons
#
# Read existing qualitative grids without running inference. Green is ground truth;
# red is detections at confidence >= 0.25, including diagnostic parity failures.

# %%
result = read_results(export_set)
for name in result["formats"]:
    for split in result["formats"][name].get("metrics", {}):
        grid = Path(result["report_dir"]) / name / f"{split}_comparison.jpg"
        if grid.is_file():
            print(f"{name}: {split}")
            display_img(grid, width=1000)

# %% [markdown]
# ## 8. Review completion
#
# Refresh saved statuses. Print unfinished or failed validation after preserving
# diagnostic evidence, so notebook task runners can continue.

# %%
result = read_results(export_set)
print(f"Manifest: {result['manifest_path']}")
print(f"Validation: {result['report_dir']}")
statuses = pd.DataFrame.from_dict(result["formats"], orient="index")
for column, key in (("runtime_precision", "precision"),
                    ("device", "device"), ("compute_units", "compute_units")):
    statuses[column] = pd.Series({
        name: outcome.get("contract", {}).get(key)
        for name, outcome in result["formats"].items()
    })
for stage in ("parity", "accuracy", "timing"):
    statuses[f"{stage}_stage"] = pd.Series({
        name: outcome.get("stages", {}).get(stage, {}).get("status")
        for name, outcome in result["formats"].items()
    })
display(statuses.reindex(columns=[
    "status", "parity_stage", "accuracy_stage", "timing_stage", "parity_status",
    "accepted", "runtime_precision", "device", "compute_units", "failed_stage", "error",
]))
try:
    require_success(result)
except RuntimeError as error:
    print(f"ERROR: {error}")
