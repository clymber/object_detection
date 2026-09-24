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

# %%
"""
Basketball Models Comparison Notebook
Discover completed full-run bundles and compare them with the shared evaluator.
"""

# %% [markdown]
# # Basketball Large Dataset Model Comparison
#
# This notebook consumes immutable evaluation bundles below
# `OUTPUT_ROOT/evaluation/basketball_large_dataset/`. It does not load model
# frameworks, start producer processes, run inference, train, or require a GPU.
# Missing models remain unavailable until their owning project finishes or
# recovers a new-protocol full run and publishes its bundle.

# %%
from __future__ import annotations

from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

from detection_evaluation import (
	MODEL_NAMES,
	discover_bundles,
	write_bundle_comparisons,
)
from detection_evaluation.config import DATA_ROOT, OUTPUT_ROOT

# %% [markdown]
# ## Comparison settings
#
# Leave a model out of `RUN_SELECTIONS` to select its newest compatible full
# bundle by protocol creation time and run name. Set a value to a run name or
# absolute bundle directory to require that exact published bundle. Explicit
# missing, smoke, truncated, or invalid selections raise instead of falling
# back to another run.

# %%
RUN_SELECTIONS: dict[str, str | Path] = {
	# "yolo11n": "yolo11n_20260919T120000",
	# "yolox_tiny": "yolox_tiny_20260919T130000",
	# "yolox_nano": "yolox_nano_20260919T140000",
	# "rfdetr_small": "rfdetr_small_20260919T150000",
}
SOURCE_DATASET = DATA_ROOT / "composed" / "coco_basketball"
REPORT_DIR = OUTPUT_ROOT / "comparisons" / "basketball_large_dataset"
ANNOTATIONS = {
	split: SOURCE_DATASET / "annotations" / f"instances_{split}.json"
	for split in ("val", "test")
}
if set(RUN_SELECTIONS) - set(MODEL_NAMES):
	raise ValueError("RUN_SELECTIONS contains an unsupported model name")
if not all(path.is_file() for path in ANNOTATIONS.values()):
	raise FileNotFoundError(
		"Expected canonical validation/test annotations below "
		f"{SOURCE_DATASET / 'annotations'}"
	)

# %% [markdown]
# ## Discover and validate bundles
#
# Each candidate is read once through the Stage 3 reader. The consumer verifies
# the immutable generation, hashes, model and split mapping, selected checkpoint,
# resolution, parameter count, training summary, and canonical source identity
# before comparison. Smoke and truncated candidates are excluded automatically.

# %%
selections = discover_bundles(OUTPUT_ROOT, selections=RUN_SELECTIONS)
selection_rows = []
for model, selection in selections.items():
	details = selection.report_details()
	summary = details.get("training_summary", {})
	selection_rows.append(
		{
			"model": model,
			"status": details["status"],
			"bundle_dir": details.get("bundle_dir"),
			"generation": details.get("generation"),
			"checkpoint_sha256": details.get("selected_checkpoint", {}).get("sha256"),
			"original_utc": details.get("protocol", {}).get("original_utc"),
			"resolution": details.get("resolution"),
			"parameters": details.get("parameter_count"),
			"completed_epochs": summary.get("completed_epochs"),
			"training_hours": summary.get("total_hours"),
			"recovery": details.get("recovery"),
		}
	)
display(pd.DataFrame(selection_rows))

# %% [markdown]
# ## Recompute validation and held-out metrics
#
# Both reports use the shared COCO evaluator. AP uses a score floor of 0.001;
# precision, recall, F1, and negative-image error measures use confidence 0.25
# and IoU 0.50. Inference timings are compared only when their benchmark
# protocol, host, accelerator, image sequence, and settings are compatible.

# %%
reports = write_bundle_comparisons(selections, ANNOTATIONS, REPORT_DIR)
for split, report in reports.items():
	display(Markdown(f"## {split.title()} metrics"))
	display(report)

# %% [markdown]
# ## Saved reports and interpretation
#
# The report directory contains the selected bundle provenance plus metric,
# inference, and training CSV/JSON/Markdown views for both splits. Training
# totals are derived only from finalized monotonic attempts; unavailable timing
# remains unavailable rather than being reconstructed.

# %%
for name in ("selection", "metrics_test", "inference_test", "training_test"):
	report_path = REPORT_DIR / f"{name}.md"
	if report_path.is_file():
		display(Markdown(report_path.read_text()))
