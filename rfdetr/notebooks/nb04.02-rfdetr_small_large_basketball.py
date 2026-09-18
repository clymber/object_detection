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
# # RF-DETR Small on the Large Basketball Dataset
#
# Fine-tune COCO-pretrained **RF-DETR Small** on the frozen dataset used by
# [nb02.02 (YOLO11n)](../../ultralytics/notebooks/nb02.02-ultra_yolo11n_large_basketball.py),
# [nb03.02 (YOLOX-Tiny)](../../yolox/notebooks/nb03.02-yolox_tiny_large_basketball.py),
# [nb03.03 (YOLOX-Nano)](../../yolox/notebooks/nb03.03-yolox_nano_large_basketball.py).
# This notebook follows the same workflow: train, inspect curves, reload the
# best validation checkpoint, export ONNX, evaluate the held-out test set, and
# review images.
#
# **Setup:**
#
# 1. Run `bash scripts/setup_conda_envs.sh rfdetr` from the workspace root.
# 2. Select **Python (Object Detection RF-DETR)** as this notebook's kernel.
# 3. Put the frozen dataset under `DATA_ROOT/composed/` or set an absolute
#    `OBJECT_DETECTION_DATA_ROOT` to an existing dataset directory.
# 4. Use `rfdetr/notebooks/smoke.py` for an import/kernel check. Run a
#    short GPU smoke check before starting a full training run.
# 5. Choose a mode in **Experiment settings** and keep the benchmark policy
#    consistent across models.
#
# The experiment uses 640 x 640 input, overriding Small's 512-pixel default,
# and up to 100 epochs. The final comparison recomputes metrics from saved
# YOLO predictions; exporting those predictions does not require retraining.
# You can complete this notebook before the YOLO exports are available.

# %%
from __future__ import annotations

import gc
import os
from pathlib import Path

import pandas as pd
import torch
from dataset_builder.rfdetr import prepare_coco_dataset, summarize_dataset
from detection_common import aligned_print, configure_stdio_relative_path
from detection_common.utils.image import display as display_img
from detection_common.utils.json_io import write_json
from detection_evaluation import write_comparison
from IPython.display import Markdown, display

from rfdetr_pipeline import rfdetr as rfdetr_platform
from rfdetr_pipeline.config import (
    DATA_ROOT,
    OUTPUT_ROOT,
    SUBPROJECT_ROOT,
    WORKSPACE_ROOT,
)

configure_stdio_relative_path(WORKSPACE_ROOT)
cache_paths = rfdetr_platform.configure_caches(WORKSPACE_ROOT)
runtime = rfdetr_platform.runtime_report()
aligned_print(runtime)

# %% [markdown]
# ## Experiment settings
#
# Leave `RUN_OVERRIDES` empty to use the defaults or `RFDETR_*` variables from
# the active shell. For interactive use, edit the dictionary below. Its values
# take precedence without changing the kernel environment.
#
# | Purpose | `RUN_OVERRIDES` |
# | --- | --- |
# | Full training (default) | `{}` |
# | Short GPU smoke check | `{"smoke_run": True}` |
# | Continue an interrupted run | `{"mode": "resume", "run_dir": "outputs/..."}` |
# | Rebuild evaluation and reports | `{"mode": "evaluate", "run_dir": "outputs/..."}` |
#
# Use the exact directory printed by the original run for `run_dir`.
# Resume/evaluate inherit its saved settings; incompatible overrides fail
# before training. Resume continues the original total epoch budget. Once
# that budget is complete, use `evaluate`. A fresh run always allocates a new
# directory, so rerunning fresh training does not replace an earlier run.
#
# A **GPU smoke run** uses two epochs with
# 16 training, 8 validation, and 8 test images selected within their existing
# splits. It exercises training, checkpoint reload, plots, and both evaluation
# paths before a full run. Its artifacts are excluded from the comparison.
# For full training afterwards, start a fresh run with smoke off.
#
# Defaults are 100 epochs, batch size 4, accumulation 1, seed 42, AMP training,
# and zero loader workers. Increase the physical batch if GPU memory permits;
# on an out-of-memory error, reduce it or enable `gradient_checkpointing` in
# a new run. Workers=0 avoids container shared-memory limits.
#
# Accumulation defaults to 1 because RF-DETR 1.10.1 and Lightning 2.6.1 both
# normalize accumulated losses. Values above 1 therefore change gradient
# scaling; they are not equivalent to increasing the physical batch. The
# saved configuration records the actual settings for comparison with YOLO.
# Training has a 100-epoch maximum and stops after 10 consecutive validation
# epochs without an EMA AP50:95 improvement of at least 0.001. Both regular
# and EMA weights are evaluated, but the smoother EMA metric controls stopping.
# Test evaluation during fitting remains disabled, so the held-out test split
# cannot influence when training stops.

# %%
RUN_OVERRIDES = {
    # "smoke_run": True,
    # "batch_size": 4,
    # "gradient_checkpointing": True,
    # "early_stopping_patience": 10,
    # "mode": rfdetr_platform.RunMode.EVALUATE,
    # "run_dir": "outputs/runs/basketball/rfdetr_small_basketball_large_dataset",
}
settings = rfdetr_platform.settings_from_env(overrides=RUN_OVERRIDES)
SOURCE_DATASET = DATA_ROOT / "composed" / "coco_basketball"
dataset_name = "basketball"
if settings.smoke_run:
    dataset_name += "_smoke"
DERIVED_DATASET = DATA_ROOT / "processed" / "rfdetr" / dataset_name

# Set to a shared output directory used by the Ultralytics and YOLOX
# `export-*-baseline` commands (see their project READMEs).
# If omitted, YOLO rows are explicitly unavailable while RF-DETR still runs.
BASELINE_EXPORT_DIR = os.environ.get("RFDETR_BASELINE_EXPORT_DIR") or None
benchmark_setting = os.environ.get("RFDETR_BENCHMARK", "1")
if benchmark_setting not in {"0", "1"}:
    raise ValueError("RFDETR_BENCHMARK must be 0 or 1")
BENCHMARK = benchmark_setting == "1"

aligned_print({
    **vars(settings), "samples_per_optimizer_step": settings.samples_per_optimizer_step,
    "source_dataset": SOURCE_DATASET, "derived_dataset": DERIVED_DATASET,
    "baseline_exports": BASELINE_EXPORT_DIR, "benchmark": BENCHMARK,
})

# %% [markdown]
# ## Dataset identity and loader layout
#
# The train/validation/test counts must be 11,394 / 1,155 / 1,395 before any
# smoke subset is selected. Preserve the single basketball class, every
# negative image, and all source boxes and split membership. The derived
# layout maps `val` to `valid` for RF-DETR, with links to the original images.
#
# The canonical source is globally deduplicated. The manifest records annotation
# and image hashes; reusing the derived dataset rechecks them. This can take a few
# minutes. Stale derived data raises an error: use a new derived directory after
# an intentional source change and treat it as a different experiment.

# %%
manifest = prepare_coco_dataset(
    SOURCE_DATASET,
    DERIVED_DATASET,
    expected_counts={"train": 11394, "val": 1155, "test": 1395},
    category_id=None,
    smoke_limits={"train": 16, "val": 8, "test": 8} if settings.smoke_run else None,
)
display(pd.DataFrame(summarize_dataset(manifest)))
print(f"Dataset fingerprint: {manifest['fingerprint']}")
print(f"Duplicate-content groups preserved: {len(manifest['duplicate_image_groups'])}")
cross_split_duplicates = [
    group for group in manifest["duplicate_image_groups"] if group["cross_split"]
]
print(f"Cross-split duplicate groups: {len(cross_split_duplicates)}")
for group in cross_split_duplicates:
    print(group["images"])

# %% [markdown]
# ## Initialize and fine-tune RF-DETR Small
#
# A fresh run gets a new output directory. Resume requires that run's full
# `last.ckpt`, including optimizer/scheduler state, and its original epoch
# budget. A completed run should be opened in evaluation mode.
#
# Before fitting, check the actual RF-DETR loader's image IDs, class mapping,
# and negative targets in every split. Native loss and validation metrics go
# to `metrics.csv`; `results.csv` normalizes the sparse log rows by epoch.
# Seeds are set before constructing the new one-class head and before fitting.
# Model, backbone, and Hugging Face downloads stay under `models/`.
#
# Training writes to `outputs/runs/basketball/`. Keep the complete run folder
# when moving results between sessions: checkpoint files and their supporting
# configuration are needed together.
#
# Expected startup warnings explain that the pretrained checkpoint has 90 COCO
# classes and that Small uses patch size 16 / resized positional encodings. The
# one-class head is deliberately reinitialized, while the pretrained RF-DETR
# weights supply the fine-tuning initialization. The torchvision augmentation
# notice documents the explicitly selected, antialiased preprocessing backend.
#
# The training cell keeps its output expanded. RF-DETR's TQDM bar refreshes
# every five batches and shows the epoch, batch count, loss, and GPU memory.
# Updates can pause during validation and EMA evaluation at the end of an epoch.

# %%
run_dir = rfdetr_platform.prepare_run(settings, manifest, runtime)
print(f"Run directory: {run_dir}")

# %%
if settings.mode is not rfdetr_platform.RunMode.EVALUATE:
    training_model = rfdetr_platform.build_model(settings)
    loader_report = rfdetr_platform.verify_loader(training_model, settings, manifest)
    write_json(run_dir / "loader_check.json", loader_report)
    history = rfdetr_platform.fit_model(training_model, settings, manifest, run_dir)
    del training_model
    gc.collect()
    torch.cuda.empty_cache()
else:
    history = rfdetr_platform.read_training_history(run_dir)

# %% [markdown]
# ## Training history and selected checkpoint
#
# These are RF-DETR's own classification, box, and GIoU loss terms. Validation
# losses are explicitly enabled. Native regular/EMA AP curves describe training;
# the common evaluator below supplies the final cross-model metrics.
# Per-epoch fixed-confidence precision/recall are not synthesized from AP.
#
# | File in the run directory | Purpose |
# | --- | --- |
# | `last.ckpt` | Full state for recovery after the last completed epoch |
# | `checkpoint_best_total.pth` | Selected inference weights (regular or EMA) |
# | `checkpoint_best_total.onnx` | Static batch-one ONNX export of selected weights |
# | `checkpoint_best_regular.pth`, `checkpoint_best_ema.pth` | Epoch metadata |
# | `run_config.json`, `resolved_config.json` | Requested and resolved settings |
# | `dataset_manifest.json`, `loader_check.json` | Dataset and loader checks |
#
# The ONNX graph exposes RF-DETR's raw boxes and logits. Consumers must apply
# the detector's decoding and postprocessing to obtain final detections.
#
# RF-DETR 1.10.1 strips epoch/resolution metadata from the total checkpoint.
# The helper verifies its weights against the selected source checkpoint,
# recovers the best epoch, and explicitly restores the trained resolution.
# ONNX export runs in a fresh Python process because RF-DETR's in-process
# exporter can hang under IPython. The worker suppresses the verbose graph,
# validates a staged artifact, and atomically promotes it into the run.

# %%
loss_figure, metric_figure = rfdetr_platform.plot_history(history)
loss_figure.savefig(run_dir / "training_losses.png", bbox_inches="tight")
metric_figure.savefig(run_dir / "validation_metrics.png", bbox_inches="tight")
display_img(loss_figure, close=True)
display_img(metric_figure, close=True)

onnx_path = rfdetr_platform.ensure_onnx_model_in_subprocess(
    SUBPROJECT_ROOT, run_dir
)
print(f"ONNX model: {onnx_path}")

best_model, best_metadata = rfdetr_platform.load_best_model(run_dir)
aligned_print(best_metadata)

# %% [markdown]
# ## Validation and held-out test evaluation
#
# Reload `checkpoint_best_total.pth` before evaluating both splits. Predictions
# are exported in the original image coordinates and category IDs. All images,
# including empty ground truth and empty predictions, enter the evaluation.
#
# AP uses score >= 0.001 and COCO `maxDets=[1,10,100]`. Precision, recall, and
# F1 use score >= 0.25 and IoU >= 0.50 with one-to-one matching. Background
# metrics count false detections on negative images at the same threshold.
# A lower negative-image error rate is better. The 0.001 AP export threshold
# is separate from the 0.25 operating threshold used for precision and grids.
# RF-DETR's no-object label is discarded; no NMS is added.
#
# Each split runs two inference passes: the common evaluator and RF-DETR's
# native evaluator. `val_metrics.json` / `test_metrics.json` contain common
# metrics; `native_*_metrics.json` contain native results for traceability.
# Portable `*_predictions.json` files include the checkpoint identity and
# evaluation settings. Evaluation uses FP32 even when training used AMP.

# %%
split_metrics = {}
prediction_artifacts = {}
for split in ("val", "test"):
    metrics, artifact = rfdetr_platform.evaluate_split(
        best_model, manifest, run_dir, split, best_metadata, benchmark=BENCHMARK
    )
    split_metrics[split] = metrics
    prediction_artifacts[split] = artifact

metric_names = [
    "ap50", "ap50_95", "ar100", "precision", "recall", "f1",
    "false_positives_per_negative_image",
    "negative_image_false_positive_fraction",
]
display(pd.DataFrame({
    split: {name: metrics[name] for name in metric_names}
    for split, metrics in split_metrics.items()
}).T)

# %% [markdown]
# ## Qualitative prediction review
#
# Sample IDs are selected from ground truth, independent of the predictions:
# small annotated basketballs and negative images. Each split saves its IDs,
# labels grid, and predictions grid. Red boxes use confidence >= 0.25.

# %%
for split in ("val", "test"):
    print(f"{split}: ground truth")
    display_img(run_dir / f"{split}_batch0_labels.jpg", width=1000)
    print(f"{split}: predictions")
    display_img(run_dir / f"{split}_batch0_pred.jpg", width=1000)

# %% [markdown]
# ## Compare with YOLO11n and YOLOX-Tiny
#
# Export explicitly chosen YOLO best checkpoints with the Ultralytics and
# YOLOX project environments (see their project READMEs). Use the same
# `--output-dir` for all three exports.
# Set `RFDETR_BASELINE_EXPORT_DIR` to the resulting directory. No YOLO imports
# or retraining are needed here. Missing exports appear as unavailable;
# artifacts with a different annotation hash or smoke identity are rejected.
#
# The comparison recomputes all four sets of metrics with one evaluator.
# Treat these as preliminary fine-tuning experiments: batch size, actual
# epochs, seeds, augmentations, and architecture differ. Timing is enabled by
# default; set `RFDETR_BENCHMARK=0` to skip it. Records use batch-one FP32
# inference on the same image sequence, including preprocessing, prediction
# transfer, and postprocessing but excluding disk reads. When artifacts include
# compatible benchmarks, the comparison reports median/mean latency and
# inverse-median batch-one images/s.
# Incompatible protocols, image sequences, hosts, or accelerators are rejected.
# Historical MPS timings must not be ranked against Renku CUDA timings.
#
# Comparison CSV, JSON, and Markdown files are written under
# `outputs/comparisons/basketball_large_dataset/<run name>/`. To add YOLO
# results later, set the export directory and rerun in `evaluate` mode.

# %%
comparison_dir = (
    OUTPUT_ROOT / "comparisons" / "basketball_large_dataset" / run_dir.name
)
if settings.smoke_run:
    display(Markdown("**Smoke run:** full-dataset model comparison is disabled."))
else:
    for split in ("val", "test"):
        baseline_dir = Path(BASELINE_EXPORT_DIR) if BASELINE_EXPORT_DIR else None
        if baseline_dir is not None and not baseline_dir.is_absolute():
            baseline_dir = WORKSPACE_ROOT / baseline_dir
        artifacts = {
            model: baseline_dir / f"{model}_{split}_predictions.json"
            if baseline_dir else None
            for model in ("yolo11n", "yolox_tiny", "yolox_nano")
        }
        artifacts["rfdetr_small"] = prediction_artifacts[split]
        write_comparison(
            artifacts, SOURCE_DATASET / "annotations" / f"instances_{split}.json",
            comparison_dir, split=split,
        )
        display(Markdown((comparison_dir / f"comparison_{split}.md").read_text()))

# %% [markdown]
# ## Interpretation
#
# Inspect held-out AP together with precision/recall and negative-image errors.
# Describe misses and false detections visible in the fixed grids. Use the
# training curves and selected epoch to assess convergence or overfitting;
# do not select a different checkpoint based on test-set performance.

# %%
test = split_metrics["test"]
display(Markdown(
    f"RF-DETR Small selected **epoch {best_metadata['best_epoch']}** "
    f"({best_metadata['selected_weights']} weights) using validation AP50:95. "
    f"Held-out test AP50:95 is **{test['ap50_95']:.4f}**, precision is "
    f"**{test['precision']:.4f}**, and recall is **{test['recall']:.4f}**. "
    f"Artifacts are saved in `{run_dir}`."
))
