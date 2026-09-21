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
# and up to 100 epochs. Cross-model comparison now lives in
# [nb05.01](../../evaluation/notebooks/nb05.01-basketball_models_comparison.py),
# which consumes published bundles without loading detector frameworks.

# %%
from __future__ import annotations

import gc

import pandas as pd
import torch
from dataset_builder import prepare_smoke_layouts, smoke_layout_directory
from dataset_builder.rfdetr import prepare_coco_dataset, summarize_dataset
from detection_common import aligned_print, configure_stdio_relative_path
from detection_common.utils.image import display as display_img
from detection_common.utils.json_io import read_json, write_json
from IPython.display import Markdown, display
from rfdetr_pipeline import rfdetr as rfdetr_platform
from rfdetr_pipeline.config import DATA_ROOT, WORKSPACE_ROOT

configure_stdio_relative_path(WORKSPACE_ROOT)
cache_paths = rfdetr_platform.configure_caches(WORKSPACE_ROOT)
runtime = rfdetr_platform.runtime_report()
aligned_print(runtime)

# %% [markdown]
# ## Experiment settings
#
# Leave `RUN_OVERRIDES` empty to use the defaults or `RFDETR_*` variables from
# the active shell. Its values take precedence without changing the kernel
# environment. A smoke run is excluded from the full comparison.

# %%
RUN_OVERRIDES = {
    # "smoke_run": True,
    # "batch_size": 4,
    # "gradient_checkpointing": True,
    # "early_stopping_patience": 10,
    # "run_dir": "outputs/runs/basketball/rfdetr_small_20260919T190000",
}
settings = rfdetr_platform.settings_from_env(overrides=RUN_OVERRIDES)
FULL_SOURCE_DATASET = DATA_ROOT / "composed" / "coco_basketball"
if settings.smoke_run:
    smoke_root = smoke_layout_directory(FULL_SOURCE_DATASET)
    smoke_record = prepare_smoke_layouts(FULL_SOURCE_DATASET, smoke_root)
    if smoke_record["smoke_limits"] != {"train": 16, "val": 8, "test": 8}:
        raise ValueError("RF-DETR smoke layout must use the complete 16/8/8 splits")
    SOURCE_DATASET = smoke_root / "coco"
    DERIVED_DATASET = smoke_root / "rfdetr"
else:
    SOURCE_DATASET = FULL_SOURCE_DATASET
    DERIVED_DATASET = DATA_ROOT / "processed" / "rfdetr" / "basketball"

aligned_print(
    {
        **vars(settings),
        "samples_per_optimizer_step": settings.samples_per_optimizer_step,
        "source_dataset": SOURCE_DATASET,
        "derived_dataset": DERIVED_DATASET,
    }
)

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
if settings.smoke_run:
    manifest = read_json(DERIVED_DATASET / "manifest.json")
else:
    manifest = prepare_coco_dataset(
        SOURCE_DATASET,
        DERIVED_DATASET,
        expected_counts={"train": 11394, "val": 1155, "test": 1395},
        category_id=None,
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
# budget. A completed run is recovered with the postprocess CLI instead.
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
training_model = rfdetr_platform.build_model(settings)
loader_report = rfdetr_platform.verify_loader(training_model, settings, manifest)
write_json(run_dir / "loader_check.json", loader_report)
history = rfdetr_platform.fit_model(training_model, settings, manifest, run_dir)
del training_model
gc.collect()
torch.cuda.empty_cache()

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
# Recovery validates the immutable protocol and dataset identity, regenerates
# native outputs, and atomically republishes both prediction artifacts without
# fitting or appending an attempt. It reuses a valid ONNX graph or replaces it
# through the verified exporter before evaluating selected weights.

# %%
postprocess = rfdetr_platform.postprocess_run(run_dir)
history = postprocess["history"]
best_metadata = postprocess["best_metadata"]
display_img(run_dir / "training_losses.png", width=1000)
display_img(run_dir / "validation_metrics.png", width=1000)
print(f"ONNX model: {postprocess['onnx_path']}")
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
split_metrics = postprocess["split_metrics"]
prediction_artifacts = postprocess["prediction_artifacts"]

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
