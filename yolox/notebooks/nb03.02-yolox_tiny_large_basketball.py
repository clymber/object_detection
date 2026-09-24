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
# # YOLOX-Tiny on the Large Basketball Dataset
#
# This notebook fine-tunes YOLOX-Tiny on
# `DATA_ROOT/composed/coco_basketball`.
#
# The run is designed to be comparable with `nb02.02-ultra_yolo11n_large_basketball.py`:
# 640x640 input, 100 epochs by default, validation/test metrics, training plots, and
# qualitative prediction grids. The YOLOX-specific training/evaluation machinery lives
# in `yolox_pipeline.yolox` so the notebook stays readable.

# %%
"""
Fine-tune and evaluate YOLOX-Tiny on the large basketball dataset.
"""
# %load_ext autoreload
# %autoreload 2
# %aimport -torch, -IPython

# %%
from __future__ import annotations  # noqa: F404

import os

import torch
from detection_common import (
    Device,
    aligned_print,
    allocate_run_directory,
    configure_stdio_relative_path,
    ensure_dir,
)
from detection_common.utils.image import display as display_img
from IPython.display import Markdown, display

from yolox_pipeline import producer
from yolox_pipeline import yolox as yolox_platform
from yolox_pipeline.config import (
    DATA_ROOT,
    OUTPUT_ROOT,
    WORKSPACE_ROOT,
)

configure_stdio_relative_path(WORKSPACE_ROOT)

DEVICE = torch.device(Device.auto_choose())
if DEVICE.type == "mps":
    yolox_platform.patch_mps_compatibility()

# %% [markdown]
# ## Experiment Setup
#
# The default run trains for 100 epochs. For a short smoke test, set
# `YOLOX_TINY_SMOKE=1` before executing the notebook.
#
# Output is quiet by default. Set `YOLOX_TINY_PROGRESS=1` for progress bars or
# `YOLOX_TINY_VERBOSE=1` for detailed dataset/eval messages and per-epoch logs.
#
# A fresh run is the default. To resume one in place, set `RESUME_RUN_DIR` below to its
# run directory. Relative paths are resolved from `OUTPUT_ROOT`. The recoverable
# `weights/last_ckpt.pth` checkpoint is restored before any further training.

# %%
# Standard Python environment settings for this notebook.
# Uncomment values to override the defaults before reading the settings.
#
# os.environ["YOLOX_TINY_VERBOSE"] = "1"
os.environ["YOLOX_TINY_PROGRESS"] = "1"
# os.environ["YOLOX_TINY_SMOKE"] = "1"

RESUME_RUN_DIR: str | None = None
# RESUME_RUN_DIR = "runs/basketball/yolox_tiny_20260920T183040"

if RESUME_RUN_DIR is None:
    os.environ.pop("YOLOX_TINY_RESUME_RUN", None)
else:
    os.environ["YOLOX_TINY_RESUME_RUN"] = RESUME_RUN_DIR


# %%
settings = yolox_platform.training_settings_from_env(default_epochs=100)

PRETRAINED_PATH = (
    ensure_dir(WORKSPACE_ROOT / "models" / "pretrained" / "yolox") / "yolox_tiny.pth"
)
if settings.run_mode is yolox_platform.RunMode.RESUME:
    producer_settings = producer.settings_from_run(settings.resolved_resume_run_dir)
    settings = producer_settings.training
else:
    producer_settings = producer.ProducerSettings(settings)
dataset_paths = producer.resolve_dataset_paths(
    DATA_ROOT / "composed" / "coco_basketball",
    DATA_ROOT / "composed" / "yolo_basketball",
    smoke_run=settings.smoke_run,
)
DATASET_DIR = dataset_paths.source_dir
model_variant = producer.variant("tiny")

run_mode = settings.run_mode

if run_mode is yolox_platform.RunMode.FRESH:
    allocation = allocate_run_directory(OUTPUT_ROOT, "basketball", "yolox_tiny")
    run_dir = allocation.path
    original_utc = allocation.created_at
else:
    run_dir = settings.resolved_resume_run_dir
    original_utc = None
    resume_checkpoint = run_dir / "weights" / "last_ckpt.pth"
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Resume run directory not found: {run_dir}")
    if not resume_checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
project_space = run_dir.parent
project_name = run_dir.name

aligned_print(
    {
        "run_dir": run_dir,
        "run_mode": run_mode,
        "device": DEVICE,
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "image_size": settings.image_size,
        "train_batch_limit": settings.train_batch_limit,
        "show_progress": settings.show_progress,
        "verbose_output": settings.verbose_output,
        "smoke_run": settings.smoke_run,
    }
)

# %% [markdown]
# ## Dataset Summary
#
# The dataset has one class, `basketball`, and uses independent basketball sources for
# the train, validation, and test splits. It also includes tennis images as negative
# examples to test whether the detector avoids false basketball predictions.

# %%
dataset_summary = yolox_platform.summarize_coco_dataset(DATASET_DIR)
display(dataset_summary)

# %% [markdown]
# ## YOLOX Configuration
#
# The COCO-pretrained YOLOX-Tiny checkpoint is cached under `models/pretrained/yolox`.
# Its 80-class classification head is skipped when loading into this one-class
# basketball model; the backbone and detection features are still initialized from the
# pretrained checkpoint.

# %%
PRETRAINED_PATH = yolox_platform.ensure_pretrained_checkpoint(PRETRAINED_PATH)
exp = yolox_platform.YOLOXTinyExp(
    dataset_dir=DATASET_DIR,
    output_dir=project_space,
    max_epoch=settings.epochs,
    image_size=settings.image_size,
    project_name=project_name,
    seed=settings.seed,
)

aligned_print(
    {
        "checkpoint": PRETRAINED_PATH,
        "experiment": exp.exp_name,
        "classes": exp.num_classes,
        "input_size": exp.input_size,
        "test_size": exp.test_size,
        "no_aug_epochs": exp.no_aug_epochs,
        "dataset": DATASET_DIR,
    }
)

# %% [markdown]
# ## Fine-Tune YOLOX-Tiny
#
# This cell is collapsed by default because the full run can take a while.
# Metrics are written to `results.csv`; the best checkpoint is selected by
# validation `mAP50-95`.

# %% jupyter={"outputs_hidden": true}
history = producer.fit_run(
    run_dir,
    dataset_paths,
    producer_settings,
    model_variant,
    exp,
    PRETRAINED_PATH,
    DEVICE,
    WORKSPACE_ROOT,
    original_utc=original_utc,
    resumed=run_mode is yolox_platform.RunMode.RESUME,
)

# %% [markdown]
# ## Plot the Training History
#
# YOLOX uses different loss terms from Ultralytics YOLO, but the validation
# losses are plotted alongside training losses when available. The validation
# metrics are kept comparable: precision, recall, mAP50, and mAP50-95.

# %%
history = yolox_platform.read_training_history(run_dir / "results.csv")

fig, axes = yolox_platform.plot_train_losses(history)
display_img(fig, close=True)

fig, axes = yolox_platform.plot_detection_metrics(history)
display_img(fig, close=True)

# %% [markdown]
# ## Export the Best Checkpoint to ONNX
#
# Export the selected EMA weights through a fresh CPU model. The ONNX graph uses
# a fixed batch-one input at the experiment test size and leaves YOLOX decoding
# and non-maximum suppression to the deployment runtime.

# %%
best_model_path = run_dir / "weights" / "best_ckpt.pth"
eval_exp = yolox_platform.YOLOXTinyExp(
    dataset_dir=DATASET_DIR,
    output_dir=project_space,
    max_epoch=settings.epochs,
    image_size=settings.image_size,
    project_name=project_name,
    seed=settings.seed,
)

# %%
onnx_model_path = yolox_platform.export_trained_model_to_onnx(
    eval_exp,
    best_model_path,
    run_dir / "weights" / "best_ckpt.onnx",
)
print(f"Exported ONNX model: {onnx_model_path}")

# %% [markdown]
# ## Validation and Test Metrics
#
# Reload the best checkpoint and evaluate it on both validation and held-out
# test splits.

# %%
best_model = yolox_platform.load_trained_model(eval_exp, best_model_path, DEVICE)

# %% [markdown]
# ## Protocol Bundle
#
# Reload the selected checkpoint through the same YOLOX adapter and atomically
# publish protocol-bearing validation and test prediction artifacts.

# %%
bundle_manifest = producer.recover_and_publish(
    run_dir,
    dataset_paths,
    producer_settings,
    model_variant,
    DEVICE,
)
print(bundle_manifest["generation"])

validation_metrics = yolox_platform.evaluate_model(
    best_model,
    eval_exp,
    "val",
    settings.batch_size,
    DEVICE,
    progress=settings.show_progress,
    verbose=settings.verbose_output,
)
test_metrics = yolox_platform.evaluate_model(
    best_model,
    eval_exp,
    "test",
    settings.batch_size,
    DEVICE,
    progress=settings.show_progress,
    verbose=settings.verbose_output,
)

print("Validation metrics:")
aligned_print(
    {
        "Precision": validation_metrics["metrics/precision(B)"],
        "Recall": validation_metrics["metrics/recall(B)"],
        "mAP50": validation_metrics["metrics/mAP50(B)"],
        "mAP50-95": validation_metrics["metrics/mAP50-95(B)"],
        "Forward ms/image": validation_metrics["speed/forward_ms"],
        "NMS ms/image": validation_metrics["speed/nms_ms"],
    }
)

print("\nTest metrics:")
aligned_print(
    {
        "Precision": test_metrics["metrics/precision(B)"],
        "Recall": test_metrics["metrics/recall(B)"],
        "mAP50": test_metrics["metrics/mAP50(B)"],
        "mAP50-95": test_metrics["metrics/mAP50-95(B)"],
        "Forward ms/image": test_metrics["speed/forward_ms"],
        "NMS ms/image": test_metrics["speed/nms_ms"],
    }
)

if settings.verbose_output:
    print("\nCOCO test summary:")
    print(test_metrics["summary"])

# %% [markdown]
# ## Qualitative Prediction Review
#
# Save and display ground-truth and prediction grids for validation and test
# images, matching the style of the sample artifacts in `nb02`.

# %%
val_labels_path, val_preds_path = yolox_platform.save_sample_visualizations(
    best_model,
    eval_exp,
    "val",
    run_dir,
    DEVICE,
    verbose=settings.verbose_output,
)
test_labels_path, test_preds_path = yolox_platform.save_sample_visualizations(
    best_model,
    eval_exp,
    "test",
    run_dir,
    DEVICE,
    verbose=settings.verbose_output,
)

print("Validation ground truth sample images:")
display_img(val_labels_path, width=900)
print("Validation predicted sample images:")
display_img(val_preds_path, width=900)

print("Test ground truth sample images:")
display_img(test_labels_path, width=900)
print("Test predicted sample images:")
display_img(test_preds_path, width=900)

# %% [markdown]
# ## Final Interpretation and Next Steps
#
# The final report is generated from the observed metrics so the notebook reads
# like an experiment log after execution.

# %%
display(Markdown(yolox_platform.format_final_interpretation(history, test_metrics)))
