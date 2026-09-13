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
#     display_name: Python (Object Detection YOLOX)
#     language: python
#     name: object-detection-yolox
# ---

# %% [markdown]
# # YOLOX-Tiny on the Basketball Dataset
#
# This notebook fine-tunes YOLOX-Tiny on
# `DATA_ROOT/composed/coco_basketball_105_22_23`.
#
# The run is designed to be comparable with
# `nb02-ultralytics_yolo11n_on_basketball.py`: 640x640 input, 150 epochs by
# default, validation/test metrics, training plots, and qualitative prediction
# grids. The YOLOX-specific training/evaluation machinery lives in
# `yolox_pipeline.yolox` so the notebook stays readable.

# %%
"""
Fine-tune and evaluate YOLOX-Tiny on a customized basketball dataset.
"""

# %load_ext autoreload
# %autoreload 2

from __future__ import annotations

import torch
from detection_common import (
    Device,
    aligned_print,
    configure_stdio_relative_path,
    ensure_dir,
    increment_path,
)
from detection_common.utils.image import display as display_img
from IPython.display import Markdown, display
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
# The default run trains for 150 epochs. For a short smoke test, set
# `YOLOX_TINY_SMOKE=1` before executing the notebook.
#
# Output is quiet by default. Set `YOLOX_TINY_PROGRESS=1` for progress bars or
# `YOLOX_TINY_VERBOSE=1` for detailed dataset/eval messages and per-epoch logs.

# %%
settings = yolox_platform.training_settings_from_env(default_epochs=150)

PRETRAINED_PATH = (
    ensure_dir(WORKSPACE_ROOT / "models" / "pretrained" / "yolox")
    / "yolox_tiny.pth"
)
DATASET_DIR = DATA_ROOT / "composed" / "coco_basketball_105_22_23"

project_space = ensure_dir(OUTPUT_ROOT / "runs" / "basketball")
project_name_base = "yolox_tiny_basketball"
run_dir = ensure_dir(increment_path(project_space / project_name_base))
project_name = run_dir.name

aligned_print(
    {
        "run_dir": run_dir,
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
# The dataset has one class, `basketball`, and includes background images in each
# split. Those negative examples help test whether the detector avoids false
# basketball predictions.

# %%
dataset_summary = yolox_platform.summarize_coco_dataset(DATASET_DIR)
display(dataset_summary)

# %% [markdown]
# ## YOLOX Configuration
#
# The COCO-pretrained YOLOX-Tiny checkpoint is cached under
# `models/pretrained/yolox`. Its 80-class classification head is skipped when
# loading into this one-class basketball model; the backbone and detection
# features are still initialized from the pretrained checkpoint.

# %%
PRETRAINED_PATH = yolox_platform.ensure_pretrained_checkpoint(PRETRAINED_PATH)
exp = yolox_platform.BasketballTinyExp(
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
history = yolox_platform.fit_yolox_tiny(
    exp,
    PRETRAINED_PATH,
    run_dir,
    DATASET_DIR,
    settings,
    DEVICE,
    WORKSPACE_ROOT,
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
eval_exp = yolox_platform.BasketballTinyExp(
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
