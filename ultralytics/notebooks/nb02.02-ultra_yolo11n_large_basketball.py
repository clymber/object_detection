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
"""
Performance evaluation of pretrained YOLO11 on customized basketball dataset.
"""
# %load_ext autoreload
# %autoreload 2
# %aimport -ultralytics

import os
from pathlib import Path
from typing import cast

# This must run before any library imports PyTorch, including Ultralytics.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from detection_common import configure_stdio_relative_path
from ultralytics_pipeline.config import (
    DATA_ROOT,
    OUTPUT_ROOT,
    WORKSPACE_ROOT,
)

# Display project paths relative to project root directory for consistency.
configure_stdio_relative_path(WORKSPACE_ROOT)

# %%
from detection_common import (
    Device,
    aligned_print,
    ensure_dir,
)
from detection_common.utils.image import display as display_img
from ultralytics_pipeline import ultralytics as ultralitics_platform

# Must be called before importing ultralytics.
ultralitics_platform.configure_privacy()
from ultralytics import YOLO  # noqa: E402

# %%
PRETRAINED_DIR = ensure_dir(WORKSPACE_ROOT / "models" / "pretrained" / "ultralytics")
DATASET_DIR = ensure_dir(DATA_ROOT)
DATA_YAML = DATASET_DIR / "composed" / "yolo_basketball_11501_1156_1395" / "data.yaml"

# %% [markdown]
# ## Fine-tune Ultrlytics YOLO11 on a custom dataset.

# %%
project_space = OUTPUT_ROOT / "runs" / "basketball"
project_name_base = "yolo11n_basketball_large_dataset"
# Avoid container shared-memory exhaustion by default. Hosts with a larger /dev/shm
# allocation can set ULTRALYTICS_WORKERS to a positive integer.
DATALOADER_WORKERS = int(os.environ.get("ULTRALYTICS_WORKERS", "0"))
if DATALOADER_WORKERS < 0:
    raise ValueError("ULTRALYTICS_WORKERS must be zero or greater")

resume_checkpoint: Path | None = None
# resume_checkpoint: Path | None = (
#     project_space
#     / "yolo11n_basketball_large_dataset-2"
#     / "weights"
#     / "last.pt"
# )

if resume_checkpoint is None:
    basketball_model = YOLO(PRETRAINED_DIR / "yolo11n.pt")
    results = basketball_model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        device=Device.auto_choose(),
        project=str(project_space),
        name=project_name_base,
        patience=25,
        batch=16,
        workers=DATALOADER_WORKERS,
        cache=False,
    )
else:
    if not resume_checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")

    basketball_model = YOLO(resume_checkpoint)
    results = basketball_model.train(
        resume=True,
        device=Device.auto_choose(),
        batch=16,
        workers=DATALOADER_WORKERS,
        cache=False,
    )
results = cast(ultralitics_platform.TrainingResult, results)
run_dir = Path(results.save_dir)
print(f"Training run directory: {run_dir}")


# %% [markdown]
# Downstream cells use the actual Ultralytics save directory reported by
# `results.save_dir`. This keeps plots, metrics, and checkpoint evaluation aligned if
# Ultralytics increments the run name.

# %% [markdown]
# To continue an interrupted run, set `resume_checkpoint` to that run's
# `weights/last.pt` path and rerun the training cell. Ultralytics restores the run
# configuration, optimizer state, and completed epoch from the checkpoint. Leave it
# as `None` to start a new run from the pretrained YOLO11n weights.

# %% [markdown]
# ## Plot the training history

# %%
import matplotlib.pyplot as plt
import pandas as pd


def read_training_history(results_csv):
    """
    Read an Ultralytics YOLO training results.csv file.
    """
    history = pd.read_csv(results_csv)
    history.columns = history.columns.str.strip()
    return history


def plot_train_val_losses(history):
    """
    Plot train and validation YOLO losses.
    """
    loss_names = ["box_loss", "cls_loss", "dfl_loss"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)

    for axis, loss_name in zip(axes, loss_names, strict=True):
        axis.plot(history["epoch"], history[f"train/{loss_name}"], label="train")
        axis.plot(history["epoch"], history[f"val/{loss_name}"], label="validation")
        axis.set_title(loss_name)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.tight_layout()
    return fig, axes


def plot_detection_metrics(history):
    """
    Plot YOLO validation detection metrics.
    """
    metric_columns = [
        ("precision", "metrics/precision(B)"),
        ("recall", "metrics/recall(B)"),
        ("mAP50", "metrics/mAP50(B)"),
        ("mAP50-95", "metrics/mAP50-95(B)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    for axis, (metric_name, column) in zip(axes.ravel(), metric_columns, strict=True):
        axis.plot(history["epoch"], history[column])
        axis.set_title(metric_name)
        axis.set_xlabel("epoch")
        axis.set_ylabel("metric")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)

    fig.tight_layout()
    return fig, axes

# %%
print(f"Using run directory: {run_dir}")
history = read_training_history(run_dir / "results.csv")

fig, axes = plot_train_val_losses(history)
display_img(fig, close=True)

# %% [markdown]
# Compare the training and validation curves after the run. Falling losses in both
# sets indicate continued learning. A widening gap, with training loss improving while
# validation loss stalls or rises, indicates overfitting.

# %%
fig, axes = plot_detection_metrics(history)
display_img(fig, close=True)

# %%
selection_metric = "metrics/mAP50-95(B)"
best_history_row = cast(
    pd.Series,
    history.loc[history[selection_metric].idxmax()],
)

print(f"Best validation epoch by {selection_metric}:")
aligned_print({
    "Epoch": int(best_history_row["epoch"]),
    "Precision": best_history_row["metrics/precision(B)"],
    "Recall": best_history_row["metrics/recall(B)"],
    "mAP50": best_history_row["metrics/mAP50(B)"],
    "mAP50-95": best_history_row[selection_metric],
})

# %% [markdown]
# ### Training summary
#
# Use the reported best validation epoch and the plots above to characterize the run.
# Compare precision and recall to identify whether false positives or missed
# basketballs are the larger issue, then confirm the behavior on the held-out test set
# below.


# %%
BEST_MODEL_PATH = run_dir / "weights" / "best.pt"
eval_model = YOLO(BEST_MODEL_PATH)

# %% [markdown]
# ## Export Best Checkpoint to ONNX

# %%
onnx_model_path = Path(
    eval_model.export(
        format="onnx",
        imgsz=640,
        batch=1,
        device="cpu",
        dynamic=False,
        simplify=False,
    )
)
if not onnx_model_path.is_file():
    raise FileNotFoundError(f"ONNX export not found: {onnx_model_path}")
print(f"Exported ONNX model: {onnx_model_path}")

# %% [markdown]
# ## Validation and Test Metrics

# %%
validation_metrics = eval_model.val(
    data=DATA_YAML,
    imgsz=640,
    device=Device.auto_choose(),
    split="val",
    plots=True,
    project=str(project_space),
    name=f"{run_dir.name}_val",
)
test_metrics = eval_model.val(
    data=DATA_YAML,
    imgsz=640,
    device=Device.auto_choose(),
    split="test",
    plots=True,
    project=str(project_space),
    name=f"{run_dir.name}_test",
)

# %% [markdown]
# ### Validation metrics summary
# %%
print("Validation metrics:")
aligned_print({
    "Precision": validation_metrics.box.mp,
    "Recall": validation_metrics.box.mr,
    "mAP50": validation_metrics.box.map50,
    "mAP50-95": validation_metrics.box.map,
})
print("\nSpeed ms/image:")
aligned_print(validation_metrics.speed)

print("\nTest metrics:")
aligned_print({
    "Precision": test_metrics.box.mp,
    "Recall": test_metrics.box.mr,
    "mAP50": test_metrics.box.map50,
    "mAP50-95": test_metrics.box.map,
})
print("\nSpeed ms/image:")
aligned_print(test_metrics.speed)

# %% [markdown]
# ### Sample prediction VS ground truth

# %%
validation_output_dir = Path(validation_metrics.save_dir)
test_output_dir = Path(test_metrics.save_dir)

print("Validation ground truth sample images:")
display_img(validation_output_dir / "val_batch0_labels.jpg", width=640)
print("Validation predicted sample images:")
display_img(validation_output_dir / "val_batch0_pred.jpg", width=640)

print("Test ground truth sample images:")
display_img(test_output_dir / "val_batch0_labels.jpg", width=640)
print("Test predicted sample images:")
display_img(test_output_dir / "val_batch0_pred.jpg", width=640)
