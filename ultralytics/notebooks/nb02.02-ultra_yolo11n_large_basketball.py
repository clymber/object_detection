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
    allocate_run_directory,
    aligned_print,
    ensure_dir,
)
from detection_common.utils.image import display as display_img

from ultralytics_pipeline import ultralytics as ultralitics_platform
from ultralytics_pipeline import producer

# Must be called before importing ultralytics.
ultralitics_platform.configure_privacy()
from ultralytics import YOLO  # noqa: E402

# %%
PRETRAINED_DIR = ensure_dir(WORKSPACE_ROOT / "models" / "pretrained" / "ultralytics")
DATASET_DIR = ensure_dir(DATA_ROOT)
SOURCE_DATASET_DIR = DATASET_DIR / "composed" / "coco_basketball"
YOLO_DATASET_DIR = DATASET_DIR / "composed" / "yolo_basketball"

# %% [markdown]
# ## Fine-tune Ultrlytics YOLO11 on a custom dataset.

# %%
# Avoid container shared-memory exhaustion by default. Hosts with a larger /dev/shm
# allocation can set ULTRALYTICS_WORKERS to a positive integer.
DATALOADER_WORKERS = int(os.environ.get("ULTRALYTICS_WORKERS", "0"))
if DATALOADER_WORKERS < 0:
    raise ValueError("ULTRALYTICS_WORKERS must be zero or greater")


resume_checkpoint: Path | None = None
# resume_checkpoint: Path | None = (
#     OUTPUT_ROOT
#     / "runs"
#     / "basketball"
#     / "yolo11n_20260919T190000"
#     / "weights"
#     / "last.pt"
# )

training_settings = (
    producer.settings_from_environment(workers=DATALOADER_WORKERS)
    if resume_checkpoint is None
    else producer.settings_from_run(resume_checkpoint.resolve().parent.parent)
)
dataset_paths = producer.resolve_dataset_paths(
    SOURCE_DATASET_DIR,
    YOLO_DATASET_DIR,
    smoke_run=training_settings.smoke_run,
)
DATA_YAML = dataset_paths.yolo_dir / "data.yaml"

if resume_checkpoint is None:
    allocation = allocate_run_directory(OUTPUT_ROOT, "basketball", "yolo11n")
    run_dir = allocation.path
    basketball_model = YOLO(PRETRAINED_DIR / "yolo11n.pt")
    original_utc = allocation.created_at
else:
    if not resume_checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
    run_dir = resume_checkpoint.resolve().parent.parent
    basketball_model = YOLO(resume_checkpoint)
    original_utc = None

results = producer.train_pinned_run(
    basketball_model,
    run_dir,
    dataset_paths,
    training_settings,
    original_utc=original_utc,
    resumed=resume_checkpoint is not None,
)
results = cast(ultralitics_platform.TrainingResult, results)
ultralitics_platform.assert_run_directory(results.save_dir, run_dir)
print(f"Training run directory: {run_dir}")


# %% [markdown]
# Downstream cells use the directory reserved before training. The assertions above
# ensure Ultralytics has not changed that path.

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
bundle_manifest = producer.recover_and_publish(
    run_dir,
    dataset_paths,
    training_settings,
    load_model=YOLO,
)
print(f"Published bundle generation: {bundle_manifest['generation']}")

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
    project=str(run_dir / "evaluation"),
    name="val",
    exist_ok=True,
)
test_metrics = eval_model.val(
    data=DATA_YAML,
    imgsz=640,
    device=Device.auto_choose(),
    split="test",
    plots=True,
    project=str(run_dir / "evaluation"),
    name="test",
    exist_ok=True,
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
