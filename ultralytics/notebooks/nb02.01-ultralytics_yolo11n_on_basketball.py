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

# Must be called before importing ultralytics.
ultralitics_platform.configure_privacy()
from ultralytics import YOLO  # noqa: E402

# %%
PRETRAINED_DIR = ensure_dir(WORKSPACE_ROOT / "models" / "pretrained" / "ultralytics")
DATASET_DIR = ensure_dir(DATA_ROOT)
DATA_YAML = DATASET_DIR / "composed" / "yolo_basketball_small" / "data.yaml"

# %% [markdown]
# ## Fine-tune Ultrlytics YOLO11 on a custom dataset.

# %%
allocation = allocate_run_directory(OUTPUT_ROOT, "basketball_small", "yolo11n")
run_dir = allocation.path
basketball_model = YOLO(PRETRAINED_DIR / "yolo11n.pt")
basketball_model.add_callback(
    "on_pretrain_routine_start",
    ultralitics_platform.reserved_run_callback(run_dir),
)

results = basketball_model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    device=Device.auto_choose(),
    save_dir=str(run_dir),
    project=str(run_dir.parent),
    name=run_dir.name,
    exist_ok=True,
)
results = cast(ultralitics_platform.TrainingResult, results)
ultralitics_platform.assert_run_directory(results.save_dir, run_dir)
print(f"Training run directory: {run_dir}")


# %% [markdown]
# Downstream cells use the directory reserved before training. The assertions above
# ensure Ultralytics has not changed that path.

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
# The model is learning: all three training losses (box_loss, cls_loss,
# dfl_loss) decrease steadily through all epochs. That means the network is
# fitting the training data and improving localization/classification on the
# train set.  
#
# Validation behavior is noisier than training. It's as expected, because the
# validation set is tiny currently. Validation losses remain clearly higher
# than training losses. This suggests a generalization gap: the model fits the
# training set better than the validation set.

# %%
fig, axes = plot_detection_metrics(history)
display_img(fig, close=True)

# %% [markdown]
# The `precision` ends high, while the `recall` remains modest. So the current
# checkpoint of the model usually predicts a basketball object correctly, but
# missing too many objects.
#
# The best checkpoint is likely near the late plateau, around epoch ~90 rather
# than the final epoch being dramatically better. From the CSV we inspected
# earlier, best mAP50-95 was around epoch 92.

# %% [markdown]
# ### Training summary
#
# The YOLO11n fine-tune is working, but performance is recall-limited. The model
# has learned useful detections, but it is missing too many objects. Next step
# focus on improving recall: inspect missed validation predictions, check label
# quality, add more varied examples, and tune inference confidence/NMS thresholds.


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
    project=str(run_dir / "evaluation"),
    name=f"{run_dir.name}_val",
)
test_metrics = eval_model.val(
    data=DATA_YAML,
    imgsz=640,
    device=Device.auto_choose(),
    split="test",
    plots=True,
    project=str(run_dir / "evaluation"),
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
