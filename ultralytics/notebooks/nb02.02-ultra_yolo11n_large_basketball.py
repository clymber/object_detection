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

from dataset_builder import Dataset as VersionedDataset

# This must run before any library imports PyTorch, including Ultralytics.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from detection_common import configure_stdio_relative_path

from ultralytics_pipeline.config import (
    DATA_ROOT,
    WORKSPACE_ROOT,
)

# Display project paths relative to project root directory for consistency.
configure_stdio_relative_path(WORKSPACE_ROOT)

# %%
from detection_common import (
    aligned_print,
)
from detection_common.utils.image import display as display_img

from ultralytics_pipeline import producer
from ultralytics_pipeline import ultralytics as ultralitics_platform

# Must be called before importing ultralytics.
ultralitics_platform.configure_privacy()
from ultralytics import YOLO  # noqa: E402

# %% [markdown]
# ## Experiment settings
#
# `basketball-v0.1` preserves reviewed source splits and background images.
# Full runs use 640-pixel input and up to 100 epochs. Set `ULTRALYTICS_SMOKE=1`
# for two complete epochs on the shared 16/8/8 subset. Set `ULTRALYTICS_WORKERS`
# for loader workers (default 0), or `ULTRALYTICS_DEVICE` for an explicit device.
# Set `resume_checkpoint` to an interrupted run's `weights/last.pt` to resume.

# %%
DATASET_NAME = "basketball-v0.1"
dataset = VersionedDataset(DATA_ROOT / "composed" / DATASET_NAME)
resume_checkpoint: Path | None = None
experiment = producer.prepare_experiment(
    dataset,
    resume_checkpoint=resume_checkpoint,
)
aligned_print(experiment.summary)
run_dir = experiment.run_dir
dataset_paths = experiment.paths
training_settings = experiment.settings
DATA_YAML = dataset_paths.yolo_dir / "data.yaml"

# %%
results = producer.train_pinned_run(
    experiment.model,
    run_dir,
    dataset_paths,
    training_settings,
    original_utc=experiment.original_utc,
    resumed=experiment.resumed,
)
print(f"Training run directory: {run_dir}")

# %% [markdown]
# ## Plot the training history

# %%
import pandas as pd

# %%
print(f"Using run directory: {run_dir}")
history = ultralitics_platform.read_training_history(run_dir / "results.csv")

fig, axes = ultralitics_platform.plot_train_val_losses(history)
display_img(fig, close=True)

# %% [markdown]
# Compare the training and validation curves after the run. Falling losses in both
# sets indicate continued learning. A widening gap, with training loss improving while
# validation loss stalls or rises, indicates overfitting.

# %%
fig, axes = ultralitics_platform.plot_detection_metrics(history)
display_img(fig, close=True)

# %%
selection_metric = "metrics/mAP50-95(B)"
best_history_row = cast(
    pd.Series,
    history.loc[history[selection_metric].idxmax()],
)

print(f"Best validation epoch by {selection_metric}:")
aligned_print(
    {
        "Epoch": int(best_history_row["epoch"]),
        "Precision": best_history_row["metrics/precision(B)"],
        "Recall": best_history_row["metrics/recall(B)"],
        "mAP50": best_history_row["metrics/mAP50(B)"],
        "mAP50-95": best_history_row[selection_metric],
    }
)

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
print(f"Exported ONNX model: {onnx_model_path}")

# %% [markdown]
# ## Validation and Test Metrics

# %%
validation_metrics = eval_model.val(
    data=DATA_YAML,
    imgsz=640,
    device=training_settings.device,
    split="val",
    plots=True,
    project=str(run_dir / "evaluation"),
    name="val",
    exist_ok=True,
)
test_metrics = eval_model.val(
    data=DATA_YAML,
    imgsz=640,
    device=training_settings.device,
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
aligned_print(
    {
        "Precision": validation_metrics.box.mp,
        "Recall": validation_metrics.box.mr,
        "mAP50": validation_metrics.box.map50,
        "mAP50-95": validation_metrics.box.map,
    }
)
print("\nSpeed ms/image:")
aligned_print(validation_metrics.speed)

print("\nTest metrics:")
aligned_print(
    {
        "Precision": test_metrics.box.mp,
        "Recall": test_metrics.box.mr,
        "mAP50": test_metrics.box.map50,
        "mAP50-95": test_metrics.box.map,
    }
)
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
