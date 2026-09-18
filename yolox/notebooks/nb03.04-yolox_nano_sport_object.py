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
# # YOLOX-Nano on the Sport Object Dataset
#
# This notebook fine-tunes YOLOX-Nano on `DATA_ROOT/composed/coco_sport_object`.
# The two classes are `football` at index 0 and `basketball` at index 1. It follows
# the established Nano workflow: training, validation/test metrics, ONNX export, and
# qualitative prediction grids.

# %%
"""
Fine-tune and evaluate YOLOX-Nano on the sport object dataset.
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
	configure_stdio_relative_path,
	ensure_dir,
	increment_path,
)
from detection_common.utils.image import display as display_img
from IPython.display import Markdown, display

from yolox_pipeline import yolox as yolox_platform
from yolox_pipeline.config import DATA_ROOT, OUTPUT_ROOT, WORKSPACE_ROOT

configure_stdio_relative_path(WORKSPACE_ROOT)

DEVICE = torch.device(Device.auto_choose())
if DEVICE.type == "mps":
	yolox_platform.patch_mps_compatibility()

# %% [markdown]
# ## Experiment Setup
#
# The default run trains for 100 epochs. Set `YOLOX_NANO_SPORT_SMOKE=1` for a short
# smoke test. Set `YOLOX_NANO_SPORT_PROGRESS=1` for progress bars, or
# `YOLOX_NANO_SPORT_VERBOSE=1` for detailed training and evaluation output.
#
# Set `RESUME_RUN_DIR` to resume in place from a run directory containing
# `weights/last_ckpt.pth`.

# %%
# os.environ["YOLOX_NANO_SPORT_VERBOSE"] = "1"
os.environ["YOLOX_NANO_SPORT_PROGRESS"] = "1"
# os.environ["YOLOX_NANO_SPORT_SMOKE"] = "1"

RESUME_RUN_DIR: str | None = None
# RESUME_RUN_DIR = "outputs/runs/sport_object/yolox_nano_sport_object"

if RESUME_RUN_DIR is None:
	os.environ.pop("YOLOX_NANO_SPORT_RESUME_RUN", None)
else:
	os.environ["YOLOX_NANO_SPORT_RESUME_RUN"] = RESUME_RUN_DIR

# %%
CLASS_NAMES = ("football", "basketball")
settings = yolox_platform.training_settings_from_env(
	default_epochs=100,
	env_prefix="YOLOX_NANO_SPORT",
)

PRETRAINED_PATH = (
	ensure_dir(WORKSPACE_ROOT / "models" / "pretrained" / "yolox") / "yolox_nano.pth"
)
DATASET_DIR = DATA_ROOT / "composed" / "coco_sport_object"
project_space = ensure_dir(OUTPUT_ROOT / "runs" / "sport_object")
project_name_base = "yolox_nano_sport_object"
run_mode = settings.run_mode

if run_mode is yolox_platform.RunMode.FRESH:
	run_dir = ensure_dir(increment_path(project_space / project_name_base))
else:
	run_dir = settings.resolved_resume_run_dir
	resume_checkpoint = run_dir / "weights" / "last_ckpt.pth"
	if not run_dir.is_dir():
		raise NotADirectoryError(f"Resume run directory not found: {run_dir}")
	if not resume_checkpoint.is_file():
		raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
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
# The composed dataset preserves the source train, validation, and test assignments.
# It contains football and basketball annotations plus retained background images.

# %%
dataset_summary = yolox_platform.summarize_coco_dataset(DATASET_DIR)
display(dataset_summary)

# %% [markdown]
# ## YOLOX Configuration
#
# The COCO-pretrained Nano checkpoint initializes the depthwise backbone and
# detection features. Its 80-class head is skipped when loading the two-class model.

# %%
PRETRAINED_PATH = yolox_platform.ensure_pretrained_checkpoint(
	PRETRAINED_PATH,
	yolox_platform.YOLOX_NANO_WEIGHTS_URL,
)
exp = yolox_platform.BasketballNanoExp(
	dataset_dir=DATASET_DIR,
	output_dir=project_space,
	max_epoch=settings.epochs,
	image_size=settings.image_size,
	project_name=project_name,
	class_names=CLASS_NAMES,
	seed=settings.seed,
)

aligned_print(
	{
		"checkpoint": PRETRAINED_PATH,
		"experiment": exp.exp_name,
		"classes": exp.num_classes,
		"class_names": CLASS_NAMES,
		"input_size": exp.input_size,
		"test_size": exp.test_size,
		"no_aug_epochs": exp.no_aug_epochs,
		"dataset": DATASET_DIR,
	}
)

# %% [markdown]
# ## Fine-Tune YOLOX-Nano
#
# Metrics are written to `results.csv`; the best checkpoint is selected by validation
# `mAP50-95`.

# %% jupyter={"outputs_hidden": true}
history = yolox_platform.fit_yolox_nano(
	exp,
	PRETRAINED_PATH,
	run_dir,
	DATASET_DIR,
	settings,
	DEVICE,
	WORKSPACE_ROOT,
	resume=run_mode is yolox_platform.RunMode.RESUME,
)

# %% [markdown]
# ## Plot the Training History

# %%
history = yolox_platform.read_training_history(run_dir / "results.csv")

fig, axes = yolox_platform.plot_train_losses(history)
display_img(fig, close=True)

fig, axes = yolox_platform.plot_detection_metrics(history)
display_img(fig, close=True)

# %% [markdown]
# ## Export the Best Checkpoint to ONNX

# %%
best_model_path = run_dir / "weights" / "best_ckpt.pth"
eval_exp = yolox_platform.BasketballNanoExp(
	dataset_dir=DATASET_DIR,
	output_dir=project_space,
	max_epoch=settings.epochs,
	image_size=settings.image_size,
	project_name=project_name,
	class_names=CLASS_NAMES,
	seed=settings.seed,
)

onnx_model_path = yolox_platform.export_trained_model_to_onnx(
	eval_exp,
	best_model_path,
	run_dir / "weights" / "best_ckpt.onnx",
)
print(f"Exported ONNX model: {onnx_model_path}")

# %% [markdown]
# ## Validation and Test Metrics

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

# %%
display(Markdown(yolox_platform.format_final_interpretation(history, test_metrics)))
