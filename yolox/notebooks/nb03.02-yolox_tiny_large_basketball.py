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
# # YOLOX-Tiny on basketball-v0.1
#
# Fine-tune the existing COCO-pretrained model at 640 x 640 for up to 100 epochs.
# Reviewed basketball source splits and background images are preserved.
# The producer validates the direct COCO loader, captures run provenance, and
# publishes validation/test bundles. It does not require a YOLO-format export.

# %%
from __future__ import annotations

from dataset_builder import Dataset as VersionedDataset
from detection_common import aligned_print, configure_stdio_relative_path
from detection_common.utils.image import display as display_img
from IPython.display import Markdown, display

from yolox_pipeline import producer
from yolox_pipeline import yolox as yolox_platform
from yolox_pipeline.config import DATA_ROOT, WORKSPACE_ROOT

configure_stdio_relative_path(WORKSPACE_ROOT)

# %% [markdown]
# ## Experiment settings
#
# Set `YOLOX_TINY_SMOKE=1` for two complete epochs on the shared 16/8/8
# subset. `YOLOX_TINY_PROGRESS` and `YOLOX_TINY_VERBOSE` control output.
# Set `RESUME_RUN_DIR` to an interrupted run; relative paths use `OUTPUT_ROOT`.

# %%
DATASET_NAME = "basketball-v0.1"
CLASS_NAMES = ("basketball",)
RESUME_RUN_DIR: str | None = None
dataset = VersionedDataset(DATA_ROOT / "composed" / DATASET_NAME)
experiment = producer.prepare_experiment(
    dataset,
    model_name="tiny",
    class_names=CLASS_NAMES,
    logical_dataset="basketball",
    source_notebook="nb03.02-yolox_tiny_large_basketball.ipynb",
    env_prefix="YOLOX_TINY",
    resume_run_dir=RESUME_RUN_DIR,
)
aligned_print(experiment.summary)

# %% [markdown]
# ## Dataset and model
#
# The displayed counts describe the full versioned source. Smoke mode uses the
# deterministic subset. The pretrained classification head is replaced with
# the single class above; backbone and detection features are retained.

# %%
display(experiment.summary["splits"])
run_dir = experiment.run_dir
dataset_paths = experiment.paths
producer_settings = experiment.settings
settings = producer_settings.training
run_mode = settings.run_mode
model_variant = experiment.model
DEVICE = experiment.device
PRETRAINED_PATH = experiment.pretrained_path
exp = producer.evaluation_experiment(experiment)

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
    original_utc=experiment.original_utc,
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
eval_exp = producer.evaluation_experiment(experiment)

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


# %% [markdown]
# ## Qualitative Prediction Review
#
# Save and display ground-truth and prediction grids for validation and test
# images, matching the style of the sample artifacts in `nb02` and `nb03.02`.

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
