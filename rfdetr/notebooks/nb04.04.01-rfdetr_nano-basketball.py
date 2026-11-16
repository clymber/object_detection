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
# # RF-DETR Nano on the Basketball-v0.1 Dataset
#
# Fine-tune COCO-pretrained **RF-DETR Nano**, display training history,
# review selected-checkpoint predictions, and summarize
# validation and held-out test performance.
#
# Defaults match nb04.02: 640 x 640 input and up to 100 epochs on the same
# `basketball-v0.1` dataset. Split sizes, category IDs, and class names are
# read from the selected dataset.
#
# Run with the `object-detection-rfdetr` kernel in a Renku CUDA session.
# [Nano supports custom resolutions](https://rfdetr.roboflow.com/latest/reference/nano/);
# 640 keeps the input size consistent with the Small experiment.

# %%
from __future__ import annotations

import gc

import pandas as pd
import torch
from dataset_builder import Dataset as VersionedDataset
from dataset_builder.rfdetr import summarize_dataset
from detection_common import aligned_print, configure_stdio_relative_path
from detection_common.utils.image import display as display_img
from detection_common.utils.json_io import write_json
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
# Keep `model_size="nano"` to select Nano. Other settings use the defaults or
# `RFDETR_*` variables from the active shell. Overrides take precedence without
# changing the kernel environment. Set `smoke_run` to train for two epochs on
# a small subset. Resume only a Nano run with its saved configuration.

# %%
DATASET_NAME = "basketball-v0.1"
RUN_OVERRIDES = {
    "model_size": "nano",
    # "smoke_run": True,
    # "batch_size": 4,
    # "gradient_checkpointing": True,
    # "early_stopping_patience": 10,
    # "mode": "resume",
    # "run_dir": "outputs/runs/basketball/rfdetr_nano_20260919T190000",
}
settings = rfdetr_platform.settings_from_env(overrides=RUN_OVERRIDES)
dataset = VersionedDataset(DATA_ROOT / "composed" / DATASET_NAME)
manifest, dataset_summary = rfdetr_platform.prepare_dataset(dataset, settings)
aligned_print({**vars(settings), **dataset_summary})

# %% [markdown]
# ## Dataset summary
#
# Validate the COCO annotations and prepare RF-DETR's `train/valid/test` layout,
# preserving class mappings and background images. Full layouts are cached under
# `DATA_ROOT/processed/rfdetr/<DATASET_NAME>/`; smoke runs use a subset.

# %%
display(pd.DataFrame(summarize_dataset(manifest)))
print(f"Dataset fingerprint: {manifest['fingerprint']}")

# %% [markdown]
# ## Initialize and fine-tune RF-DETR Nano
#
# Each fresh run gets a directory under `outputs/runs/basketball/`. Resume an
# interrupted run with `mode="resume"` and its `run_dir`; keep the full run
# folder, including `last.ckpt` and configuration files.
#
# The loader check verifies split membership, class mapping, and empty targets
# before fitting. The detection head uses the dataset's classes. Native loss and
# validation metrics are saved to `metrics.csv`, with one row per epoch in
# `results.csv`. Downloads stay under `models/`.
#
# The progress bar refreshes every five batches and can pause during validation
# and EMA evaluation at the end of an epoch.

# %%
run_dir = rfdetr_platform.prepare_run(settings, manifest, runtime)
print(f"Run directory: {run_dir}")

# %%
training_model = rfdetr_platform.build_model(
    settings, class_names=manifest["category_mapping"]["class_names"]
)
loader_report = rfdetr_platform.verify_loader(training_model, settings, manifest)
write_json(run_dir / "loader_check.json", loader_report)
history = rfdetr_platform.fit_model(training_model, settings, manifest, run_dir)
del training_model
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ## Training history and checkpoint evaluation
#
# Display native training/validation losses and regular/EMA validation AP.
# Reload the checkpoint selected by validation AP50:95, and
# evaluate validation and test images to create metrics and prediction grids.
#
# | File in the run directory | Purpose |
# | --- | --- |
# | `last.ckpt` | Full state for recovery after the last completed epoch |
# | `checkpoint_best_total.pth` | Selected inference weights (regular or EMA) |
# | `checkpoint_best_regular.pth`, `checkpoint_best_ema.pth` | Best regular/EMA weights and metadata |
# | `run_config.json`, `resolved_config.json` | Requested and resolved settings |
# | `dataset_manifest.json`, `loader_check.json` | Dataset and loader checks |
#
# The helper recovers the selected epoch and trained resolution from the source
# checkpoint. Outputs are saved in the run directory.

# %%
postprocess = rfdetr_platform.evaluate_checkpoint_run(
    run_dir, device=settings.device, publish=False
)
history = postprocess["history"]
best_metadata = postprocess["best_metadata"]
split_metrics = postprocess["split_metrics"]
display_img(run_dir / "training_losses.png", width=1000)
display_img(run_dir / "validation_metrics.png", width=1000)
aligned_print(best_metadata)

# %% [markdown]
# ## Qualitative prediction review
#
# Sample IDs are selected from ground truth, independent of the predictions:
# small annotated objects and negative images. Each split saves its IDs,
# labels grid, and predictions grid. Red boxes use confidence >= 0.25.

# %%
for split in ("val", "test"):
    print(f"{split}: ground truth")
    display_img(run_dir / f"{split}_batch0_labels.jpg", width=1000)
    print(f"{split}: predictions")
    display_img(run_dir / f"{split}_batch0_pred.jpg", width=1000)

# %% [markdown]
# ## Performance summary
#
# Evaluate every image, including backgrounds and images with no predictions.
# AP uses score >= 0.001 and COCO `maxDets=[1,10,100]`. Precision, recall, and
# F1 use score >= 0.25 and IoU >= 0.50. Background errors use the same score
# threshold. Evaluation uses FP32.
# Metrics are saved to `val_metrics.json` and `test_metrics.json`; native results
# are saved to `native_*_metrics.json` and detections to `*_predictions.json`.
#
# Review AP alongside precision/recall, background errors, and the grids above.
# Use the training curves to assess convergence or overfitting. Checkpoint
# selection uses validation performance only.

# %%
metric_names = [
    "ap50",
    "ap50_95",
    "ar100",
    "precision",
    "recall",
    "f1",
    "false_positives_per_negative_image",
    "negative_image_false_positive_fraction",
]
display(
    pd.DataFrame(
        {
            split: {name: metrics[name] for name in metric_names}
            for split, metrics in split_metrics.items()
        }
    ).T
)

# %%
test = split_metrics["test"]
display(Markdown(
    f"RF-DETR Nano selected **epoch {best_metadata['best_epoch']}** "
    f"({best_metadata['selected_weights']} weights) using validation AP50:95. "
    f"Held-out test AP50:95 is **{test['ap50_95']:.4f}**, precision is "
    f"**{test['precision']:.4f}**, and recall is **{test['recall']:.4f}**. "
    f"Artifacts are saved in `{run_dir}`."
))

# %% [markdown]
# Export and runtime validation:
# [open the export notebook](nb04.04.02-rfdetr_nano-basketball-export.py).
