# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     notebook_metadata_filter: jupytext,title,authors,-kernelspec,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Basketball ONNX Video Inference
#
# - Run **YOLOX-Tiny, YOLOX-Nano, YOLO11n, and RF-DETR Small** one at a time.
# - Process the full dribbling clip with the same confidence threshold and
#   provider mode for every model.
# - Use the evaluation kernel and framework-neutral ONNX adapters.
# - No tracking or ground truth: coverage is descriptive, and annotated previews
#   are for qualitative review only.

# %%
from __future__ import annotations

import json
import os
import platform
import tempfile
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import onnxruntime as ort
import pandas as pd
from detection_common.runtime import configure_stdio_relative_path
from detection_common.utils.digest import file_digest
from detection_common.utils.json_io import print_json, read_json, write_json
from IPython.display import Markdown, display
from PIL import Image

from detection_evaluation.config import (
    DATA_ROOT,
    OUTPUT_ROOT,
    SUBPROJECT_ROOT,
    WORKSPACE_ROOT,
)
from detection_evaluation.onnx_inference import ModelFamilies, provider_modes
from detection_evaluation.video_inference import (
    comparison_rows,
    describe_model,
    discover_video_models,
    portable_path,
    run_video_inference,
)

configure_stdio_relative_path(WORKSPACE_ROOT)

# %% [markdown]
# ## 1. Configuration
#
# The default `cpu_reference` explicitly requests `CPUExecutionProvider`, allowing
# comparison with the earlier CPU video baseline. Set `PROVIDER_MODE` to
# `coreml_first` to request CoreML with CPU fallback for unsupported graph nodes..
#
# All frames are processed at the source dimensions and nominal playback rate.
# The annotated MP4 contains no audio. Original training dataset files are not
# required: category IDs come from each run's saved canonical mapping.

# %%
VIDEO_PATH = (
    DATA_ROOT
    / "sources"
    / "downloads"
    / "basketball_videos"
    / "basketball_dribbling.webm"
)
RUNS_DIR = OUTPUT_ROOT / "runs" / "basketball"
ARTIFACT_ROOT = OUTPUT_ROOT / "evaluation" / "video_inference"
COREML_CACHE_ROOT = OUTPUT_ROOT / "cache" / "onnxruntime" / "coreml"

CONFIDENCE_THRESHOLD = 0.25
PROVIDER_MODE = "cpu_reference"  # Alternative: "coreml_first".

MODEL_DECLARATIONS = (
    ("yolox_tiny", ModelFamilies.YOLOX, "yolox_tiny*/**/*.onnx"),
    ("yolox_nano", ModelFamilies.YOLOX, "yolox_nano*/**/*.onnx"),
    ("yolo11n", ModelFamilies.YOLO11, "yolo11n*/**/*.onnx"),
    ("rfdetr_small", ModelFamilies.RFDETR, "rfdetr_small*/**/*.onnx"),
)

# %% [markdown]
# ## 2. Discover eligible exports
#
# Select the newest eligible ONNX by file modification time, breaking ties by path. All
# selected runs must share a canonical source fingerprint.
#
# Saved NMS and maximum-detection settings are reused where available. The
# defaults are YOLOX IoU 0.65, YOLO11n IoU 0.70, and maximum 300 detections.
# RF-DETR uses no NMS. Each selected graph will pass `onnx.checker` and the family
# contract check before processing; an incompatible graph fails visibly.

# %%
models, excluded = discover_video_models(
    RUNS_DIR, MODEL_DECLARATIONS, confidence_threshold=CONFIDENCE_THRESHOLD
)
if excluded:
    display(pd.DataFrame(excluded))
else:
    print("No ineligible ONNX candidates were found.")

model_metadata = [describe_model(model) for model in models]
discovery = pd.DataFrame(model_metadata)
discovery["modified_utc"] = pd.to_datetime(discovery["onnx_mtime_ns"], utc=True)
discovery["size_mb"] = discovery["onnx_size_bytes"] / 1024**2
with pd.option_context("display.max_colwidth", None):
    display(
        discovery[
            [
                "model",
                "run_dir",
                "onnx_path",
                "modified_utc",
                "size_mb",
                "confidence_threshold",
                "nms_iou",
                "max_detections",
            ]
        ]
    )

# %% [markdown]
# ## 3. Check runtime and input
#
# Session loading and graph validation happen once per model. Configured and
# actual session providers are recorded in each summary. CoreML compilation
# caches remain under the shared output root.

# %%
if not VIDEO_PATH.is_file():
    raise FileNotFoundError(VIDEO_PATH)
runtime = {
    "onnxruntime": ort.__version__,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "available_providers": ort.get_available_providers(),
}
modes = {}
for model in models:
    choices = {
        mode.name: mode
        for mode in provider_modes(
            runtime["available_providers"], COREML_CACHE_ROOT, model.spec.name
        )
    }
    if PROVIDER_MODE not in choices:
        raise ValueError(f"Unknown provider mode: {PROVIDER_MODE}")
    mode = choices[PROVIDER_MODE]
    if not mode.available:
        raise RuntimeError(mode.unavailable_reason)
    modes[model.spec.name] = mode

video_metadata = {
    "video_path": portable_path(VIDEO_PATH),
    "video_sha256": file_digest(VIDEO_PATH),
}
print_json({**runtime, **video_metadata, "provider_mode": PROVIDER_MODE})

# %% [markdown]
# ## 4. Process the complete clip
#
# Each execution creates a fresh directory and preserves previous outputs.
# Empty-detection frames remain in both the video and JSONL log. 
#
# **Timing boundaries:**
#
# - Processing seconds include capture/writer setup, decoding, RGB conversion,
#   preprocessing, inference, postprocessing, drawing, JSONL/video writes, and
#   output closure. Discovery, hashing, session loading, graph validation,
#   metadata reporting, and previews are excluded.
# - Mean inference milliseconds measure RGB conversion, preprocessing,
#   `session.run`, and postprocessing once per processed frame. Decoding,
#   drawing, logging, and video writing are excluded.
#
# This is one full-video pass, including its first inference. There are no
# warm-up calls or repeated benchmark samples, so these timings differ from a
# warmed single-image benchmark.

# %%
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
execution_dir = Path(
    tempfile.mkdtemp(
        prefix=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ_"), dir=ARTIFACT_ROOT
    )
)
write_json(
    execution_dir / "selection.json",
    {
        "schema_version": 1,
        "execution_id": execution_dir.name,
        **video_metadata,
        "runtime": runtime,
        "provider_mode": PROVIDER_MODE,
        "models": model_metadata,
        "excluded_candidates": excluded,
    },
    allow_nan=False,
)
print(f"Artifacts: {portable_path(execution_dir)}")

summaries = []
for model in models:
    name = model.spec.name
    print(f"Processing full clip: {name}", flush=True)
    try:
        summary = run_video_inference(
            model, VIDEO_PATH, execution_dir / name, modes[name]
        )
    except Exception as error:  # noqa: BLE001 - report failures and review partials.
        print(f"ERROR: {name}: {error}", flush=True)
        summary = read_json(execution_dir / name / "summary.json")
    summaries.append(summary)
    print_json(summary, title=f"{name}: {summary['status']}")

# %% [markdown]
# ## 5. Inspect artifacts and aligned previews
#
# `frame_index` is zero-based. `timestamp_s = frame_index / source_fps` is a
# **nominal timestamp**, not an exact WebM presentation timestamp. Each detection
# contains a source-pixel `[x, y, width, height]` box, center coordinates, saved
# category ID, class name, and confidence. Missed detections remain `[]`.
#
# Follow the artifact links for full video review. The first, middle, and last
# annotated frames are aligned across models and reduced to thumbnails before
# display. They support qualitative review, not precision, recall, or AP.

# %%
for summary in summaries:
    name = summary["model"]
    output_dir = execution_dir / name
    links = []
    for filename in ("annotated.mp4", "detections.jsonl", "summary.json"):
        path = output_dir / filename
        if path.is_file():
            href = os.path.relpath(path, SUBPROJECT_ROOT / "notebooks")
            links.append(f"[{filename}]({href})")
    display(Markdown(f"**{name}**: " + " · ".join(links)))
    if (output_dir / "detections.jsonl").is_file():
        with (output_dir / "detections.jsonl").open(encoding="utf-8") as stream:
            for line in islice(stream, 3):
                print_json(json.loads(line))
    if summary["warnings"]:
        print_json(summary["warnings"], title=f"{name} frame count warnings:")

# %%
completed = [summary for summary in summaries if summary["status"] == "completed"]
rows = comparison_rows(summaries)  # Also checks matching decoded frame counts.
if completed:
    frame_count = completed[0]["processed_frames"]
    preview_indices = (0, frame_count // 2, frame_count - 1)
    figure, axes = plt.subplots(
        len(completed), 3, figsize=(12, 2.8 * len(completed)), squeeze=False
    )
    for row, summary in enumerate(completed):
        capture = cv2.VideoCapture(
            str(execution_dir / summary["model"] / "annotated.mp4")
        )
        try:
            for column, index in enumerate(preview_indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                readable, frame = capture.read()
                if not readable:
                    raise RuntimeError(
                        f"Cannot preview {summary['model']} frame {index}"
                    )
                preview = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                preview.thumbnail((384, 216))
                axes[row, column].imshow(preview)
                axes[row, column].set_title(f"{summary['model']} · frame {index}")
                axes[row, column].axis("off")
        finally:
            capture.release()
    figure.tight_layout()
    display(figure)
    plt.close(figure)

# %% [markdown]
# ## 6. Compare full-video processing
#
# Only completed runs enter the table. Detection counts and frames with/without
# detections describe coverage; there are no frame annotations to measure
# accuracy. Processing FPS includes video I/O, while mean inference latency uses
# the narrower boundary defined above. Reported source frame counts are advisory;
# discrepancies are flagged in each summary.

# %%
comparison = pd.DataFrame(rows)
comparison.to_csv(execution_dir / "comparison.csv", index=False)
display(comparison)
print(f"Comparison: {portable_path(execution_dir / 'comparison.csv')}")
failed = [summary["model"] for summary in summaries if summary["status"] != "completed"]
if failed:
    raise RuntimeError("Video inference failed for: " + ", ".join(failed))
