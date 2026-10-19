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
# # Basketball ONNX Model Inference
#
# This notebook answers two questions for YOLOX-Tiny, YOLOX-Nano, YOLO11n, and
# RF-DETR Small:
#
# 1. Do the exported ONNX models produce sensible detections on the same
#    held-out basketball image?
# 2. How does median batch-one latency compare between CPU-only and
#    CoreML-first execution on Apple Silicon?
#
# It discovers the latest complete exports, validates their provenance and ONNX
# graphs, runs one test image through every model, and shows CPU predictions
# beside a compact provider comparison.
#
# CoreML-first means ONNX Runtime offers supported graph partitions to CoreML
# and sends unsupported work to the CPU provider. It does not imply that the
# complete graph ran on the GPU or Neural Engine. Azure is deliberately omitted
# because it is a remote endpoint provider rather than local M1 acceleration.

# %%
"""
Compare framework-neutral ONNX inference on the basketball test split.
"""

# %load_ext autoreload
# %autoreload 2
# %aimport -gc, -platform, -sys, -pathlib, -typing, -matplotlib, -onnx, -onnxruntime
# %aimport -pandas, -yaml

# %%
from __future__ import annotations

import gc
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import yaml
from detection_common import aligned_print
from detection_common.runtime import configure_stdio_relative_path
from detection_common.utils.image import BBoxFormat, letterbox
from detection_common.utils.json_io import read_json, write_json
from IPython.display import Markdown, display
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from PIL import Image

from detection_evaluation import find_latest_path, read_run_protocol
from detection_evaluation.config import DATA_ROOT, OUTPUT_ROOT, WORKSPACE_ROOT
from detection_evaluation.onnx_inference import (
    Detection,
    ModelFamilies,
    OnnxModelSpec,
    ProviderMode,
    benchmark_session,
    create_session,
    match_detections,
    provider_modes,
    select_next_test_image,
)

configure_stdio_relative_path(WORKSPACE_ROOT)

# %% [markdown]
# ## Settings
#
# File I/O, session creation/CoreML compilation, and drawing are outside the
# latency timer. Every model/provider case uses the same loaded RGB image,
# warm-up count, and timed sample count.

# %%
RUNS_DIR = OUTPUT_ROOT / "runs" / "basketball"
DATASET_DIR = DATA_ROOT / "composed" / "coco_basketball"
ANNOTATION_PATH = DATASET_DIR / "annotations" / "instances_test.json"
IMAGE_DIR = DATASET_DIR / "images" / "test"
SELECTION_STATE_PATH = (
    OUTPUT_ROOT / "notebook_state" / "nb05.03-onnx_model_inference.json"
)
COREML_CACHE_ROOT = OUTPUT_ROOT / "cache" / "onnxruntime" / "coreml"

CONFIDENCE_THRESHOLD = 0.25
MATCH_IOU_THRESHOLD = 0.50
LETTERBOX_SIZE = (640, 640)
WARMUP_RUNS = 5
TIMED_RUNS = 30

MODEL_DISPLAY_NAMES = {
    "yolox_tiny": "YOLOX-Tiny",
    "yolox_nano": "YOLOX-Nano",
    "yolo11n": "YOLO11n",
    "rfdetr_small": "RF-DETR Small",
}

if not ANNOTATION_PATH.is_file():
    raise FileNotFoundError(f"Annotation file is missing: {ANNOTATION_PATH}")
if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(f"The image directory is missing: {IMAGE_DIR}")

# %% [markdown]
# ## Discover and validate full-run ONNX exports
#
# Select the newest export for each model, then validate its immutable run
# protocol. All four runs must use the full logical basketball dataset and the
# same canonical source fingerprint.


# %%
def find_run_directory(model_path: Path) -> Path:
    """
    Find the nearest protocol-bearing run directory above an ONNX export.
    """
    for candidate in (model_path.parent, *model_path.parents):
        if (candidate / "run_protocol.json").is_file():
            return candidate
    raise FileNotFoundError(f"No run_protocol.json found above {model_path}")


def saved_postprocessing(run_dir: Path) -> dict[str, Any]:
    """
    Read model-owned test postprocessing metadata when it is available.
    """
    candidates = (
        run_dir / "evaluation" / "test_predictions.json",
        run_dir / "test_predictions.json",
    )
    for path in candidates:
        if path.is_file():
            artifact = read_json(path)
            metadata = artifact.get("metadata", {})
            postprocessing = metadata.get("postprocessing", {})
            if isinstance(postprocessing, dict):
                return postprocessing
    return {}


def yolox_nms_threshold(run_dir: Path, fallback: float = 0.65) -> float:
    """
    Recover YOLOX NMS IoU from the frozen run settings when possible.
    """
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        return fallback
    settings = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        return fallback
    return float(settings.get("nms_threshold", fallback))


model_declarations = (
    ("yolox_tiny", ModelFamilies.YOLOX, "yolox_tiny*/**/*.onnx", 0.65),
    ("yolox_nano", ModelFamilies.YOLOX, "yolox_nano*/**/*.onnx", 0.65),
    ("yolo11n",    ModelFamilies.YOLO11, "yolo11n*/**/*.onnx", 0.70),
    ("rfdetr_small", ModelFamilies.RFDETR, "rfdetr_small*/**/*.onnx", None),
)

resolved_paths = {
    name: find_latest_path(RUNS_DIR, pattern)
    for name, _, pattern, _ in model_declarations
}
missing_models = [name for name, path in resolved_paths.items() if path is None]
if missing_models:
    raise FileNotFoundError(
        "Missing basketball ONNX exports for: " + ", ".join(missing_models)
    )

model_specs: list[OnnxModelSpec] = []
model_run_dirs: dict[str, Path] = {}
model_protocols: dict[str, dict[str, Any]] = {}
for name, family, _, default_nms in model_declarations:
    model_path = resolved_paths[name]
    assert model_path is not None

    run_dir = find_run_directory(model_path)
    protocol = read_run_protocol(run_dir)
    if protocol["model"] != name or protocol["logical_dataset"] != "basketball":
        raise ValueError(f"Unexpected run protocol for {name}: {run_dir}")

    settings = protocol["training_settings"]
    if (
        protocol["smoke_run"]
        or settings.get("train_batch_limit") is not None
        or float(settings.get("fraction", 1.0)) != 1.0
    ):
        raise ValueError(f"ONNX comparison requires a complete full run: {run_dir}")

    postprocessing = saved_postprocessing(run_dir)
    if family is ModelFamilies.YOLOX:
        nms_threshold = float(
            postprocessing.get("nms_iou", yolox_nms_threshold(run_dir))
        )
    elif family is ModelFamilies.YOLO11:
        nms_threshold = float(postprocessing.get("nms_iou", default_nms))
    else:
        nms_threshold = None

    model_specs.append(
        OnnxModelSpec(
            name=name,
            family=family,
            path=model_path,
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=nms_threshold,
            max_detections=int(postprocessing.get("max_det", 300)),
        )
    )
    model_run_dirs[name] = run_dir
    model_protocols[name] = protocol

source_fingerprints = {
    name: protocol["dataset_identity"]["canonical"]["source_fingerprint"]
    for name, protocol in model_protocols.items()
}
if len(set(source_fingerprints.values())) != 1:
    raise ValueError(
        f"ONNX runs use different canonical datasets: {source_fingerprints}"
    )

for spec in model_specs:
    onnx.checker.check_model(str(spec.path))

discovery_rows = [
    {
        "model": spec.name,
        "run_dir": model_run_dirs[spec.name].relative_to(WORKSPACE_ROOT),
        "onnx_path": spec.path.relative_to(WORKSPACE_ROOT),
        "onnx_size_mb": spec.path.stat().st_size / (1024**2),
        "score_threshold": spec.score_threshold,
        "nms_iou": spec.nms_threshold,
    }
    for spec in model_specs
]
display(pd.DataFrame(discovery_rows))

# %% [markdown]
# ## Select and letterbox a non-repeating random test image
#
# Selection uses a persisted shuffled queue. A test image is not repeated until
# the entire split has been used, including when the kernel is restarted. The
# final image of one cycle cannot become the first image of the next. Pixels and
# COCO boxes are letterboxed together onto the common square inference canvas.

# %%
selected = select_next_test_image(ANNOTATION_PATH, SELECTION_STATE_PATH)
categories = sorted(
    selected.dataset.get("categories", []),
    key=lambda item: item["id"],
)
if [(item.get("id"), item.get("name")) for item in categories] != [(1, "basketball")]:
    raise ValueError(f"Expected the canonical basketball category, got {categories}")
category_ids = [int(item["id"]) for item in categories]
category_names = {int(item["id"]): item["name"] for item in categories}

image_path = IMAGE_DIR / selected.image["file_name"]
if not image_path.is_file():
    raise FileNotFoundError(image_path)
with Image.open(image_path) as source:
    source_image = source.convert("RGB").copy()

source_boxes = np.asarray(
    [annotation["bbox"] for annotation in selected.annotations],
    dtype=np.float32,
).reshape(-1, 4)
letterboxed_image, letterboxed_boxes, _, _ = letterbox(
    np.asarray(source_image),
    source_boxes,
    size=LETTERBOX_SIZE,
    bbox_format=BBoxFormat.XYWH,
)
assert letterboxed_boxes is not None
selected_image = Image.fromarray(letterboxed_image)
selected_annotations = tuple(
    {**annotation, "bbox": box.tolist()}
    for annotation, box in zip(
        selected.annotations,
        letterboxed_boxes,
        strict=True,
    )
)

aligned_print({
    "image_id": selected.image["id"],
    "file_name": selected.image["file_name"],
    "source_image_size": source_image.size,
    "letterboxed_image_size": selected_image.size,
    "ground_truth_boxes": len(selected_annotations),
    "selection_state": SELECTION_STATE_PATH,
})

# %% [markdown]
# ## Runtime and provider configuration
#
# CPU-only is the correctness reference. CoreML-first uses MLProgram with all
# Apple compute units available and CPU fallback for unsupported graph nodes.
# Session creation and CoreML compilation are excluded from inference latency.

# %%
available_providers = ort.get_available_providers()
runtime_details = {
    "onnx": onnx.__version__,
    "onnxruntime": ort.__version__,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor() or "unknown",
    "available_providers": available_providers,
}
print("Runtime info:")
_ = write_json(sys.stdout, runtime_details)
if platform.system() != "Darwin" or platform.machine() != "arm64":
    raise RuntimeError("This notebook comparison targets an Apple Silicon Mac")

# %% [markdown]
# ## Run the provider comparison
#
# Cases run sequentially so only one ONNX Runtime session occupies memory at a
# time. A CPU failure stops the notebook because CPU predictions drive the
# visual check. A CoreML failure remains visible in the comparison table.

# %%
def run_benchmark_case(
    spec: OnnxModelSpec,
    mode: ProviderMode,
) -> tuple[dict[str, Any], tuple[Detection, ...] | None]:
    """
    Benchmark one model/provider case for the selected image.
    """
    row: dict[str, Any] = {
        "model": spec.name,
        "provider": mode.name,
        "error": mode.unavailable_reason,
    }
    if not mode.available:
        return row, None

    session = None
    try:
        session = create_session(spec.path, mode)
        result = benchmark_session(
            session,
            spec,
            selected_image,
            category_ids,
            warmup=WARMUP_RUNS,
            samples=TIMED_RUNS,
        )
    except Exception as error:
        if mode.name == "cpu_reference":
            raise RuntimeError(
                f"CPU reference inference failed for {spec.name}"
            ) from error
        row["error"] = f"{type(error).__name__}: {error}"
        return row, None
    finally:
        del session
        gc.collect()

    row.update(
        {
            "predictions": len(result.predictions),
            "latency_ms": result.latency_ms,
            "images_per_second": 1000.0 / result.latency_ms,
            **match_detections(
                selected_annotations,
                result.predictions,
                iou_threshold=MATCH_IOU_THRESHOLD,
            ),
        }
    )
    return row, result.predictions


comparison_rows: list[dict[str, Any]] = []
cpu_predictions: dict[str, tuple[Detection, ...]] = {}

for spec in model_specs:
    for mode in provider_modes(available_providers, COREML_CACHE_ROOT, spec.name):
        comparison_row, predictions = run_benchmark_case(spec, mode)
        comparison_rows.append(comparison_row)
        if mode.name == "cpu_reference":
            assert predictions is not None
            cpu_predictions[spec.name] = predictions
        outcome = "complete" if comparison_row["error"] is None else "unavailable"
        print(f"{spec.name} / {mode.name}: {outcome}", flush=True)

# %% [markdown]
# ## Inspect CPU-reference predictions
#
# The five panels use the same original image and confidence threshold. CoreML
# results are omitted rather than duplicating visually equivalent panels.


# %%
def draw_panel(
    axis: Axes,
    image: Image.Image,
    boxes: list[tuple[int, tuple[float, float, float, float], float | None]],
    *,
    title: str,
    color: str,
) -> None:
    """
    Draw labeled COCO xywh boxes over one source-image panel.
    """
    axis.imshow(image)
    for category_id, (x, y, width, height), score in boxes:
        axis.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=2,
            )
        )
        label = category_names[category_id]
        if score is not None:
            label = f"{label} {score:.2f}"
        axis.text(
            x,
            max(0, y - 3),
            label,
            color="white",
            fontsize=9,
            bbox={"facecolor": color, "alpha": 0.8, "pad": 2},
        )
    axis.set_title(f"{title}\n{len(boxes)} box(es)")
    axis.axis("off")


figure, axes = plt.subplots(1, 5, figsize=(25, 6), squeeze=False)
ground_truth_boxes = [
    (
        int(annotation["category_id"]),
        tuple(float(value) for value in annotation["bbox"]),
        None,
    )
    for annotation in selected_annotations
]
draw_panel(
    axes[0, 0],
    selected_image,
    ground_truth_boxes,
    title="Ground truth",
    color="limegreen",
)
for column, spec in enumerate(model_specs, start=1):
    prediction_boxes = [
        (detection.category_id, detection.bbox, detection.score)
        for detection in cpu_predictions[spec.name]
    ]
    draw_panel(
        axes[0, column],
        selected_image,
        prediction_boxes,
        title=MODEL_DISPLAY_NAMES[spec.name],
        color="deepskyblue",
    )
figure.suptitle(
    f"Basketball test image {selected.image['id']}: {selected.image['file_name']}",
    fontsize=14,
)
figure.tight_layout()
display(figure)
plt.close(figure)

# %% [markdown]
# ## Compare providers
#
# Latency is median batch-one FP32 inference on this M1 Pro, including in-memory
# preprocessing and postprocessing. It excludes file I/O, model loading, CoreML
# compilation, and drawing. CoreML-first rows may include CPU fallback.
#
# Precision, recall, and F1 below describe only this randomly selected image at
# confidence 0.25 and IoU 0.50. Use the full-split comparison notebook for
# dataset-level AP and AR.

# %%
comparison = pd.DataFrame(
    comparison_rows,
    columns=[
        "model",
        "provider",
        "predictions",
        "precision",
        "recall",
        "f1",
        "latency_ms",
        "images_per_second",
        "error",
    ],
).round(3)
display(
    comparison.rename(
        columns={
            "f1": "F1",
            "latency_ms": "latency (ms)",
            "images_per_second": "images/s",
        }
    )
)

if comparison["error"].notna().any():
    display(
        Markdown(
            "**CoreML note:** one or more CoreML-first cases were unavailable. "
            "The CPU-reference comparison is complete; inspect the `error` column "
            "before treating CoreML performance as complete."
        )
    )
