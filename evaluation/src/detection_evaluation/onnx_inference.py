"""
Framework-neutral ONNX object-detection inference and comparison helpers.
"""

from __future__ import annotations

import enum
import os
import secrets
import time
import uuid
from collections.abc import Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from detection_common.utils.digest import file_digest
from detection_common.utils.json_io import json_bytes, read_json
from PIL import Image


class ModelFamilies(enum.StrEnum):
    """
    Identify the supported ONNX model families.
    """

    YOLOX = "yolox"
    YOLO11 = "yolo11"
    RFDETR = "rf-detr"

    @classmethod
    def is_yolo(cls, family: ModelFamilies | str):
        return cls(family) in [cls.YOLOX, cls.YOLO11]


class _RandomSource(Protocol):
    """
    Describe the random operation needed by the persisted selection queue.
    """

    def shuffle(self, values: MutableSequence[int]) -> None:
        """
        Randomize a mutable sequence in place.
        """


@dataclass(frozen=True)
class OnnxModelSpec:
    """
    Describe one exported detector and its deployment-time policies.
    """

    name: str
    family: ModelFamilies
    path: Path
    score_threshold: float = 0.25
    nms_threshold: float | None = None
    max_detections: int = 300

    def __post_init__(self) -> None:
        """
        Reject incomplete or unsupported model declarations.
        """
        if not self.name or not isinstance(self.family, ModelFamilies):
            raise ValueError("ONNX model requires a name and supported family")
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be between zero and one")
        if self.nms_threshold is not None and not 0 < self.nms_threshold <= 1:
            raise ValueError("nms_threshold must be in (0, 1]")
        if ModelFamilies.is_yolo(self.family) and self.nms_threshold is None:
            raise ValueError("YOLO models require an NMS threshold")
        if self.max_detections < 1:
            raise ValueError("max_detections must be positive")


@dataclass(frozen=True)
class Detection:
    """
    Hold one source-image detection in COCO xywh coordinates.
    """

    category_id: int
    bbox: tuple[float, float, float, float]
    score: float

    def as_coco(self) -> dict[str, Any]:
        """
        Return a JSON-friendly COCO prediction without an image ID.
        """
        return {
            "category_id": self.category_id,
            "bbox": list(self.bbox),
            "score": self.score,
        }


@dataclass(frozen=True)
class ImageTransform:
    """
    Retain preprocessing geometry needed to restore source-image boxes.
    """

    tensor: np.ndarray
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale: float | None
    pad_left: int = 0
    pad_top: int = 0


@dataclass(frozen=True)
class SessionContract:
    """
    Capture one validated detector graph's input and output contract.
    """

    input_name: str
    input_height: int
    input_width: int
    output_names: tuple[str, ...]


@dataclass(frozen=True)
class ProviderMode:
    """
    Define one explicit ONNX Runtime provider configuration.
    """

    name: str
    providers: tuple[Any, ...]
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        """
        Return whether the mode can be used on the current runtime.
        """
        return self.unavailable_reason is None


@dataclass(frozen=True)
class BenchmarkResult:
    """
    Hold predictions and median end-to-end latency for one benchmark.
    """

    predictions: tuple[Detection, ...]
    latency_ms: float


@dataclass(frozen=True)
class SelectedImage:
    """
    Hold one randomly selected COCO image and its ground-truth annotations.
    """

    dataset: dict[str, Any]
    image: dict[str, Any]
    annotations: tuple[dict[str, Any], ...]


def _write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """
    Atomically replace one small generated JSON state document.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(json_bytes(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _selection_state(
    state_path: Path,
    *,
    annotation_sha256: str,
    image_ids: set[int],
) -> tuple[list[int], int | None]:
    """
    Read a compatible shuffled queue or return fresh-state defaults.
    """
    if not state_path.is_file():
        return [], None
    try:
        state = read_json(state_path)
    except (OSError, ValueError, TypeError):
        return [], None
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        return [], None
    remaining = state.get("remaining_image_ids")
    last_image_id = state.get("last_image_id")
    if (
        state.get("annotation_sha256") != annotation_sha256
        or not isinstance(remaining, list)
        or any(type(value) is not int for value in remaining)
        or len(remaining) != len(set(remaining))
        or not set(remaining).issubset(image_ids)
        or (last_image_id is not None and last_image_id not in image_ids)
        or last_image_id in remaining
    ):
        return [], None
    return list(remaining), last_image_id


def select_next_test_image(
    annotation_path: Path | str,
    state_path: Path | str,
    *,
    random_source: _RandomSource | None = None,
) -> SelectedImage:
    """
    Select without replacement across runs and atomically persist queue state.
    """
    annotation_path = Path(annotation_path)
    state_path = Path(state_path)
    dataset = read_json(annotation_path)
    if not isinstance(dataset, dict):
        raise TypeError("COCO annotations must be a JSON object")
    images = dataset.get("images")
    annotations = dataset.get("annotations")
    if not isinstance(images, list) or len(images) < 2:
        raise ValueError("Random non-repeating selection requires at least two images")
    if not isinstance(annotations, list):
        raise TypeError("COCO annotations must contain an annotations list")
    image_ids = [entry.get("id") for entry in images if isinstance(entry, dict)]
    if (
        len(image_ids) != len(images)
        or any(type(image_id) is not int for image_id in image_ids)
        or len(set(image_ids)) != len(image_ids)
    ):
        raise ValueError("COCO images require unique integer IDs")

    digest = file_digest(annotation_path)
    remaining, last_image_id = _selection_state(
        state_path,
        annotation_sha256=digest,
        image_ids=set(image_ids),
    )
    if not remaining:
        remaining = list(image_ids)
        (random_source or secrets.SystemRandom()).shuffle(remaining)
        if remaining[0] == last_image_id:
            replacement = next(
                index
                for index, image_id in enumerate(remaining)
                if image_id != last_image_id
            )
            remaining[0], remaining[replacement] = remaining[replacement], remaining[0]

    selected_id = remaining.pop(0)
    image = next(entry for entry in images if entry["id"] == selected_id)
    selected_annotations = tuple(
        entry
        for entry in annotations
        if isinstance(entry, dict) and entry.get("image_id") == selected_id
    )
    _write_json_atomic(
        state_path,
        {
            "schema_version": 1,
            "annotation_sha256": digest,
            "last_image_id": selected_id,
            "remaining_image_ids": remaining,
        },
    )
    return SelectedImage(dataset, image, selected_annotations)


def provider_modes(
    available_providers: Sequence[str],
    cache_root: Path | str,
    model_name: str,
) -> tuple[ProviderMode, ProviderMode]:
    """
    Build CPU-reference and CoreML-first provider declarations for one model.
    """
    available = set(available_providers)
    if "CPUExecutionProvider" not in available:
        raise RuntimeError("ONNX Runtime does not provide CPUExecutionProvider")
    cpu = ProviderMode("cpu_reference", ("CPUExecutionProvider",))
    cache_dir = Path(cache_root) / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    coreml_providers: tuple[Any, ...] = (
        (
            "CoreMLExecutionProvider",
            {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "RequireStaticInputShapes": "1",
                "ModelCacheDirectory": str(cache_dir.resolve()),
            },
        ),
        "CPUExecutionProvider",
    )
    coreml = ProviderMode(
        "coreml_first",
        coreml_providers,
        None
        if "CoreMLExecutionProvider" in available
        else "CoreMLExecutionProvider is unavailable in this ONNX Runtime build",
    )
    return cpu, coreml


def create_session(model_path: Path | str, mode: ProviderMode) -> Any:
    """
    Create one ONNX Runtime session with an explicit provider priority chain.
    """
    if not mode.available:
        raise RuntimeError(mode.unavailable_reason)
    import onnxruntime as ort

    return ort.InferenceSession(str(model_path), providers=list(mode.providers))


def validate_session_contract(session: Any, family: ModelFamilies) -> SessionContract:
    """
    Validate the fixed batch-one detector contract used by this comparison.
    """
    if not isinstance(family, ModelFamilies):
        raise ValueError(f"Unsupported model family: {family}")
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise ValueError("ONNX detector must have exactly one input")
    model_input = inputs[0]
    shape = list(model_input.shape)
    if model_input.type != "tensor(float)" or len(shape) != 4:
        raise ValueError("ONNX input must be a rank-four FP32 tensor")
    if shape[0] != 1 or shape[1] != 3:
        raise ValueError("ONNX input must have static shape [1, 3, H, W]")
    height, width = shape[2:]
    if type(height) is not int or type(width) is not int or height <= 0 or width <= 0:
        raise ValueError("ONNX input height and width must be positive static integers")

    output_shapes = {output.name: list(output.shape) for output in outputs}
    anchor_count = sum((height // stride) * (width // stride) for stride in (8, 16, 32))
    if family is ModelFamilies.YOLOX:
        if output_shapes != {"output": [1, anchor_count, 6]}:
            raise ValueError(f"Unexpected YOLOX outputs: {output_shapes}")
    elif family is ModelFamilies.YOLO11:
        if output_shapes != {"output0": [1, 5, anchor_count]}:
            raise ValueError(f"Unexpected YOLO11 outputs: {output_shapes}")
    elif (
        output_shapes.get("dets") != [1, 300, 4]
        or output_shapes.get("labels") != [1, 300, 2]
        or len(output_shapes) != 2
    ):
        raise ValueError(f"Unexpected RF-DETR outputs: {output_shapes}")
    if any(output.type != "tensor(float)" for output in outputs):
        raise ValueError("ONNX detector outputs must be FP32 tensors")
    return SessionContract(
        input_name=model_input.name,
        input_height=height,
        input_width=width,
        output_names=tuple(output.name for output in outputs),
    )


def preprocess_yolox(
    image: Image.Image, input_height: int, input_width: int
) -> ImageTransform:
    """
    Reproduce YOLOX ValTransform top-left BGR letterboxing.
    """
    rgb = np.asarray(image.convert("RGB"))
    bgr = rgb[:, :, ::-1]
    original_height, original_width = bgr.shape[:2]
    scale = min(input_height / original_height, input_width / original_width)
    resized_width = max(1, int(original_width * scale))
    resized_height = max(1, int(original_height * scale))
    resized = cv2.resize(
        bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
    return ImageTransform(
        tensor,
        original_width,
        original_height,
        input_width,
        input_height,
        scale,
    )


def preprocess_yolo11(
    image: Image.Image, input_height: int, input_width: int
) -> ImageTransform:
    """
    Reproduce centered Ultralytics RGB letterboxing and normalization.
    """
    rgb = np.asarray(image.convert("RGB"))
    original_height, original_width = rgb.shape[:2]
    scale = min(input_height / original_height, input_width / original_width)
    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    horizontal = (input_width - resized_width) / 2
    vertical = (input_height - resized_height) / 2
    left, right = round(horizontal - 0.1), round(horizontal + 0.1)
    top, bottom = round(vertical - 0.1), round(vertical + 0.1)
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if canvas.shape[:2] != (input_height, input_width):
        raise ValueError("Ultralytics letterbox produced an unexpected shape")
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    return ImageTransform(
        tensor,
        original_width,
        original_height,
        input_width,
        input_height,
        scale,
        left,
        top,
    )


def bilinear_resize_half_pixel(
    source: np.ndarray, output_height: int, output_width: int
) -> np.ndarray:
    """
    Resize CHW FP32 data with align-corners-false half-pixel bilinear sampling.
    """
    if source.ndim != 3 or output_height < 1 or output_width < 1:
        raise ValueError("Resize expects CHW input and positive output dimensions")
    source = np.asarray(source, dtype=np.float32)
    source_height, source_width = source.shape[-2:]
    source_y = (np.arange(output_height, dtype=np.float32) + 0.5) * (
        source_height / output_height
    ) - 0.5
    source_x = (np.arange(output_width, dtype=np.float32) + 0.5) * (
        source_width / output_width
    ) - 0.5
    source_y = np.clip(source_y, 0.0, source_height - 1)
    source_x = np.clip(source_x, 0.0, source_width - 1)
    y0 = np.floor(source_y).astype(np.int64)
    x0 = np.floor(source_x).astype(np.int64)
    y1 = np.minimum(y0 + 1, source_height - 1)
    x1 = np.minimum(x0 + 1, source_width - 1)
    dy = (source_y - y0)[:, None]
    dx = (source_x - x0)[None, :]
    top_left = source[..., y0[:, None], x0[None, :]]
    top_right = source[..., y0[:, None], x1[None, :]]
    bottom_left = source[..., y1[:, None], x0[None, :]]
    bottom_right = source[..., y1[:, None], x1[None, :]]
    output = (1 - dy) * ((1 - dx) * top_left + dx * top_right) + dy * (
        (1 - dx) * bottom_left + dx * bottom_right
    )
    return np.asarray(output, dtype=np.float32)


def preprocess_rfdetr(
    image: Image.Image, input_height: int, input_width: int
) -> ImageTransform:
    """
    Reproduce RF-DETR's square resize and ImageNet normalization.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    original_height, original_width = rgb.shape[:2]
    chw = bilinear_resize_half_pixel(rgb.transpose(2, 0, 1), input_height, input_width)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    standard_deviation = np.array([0.229, 0.224, 0.225], dtype=np.float32)[
        :, None, None
    ]
    tensor = np.ascontiguousarray(((chw - mean) / standard_deviation)[None])
    return ImageTransform(
        tensor,
        original_width,
        original_height,
        input_width,
        input_height,
        None,
    )


def prepare_input(
    image: Image.Image, contract: SessionContract, family: ModelFamilies
) -> ImageTransform:
    """
    Dispatch image preprocessing for one validated model family.
    """
    if family is ModelFamilies.YOLOX:
        return preprocess_yolox(image, contract.input_height, contract.input_width)
    if family is ModelFamilies.YOLO11:
        return preprocess_yolo11(image, contract.input_height, contract.input_width)
    if family is ModelFamilies.RFDETR:
        return preprocess_rfdetr(image, contract.input_height, contract.input_width)
    raise ValueError(f"Unsupported model family: {family}")


def xyxy_nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """
    Apply deterministic, class-local non-maximum suppression to xyxy boxes.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,) or scores.shape != (len(boxes),):
        raise ValueError("NMS expects boxes shaped (N, 4) and scores shaped (N,)")
    if not 0 < threshold <= 1:
        raise ValueError("NMS threshold must be in (0, 1]")
    if not len(boxes):
        return []
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    indices = np.arange(len(scores))
    order = np.lexsort((indices, -scores))
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        remaining = order[1:]
        if not remaining.size:
            break
        x1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
        y1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
        x2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
        y2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        union = areas[current] + areas[remaining] - intersection
        overlap = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = remaining[overlap <= threshold]
    return keep


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert a box array from center-size to corner coordinates.
    """
    converted = np.empty_like(boxes, dtype=np.float32)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def _source_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    category_ids: Sequence[int],
    transform: ImageTransform,
    *,
    direct_normalized: bool = False,
) -> tuple[Detection, ...]:
    """
    Restore boxes to source pixels, clip them, and construct detections.
    """
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    if direct_normalized:
        boxes[:, [0, 2]] *= transform.original_width
        boxes[:, [1, 3]] *= transform.original_height
    else:
        if transform.scale is None:
            raise ValueError("Letterboxed detections require a resize scale")
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform.pad_left) / transform.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform.pad_top) / transform.scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, transform.original_width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, transform.original_height)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    order = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
    detections = []
    for index in order:
        label = int(labels[index])
        if widths[index] <= 0 or heights[index] <= 0:
            continue
        if not 0 <= label < len(category_ids):
            raise ValueError(f"Prediction label {label} has no COCO category mapping")
        detections.append(
            Detection(
                category_id=int(category_ids[label]),
                bbox=(
                    float(boxes[index, 0]),
                    float(boxes[index, 1]),
                    float(widths[index]),
                    float(heights[index]),
                ),
                score=float(scores[index]),
            )
        )
    return tuple(detections)


def decode_yolox(
    raw_output: np.ndarray,
    transform: ImageTransform,
    category_ids: Sequence[int],
    *,
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> tuple[Detection, ...]:
    """
    Decode raw YOLOX grid outputs and apply class-aware NMS.
    """
    output = np.asarray(raw_output, dtype=np.float32)
    if (
        output.ndim != 3
        or output.shape[0] != 1
        or output.shape[2] != 5 + len(category_ids)
    ):
        raise ValueError("Unexpected YOLOX output shape")
    decoded = output[0].copy()
    grids = []
    expanded_strides = []
    for stride in (8, 16, 32):
        grid_height = transform.input_height // stride
        grid_width = transform.input_width // stride
        x_grid, y_grid = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
        grid = np.stack((x_grid, y_grid), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((len(grid), 1), stride, dtype=np.float32))
    grid = np.concatenate(grids).astype(np.float32)
    strides = np.concatenate(expanded_strides)
    if len(decoded) != len(grid):
        raise ValueError("YOLOX output count does not match the stride grid")
    decoded[:, :2] = (decoded[:, :2] + grid) * strides
    decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * strides
    boxes = _cxcywh_to_xyxy(decoded[:, :4])
    class_scores = decoded[:, 4:5] * decoded[:, 5:]
    selected_boxes = []
    selected_scores = []
    selected_labels = []
    for label in range(class_scores.shape[1]):
        scores = class_scores[:, label]
        candidates = np.flatnonzero(scores >= score_threshold)
        kept = xyxy_nms(boxes[candidates], scores[candidates], nms_threshold)
        for local_index in kept:
            index = int(candidates[local_index])
            selected_boxes.append(boxes[index])
            selected_scores.append(scores[index])
            selected_labels.append(label)
    if not selected_boxes:
        return ()
    scores = np.asarray(selected_scores, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[:max_detections]
    return _source_detections(
        np.asarray(selected_boxes)[order],
        scores[order],
        np.asarray(selected_labels, dtype=np.int64)[order],
        category_ids,
        transform,
    )


def decode_yolo11(
    raw_output: np.ndarray,
    transform: ImageTransform,
    category_ids: Sequence[int],
    *,
    score_threshold: float,
    nms_threshold: float,
    max_detections: int,
) -> tuple[Detection, ...]:
    """
    Decode exported YOLO11 boxes and apply class-aware NMS.
    """
    output = np.asarray(raw_output, dtype=np.float32)
    if (
        output.ndim != 3
        or output.shape[0] != 1
        or output.shape[1] != 4 + len(category_ids)
    ):
        raise ValueError("Unexpected YOLO11 output shape")
    candidates = output[0].T
    boxes = _cxcywh_to_xyxy(candidates[:, :4])
    selected_boxes = []
    selected_scores = []
    selected_labels = []
    for label in range(len(category_ids)):
        scores = candidates[:, 4 + label]
        valid = np.flatnonzero(scores >= score_threshold)
        kept = xyxy_nms(boxes[valid], scores[valid], nms_threshold)
        for local_index in kept:
            index = int(valid[local_index])
            selected_boxes.append(boxes[index])
            selected_scores.append(scores[index])
            selected_labels.append(label)
    if not selected_boxes:
        return ()
    scores = np.asarray(selected_scores, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[:max_detections]
    return _source_detections(
        np.asarray(selected_boxes)[order],
        scores[order],
        np.asarray(selected_labels, dtype=np.int64)[order],
        category_ids,
        transform,
    )


def decode_rfdetr(
    boxes_output: np.ndarray,
    logits_output: np.ndarray,
    transform: ImageTransform,
    category_ids: Sequence[int],
    *,
    score_threshold: float,
    max_detections: int,
) -> tuple[Detection, ...]:
    """
    Decode RF-DETR normalized boxes and sigmoid query/class logits.
    """
    boxes = np.asarray(boxes_output, dtype=np.float32)
    logits = np.asarray(logits_output, dtype=np.float32)
    if (
        boxes.ndim != 3
        or logits.ndim != 3
        or boxes.shape[:2] != logits.shape[:2]
        or boxes.shape[0] != 1
        or boxes.shape[2] != 4
        or logits.shape[2] != len(category_ids) + 1
    ):
        raise ValueError("Unexpected RF-DETR output shapes")
    foreground_logits = logits[0, :, :-1]
    scores_all = 1.0 / (1.0 + np.exp(-np.clip(foreground_logits, -88, 88)))
    flat_scores = scores_all.reshape(-1)
    flat_indices = np.arange(len(flat_scores), dtype=np.int64)
    order = np.lexsort((flat_indices, -flat_scores))[:max_detections]
    order = order[flat_scores[order] > score_threshold]
    if not len(order):
        return ()
    labels = order % len(category_ids)
    query_indices = order // len(category_ids)
    xyxy = _cxcywh_to_xyxy(boxes[0, query_indices])
    return _source_detections(
        xyxy,
        flat_scores[order],
        labels,
        category_ids,
        transform,
        direct_normalized=True,
    )


def decode_outputs(
    outputs: Sequence[np.ndarray],
    contract: SessionContract,
    spec: OnnxModelSpec,
    transform: ImageTransform,
    category_ids: Sequence[int],
) -> tuple[Detection, ...]:
    """
    Dispatch family-specific decoding for one session result.
    """
    named_outputs = dict(zip(contract.output_names, outputs, strict=True))
    if spec.family is ModelFamilies.YOLOX:
        assert spec.nms_threshold is not None
        return decode_yolox(
            named_outputs["output"],
            transform,
            category_ids,
            score_threshold=spec.score_threshold,
            nms_threshold=spec.nms_threshold,
            max_detections=spec.max_detections,
        )
    if spec.family is ModelFamilies.YOLO11:
        assert spec.nms_threshold is not None
        return decode_yolo11(
            named_outputs["output0"],
            transform,
            category_ids,
            score_threshold=spec.score_threshold,
            nms_threshold=spec.nms_threshold,
            max_detections=spec.max_detections,
        )
    if spec.family is ModelFamilies.RFDETR:
        return decode_rfdetr(
            named_outputs["dets"],
            named_outputs["labels"],
            transform,
            category_ids,
            score_threshold=spec.score_threshold,
            max_detections=spec.max_detections,
        )
    raise ValueError(f"Unsupported model family: {spec.family}")


def predict_image(
    session: Any,
    spec: OnnxModelSpec,
    image: Image.Image,
    category_ids: Sequence[int],
    *,
    contract: SessionContract | None = None,
) -> tuple[Detection, ...]:
    """
    Run framework-neutral end-to-end inference for one in-memory image.
    """
    contract = contract or validate_session_contract(session, spec.family)
    transform = prepare_input(image, contract, spec.family)
    outputs = session.run(None, {contract.input_name: transform.tensor})
    return decode_outputs(outputs, contract, spec, transform, category_ids)


def benchmark_session(
    session: Any,
    spec: OnnxModelSpec,
    image: Image.Image,
    category_ids: Sequence[int],
    *,
    warmup: int = 5,
    samples: int = 30,
) -> BenchmarkResult:
    """
    Benchmark median end-to-end inference without file or session load time.
    """
    if warmup < 1 or samples < 1:
        raise ValueError("Benchmark warmup and sample counts must be positive")
    contract = validate_session_contract(session, spec.family)
    predictions = predict_image(
        session,
        spec,
        image,
        category_ids,
        contract=contract,
    )
    for _ in range(warmup):
        predict_image(session, spec, image, category_ids, contract=contract)

    milliseconds = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        predict_image(session, spec, image, category_ids, contract=contract)
        milliseconds.append((time.perf_counter_ns() - started) / 1_000_000)

    return BenchmarkResult(
        predictions=predictions,
        latency_ms=float(np.median(milliseconds)),
    )


def _bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """
    Calculate intersection-over-union for two COCO xywh boxes.
    """
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[0] + first[2], second[0] + second[2])
    y2 = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Detection],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Greedily match score-ordered detections to same-category ground truth.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("IoU threshold must be in (0, 1]")
    matched: set[int] = set()
    true_positives = false_positives = 0
    for prediction in sorted(predictions, key=lambda item: -item.score):
        overlaps = [
            (_bbox_iou(prediction.bbox, target["bbox"]), index)
            for index, target in enumerate(ground_truth)
            if index not in matched
            and prediction.category_id == int(target["category_id"])
        ]
        overlap, index = max(overlaps, default=(0.0, -1))
        if overlap >= iou_threshold and index >= 0:
            matched.add(index)
            true_positives += 1
        else:
            false_positives += 1
    false_negatives = len(ground_truth) - len(matched)
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
