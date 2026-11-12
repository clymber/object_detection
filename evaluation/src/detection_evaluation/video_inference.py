"""
Select basketball ONNX exports and stream full-video inference artifacts.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import onnx
import onnxruntime as ort
import yaml
from detection_common.utils.digest import file_digest
from detection_common.utils.json_io import json_normalize, read_json, write_json
from PIL import Image

from .baseline import find_latest_path
from .config import WORKSPACE_ROOT
from .onnx_inference import (
    Detection,
    ModelFamilies,
    OnnxModelSpec,
    ProviderMode,
    create_session,
    predict_image,
    validate_session_contract,
)
from .run_protocol import read_run_protocol, read_training_record


@dataclass(frozen=True)
class VideoModel:
    """
    Retain only the selected export and its compact saved provenance.
    """

    spec: OnnxModelSpec
    run_dir: Path
    category_id: int
    source_fingerprint: str


def portable_path(path: Path | str, workspace_root: Path = WORKSPACE_ROOT) -> str:
    """
    Express a resolved path relative to the workspace, including external roots.
    """
    return Path(
        os.path.relpath(Path(path).resolve(), workspace_root.resolve())
    ).as_posix()


def _run_directory(model_path: Path, runs_dir: Path) -> Path:
    """
    Find the nearest saved protocol without searching above the runs directory.
    """
    for parent in model_path.parents:
        if not parent.is_relative_to(runs_dir):
            break
        if (parent / "run_protocol.json").is_file():
            return parent
    raise ValueError("No enclosing run_protocol.json within the runs directory")


def _basketball_category(protocol: dict[str, Any], run_dir: Path) -> int:
    """
    Read the canonical single-class mapping and check an RF-DETR manifest if saved.
    """
    categories = protocol["dataset_identity"]["canonical"]["canonical_source"][
        "category_mapping"
    ]
    if (
        not isinstance(categories, list)
        or len(categories) != 1
        or categories[0].get("name") != "basketball"
        or type(categories[0].get("id")) is not int
    ):
        raise ValueError(
            "Expected one canonical basketball category with an integer ID"
        )
    category_id = categories[0]["id"]
    manifest_path = run_dir / "dataset_manifest.json"
    if protocol["model"] == "rfdetr_small" and manifest_path.is_file():
        mapping = read_json(manifest_path)["category_mapping"]
        if mapping["class_names"] != ["basketball"] or mapping[
            "prediction_to_source"
        ] != {"0": category_id}:
            raise ValueError("RF-DETR manifest disagrees with the canonical mapping")
    return category_id


def _postprocessing(run_dir: Path, family: ModelFamilies) -> tuple[float | None, int]:
    """
    Reuse saved NMS and detection limits with the image notebook's fallbacks.
    """
    settings: dict[str, Any] = {}
    for path in (
        run_dir / "evaluation" / "test_predictions.json",
        run_dir / "test_predictions.json",
    ):
        if path.is_file():
            settings = read_json(path).get("metadata", {}).get("postprocessing", {})
            break
    nms = None
    if family is ModelFamilies.YOLOX:
        args_path = run_dir / "args.yaml"
        args = yaml.safe_load(args_path.read_text()) if args_path.is_file() else {}
        fallback = args.get("nms_threshold", 0.65) if isinstance(args, dict) else 0.65
        nms = float(settings.get("nms_iou", fallback))
    elif family is ModelFamilies.YOLO11:
        nms = float(settings.get("nms_iou", 0.70))
    return nms, int(settings.get("max_det", 300))


def discover_video_models(
    runs_dir: Path | str,
    declarations: Sequence[tuple[str, ModelFamilies, str]],
    *,
    confidence_threshold: float = 0.25,
) -> tuple[list[VideoModel], list[dict[str, str]]]:
    """
    Select the newest eligible export per model and report rejected candidates.

    Graph validation happens after selection; an incompatible selected graph must
    fail visibly rather than cause selection of an older export.
    """
    runs_dir = Path(runs_dir).expanduser().resolve()
    selected = []
    excluded = []
    missing = []
    for name, family, pattern in declarations:
        eligible = {}
        for path in runs_dir.glob(pattern):
            if not path.is_file():
                continue
            try:
                run_dir = _run_directory(path.resolve(), runs_dir)
                protocol = read_run_protocol(run_dir)
                if (
                    protocol["model"] != name
                    or protocol["logical_dataset"] != "basketball"
                ):
                    raise ValueError("Unexpected model or logical dataset")
                settings = protocol["training_settings"]
                if (
                    protocol["smoke_run"]
                    or settings.get("train_batch_limit") is not None
                    or float(settings.get("fraction", 1.0)) != 1.0
                ):
                    raise ValueError("Requires full data without smoke or batch limits")
                training = read_training_record(run_dir)
                if (
                    not training["attempts"]
                    or training["summary"]["status"] != "completed"
                ):
                    raise ValueError("Training record is not completed")
                category_id = _basketball_category(protocol, run_dir)
                nms, max_detections = _postprocessing(run_dir, family)
                eligible[path] = VideoModel(
                    spec=OnnxModelSpec(
                        name,
                        family,
                        path.resolve(),
                        confidence_threshold,
                        nms,
                        max_detections,
                    ),
                    run_dir=run_dir,
                    category_id=category_id,
                    source_fingerprint=protocol["dataset_identity"]["canonical"][
                        "source_fingerprint"
                    ],
                )
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                excluded.append(
                    {
                        "model": name,
                        "onnx_path": portable_path(path),
                        "reason": str(error),
                    }
                )
        latest = find_latest_path(runs_dir, pattern, predicate=eligible.__contains__)
        if latest is not None:
            selected.append(eligible[latest])
        else:
            missing.append(name)
    if missing:
        reasons = "\n".join(f"{row['onnx_path']}: {row['reason']}" for row in excluded)
        raise FileNotFoundError(
            "Missing eligible basketball ONNX exports for: "
            + ", ".join(missing)
            + ("\nExcluded candidates:\n" + reasons if reasons else "")
        )
    if len({model.source_fingerprint for model in selected}) != 1:
        raise ValueError("Selected runs have different canonical source fingerprints")
    return selected, excluded


def describe_model(model: VideoModel) -> dict[str, Any]:
    """
    Hash one selected export and describe its identity and inference policies.
    """
    spec = model.spec
    stat = spec.path.stat()
    return {
        "model": spec.name,
        "family": spec.family.value,
        "run_dir": portable_path(model.run_dir),
        "onnx_path": portable_path(spec.path),
        "onnx_sha256": file_digest(spec.path),
        "onnx_size_bytes": stat.st_size,
        "onnx_mtime_ns": stat.st_mtime_ns,
        "source_fingerprint": model.source_fingerprint,
        "class_mapping": [
            {
                "model_class_id": 0,
                "category_id": model.category_id,
                "class_name": "basketball",
            }
        ],
        "confidence_threshold": spec.score_threshold,
        "nms_iou": spec.nms_threshold,
        "max_detections": spec.max_detections,
    }


def frame_record(
    detections: Sequence[Detection], frame_index: int, fps: float
) -> dict[str, Any]:
    """
    Serialize source-pixel boxes and centers with a nominal frame timestamp.
    """
    rows = []
    for detection in detections:
        x, y, width, height = detection.bbox
        rows.append(
            {
                "category_id": detection.category_id,
                "class_name": "basketball",
                "bbox": list(detection.bbox),
                "center_x": x + width / 2,
                "center_y": y + height / 2,
                "confidence": detection.score,
            }
        )
    return {
        "frame_index": frame_index,
        "timestamp_s": frame_index / fps,
        "detections": rows,
    }


def _annotate(frame: Any, detections: Sequence[Detection]) -> Any:
    """
    Draw basketball boxes and confidence labels on a copy of the BGR frame.
    """
    annotated = frame.copy()
    for detection in detections:
        x, y, width, height = detection.bbox
        start = (round(x), round(y))
        end = (round(x + width), round(y + height))
        cv2.rectangle(annotated, start, end, (0, 255, 0), 2)
        origin = (start[0], max(18, start[1] - 6))
        label = f"basketball {detection.score:.2f}"
        cv2.putText(
            annotated,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_video_inference(
    model: VideoModel,
    video_path: Path | str,
    output_dir: Path | str,
    mode: ProviderMode,
) -> dict[str, Any]:
    """
    Process a clip to EOF once, closing resources before publishing its summary.

    Failures propagate after writing a partial summary with the failing frame.
    Only the current frame and small counters are retained in memory.
    """
    video_path = Path(video_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "partial",
        "model": model.spec.name,
        "video_path": portable_path(video_path),
        "provider_mode": mode.name,
        "configured_providers": json_normalize(mode.providers),
        "actual_providers": [],
        "runtime": {
            "onnxruntime": ort.__version__,
            "onnx": onnx.__version__,
            "host": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
        "timestamp_basis": "nominal frame_index / source_fps; not WebM PTS",
        "audio_preserved": False,
        "timing": {
            "processing": "capture/writer setup through output closure; includes I/O",
            "inference": "RGB conversion, preprocessing, session.run, postprocessing",
            "first_inference_included": True,
            "warmup_runs": 0,
        },
        "processing_seconds": 0.0,
        "processed_frames": 0,
        "total_detections": 0,
        "frames_with_detections": 0,
        "warnings": [],
    }
    session = None
    started = None
    inference_seconds = 0.0
    failing_frame = None
    try:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        summary.update(describe_model(model))
        summary["video_sha256"] = file_digest(video_path)
        onnx.checker.check_model(str(model.spec.path))
        session = create_session(model.spec.path, mode)
        # Keep provider failures visible; per-node CPU fallback remains configured.
        session.disable_fallback()
        summary["actual_providers"] = session.get_providers()
        requested = [
            item if isinstance(item, str) else item[0] for item in mode.providers
        ]
        if any(provider not in summary["actual_providers"] for provider in requested):
            raise RuntimeError("Session is missing a requested execution provider")
        contract = validate_session_contract(session, model.spec.family)

        started = time.perf_counter()
        with ExitStack() as resources:
            capture = cv2.VideoCapture(str(video_path))
            resources.callback(capture.release)
            if not capture.isOpened():
                raise ValueError(f"Cannot open source video: {video_path}")
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            if any(
                not math.isfinite(value) or value <= 0 for value in (width, height, fps)
            ):
                raise ValueError(
                    "Source video requires positive dimensions and finite FPS"
                )
            width, height = int(width), int(height)
            reported = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            reported = (
                int(reported) if math.isfinite(reported) and reported > 0 else None
            )
            summary["source"] = {
                "width": width,
                "height": height,
                "fps": fps,
                "reported_frames": reported,
            }
            writer = cv2.VideoWriter(
                str(output_dir / "annotated.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            resources.callback(writer.release)
            if not writer.isOpened():
                raise RuntimeError("Cannot write MP4 with the installed mp4v codec")
            stream = resources.enter_context(
                (output_dir / "detections.jsonl").open("x", encoding="utf-8")
            )
            while True:
                readable, frame = capture.read()
                if not readable:
                    break
                failing_frame = summary["processed_frames"]
                if frame is None or frame.shape[:2] != (height, width):
                    raise ValueError("Decoded frame does not match source dimensions")
                inference_started = time.perf_counter()
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detections = predict_image(
                    session, model.spec, image, (model.category_id,), contract=contract
                )
                elapsed = time.perf_counter() - inference_started
                record = frame_record(detections, failing_frame, fps)
                writer.write(_annotate(frame, detections))
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                inference_seconds += elapsed
                summary["processed_frames"] += 1
                summary["total_detections"] += len(detections)
                summary["frames_with_detections"] += bool(detections)
                failing_frame = summary["processed_frames"]
            if not summary["processed_frames"]:
                raise ValueError("Source video is empty or has no decodable frames")
        # Output closure belongs to processing time; metadata reporting does not.
        summary["processing_seconds"] = time.perf_counter() - started
        started = None
        if reported is not None and reported != summary["processed_frames"]:
            summary["warnings"].append(
                f"Decoded {summary['processed_frames']} frames; source reported {reported}"
            )
        summary["status"] = "completed"
    except BaseException as error:
        summary["failing_frame"] = failing_frame
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if started is not None:
            summary["processing_seconds"] = time.perf_counter() - started
        session = None
        frames = summary["processed_frames"]
        duration = summary.get("processing_seconds", 0.0)
        summary["frames_without_detections"] = (
            frames - summary["frames_with_detections"]
        )
        summary["processing_fps"] = frames / duration if duration else None
        summary["mean_inference_ms"] = (
            1000 * inference_seconds / frames if frames else None
        )
        write_json(output_dir / "summary.json", summary, allow_nan=False)
    return summary


def comparison_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compare completed models only, requiring equal decoded frame counts.
    """
    completed = [summary for summary in summaries if summary["status"] == "completed"]
    if len({summary["processed_frames"] for summary in completed}) > 1:
        raise ValueError("Completed models processed different numbers of frames")
    fields = (
        "model",
        "provider_mode",
        "processed_frames",
        "total_detections",
        "frames_with_detections",
        "frames_without_detections",
        "processing_seconds",
        "processing_fps",
        "mean_inference_ms",
    )
    return [{field: summary[field] for field in fields} for summary in completed]
