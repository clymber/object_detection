"""
Run native RF-DETR on a local video with streaming detection records.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from detection_common.utils.digest import file_digest
from detection_common.utils.json_io import write_json

from .config import OUTPUT_ROOT


def _frame_record(
    detections: sv.Detections,
    frame_index: int,
    fps: float,
    category_id: int,
    class_name: str,
) -> dict[str, Any]:
    """
    Serialize native source-pixel boxes and their centers for one frame.
    """
    rows = []
    for box, confidence in zip(detections.xyxy, detections.confidence, strict=True):
        x1, y1, x2, y2 = map(float, box)
        rows.append(
            {
                "category_id": category_id,
                "class_name": class_name,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
                "confidence": float(confidence),
            }
        )
    return {
        "frame_index": frame_index,
        "timestamp_s": frame_index / fps,
        "detections": rows,
    }


def run_video_inference(
    model: Any,
    video_path: Path | str,
    checkpoint_path: Path | str,
    *,
    category_id: int = 1,
    confidence_threshold: float = 0.25,
    max_frames: int | None = None,
    output_root: Path | str = OUTPUT_ROOT,
    show_progress: bool = True,
) -> Path:
    """
    Save an annotated MP4, JSONL records, and a processing summary in a fresh run.

    The loaded model must have one foreground class. Completion reports that
    processing returned without an exception; saved-video verification is manual.
    """
    video_path = Path(video_path).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    for path in (video_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between zero and one")
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be a positive integer or None")
    if len(model.class_names) != 1:
        raise ValueError("Video inference requires one foreground class")
    class_name = model.class_names[0]
    video_info = sv.VideoInfo.from_video_path(str(video_path))
    if (
        video_info.width <= 0
        or video_info.height <= 0
        or not math.isfinite(video_info.fps)
        or video_info.fps <= 0
    ):
        raise ValueError("Source video requires positive dimensions and finite FPS")
    checkpoint_sha256 = file_digest(checkpoint_path)
    base = Path(output_root).expanduser().resolve() / "rfdetr" / "video_inference"
    base.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_"
    output_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    processed_frames = 0
    total_detections = 0

    started = time.perf_counter()
    with (output_dir / "detections.jsonl").open(
        "x", encoding="utf-8", buffering=1
    ) as stream:

        def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
            """
            Predict, annotate, and log one BGR frame, retaining empty frames.
            """
            nonlocal processed_frames, total_detections
            try:
                detections = model.predict(
                    frame[:, :, ::-1],
                    threshold=confidence_threshold,
                    include_source_image=False,
                )
                detections = detections[detections.class_id == 0]
                record = _frame_record(
                    detections, frame_index, video_info.fps, category_id, class_name
                )
                labels = [
                    f"{class_name} {confidence:.2f}"
                    for confidence in detections.confidence
                ]
                annotated = box_annotator.annotate(frame.copy(), detections=detections)
                annotated = label_annotator.annotate(
                    annotated, detections=detections, labels=labels
                )
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                processed_frames += 1
                total_detections += len(detections)
                return annotated
            except Exception as error:
                raise RuntimeError(
                    f"Video processing failed at frame {frame_index}: {error}"
                ) from error

        sv.process_video(
            source_path=str(video_path),
            target_path=str(output_dir / "annotated.mp4"),
            callback=callback,
            max_frames=max_frames,
            prefetch=1,
            writer_buffer=1,
            preserve_audio=False,
            show_progress=show_progress,
        )
    duration_s = time.perf_counter() - started
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "status": "completed",
            "video_path": video_path,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "device": str(model.model.device),
            "confidence_threshold": confidence_threshold,
            "class_mapping": [
                {
                    "model_class_id": 0,
                    "category_id": category_id,
                    "class_name": class_name,
                }
            ],
            "source": {
                "width": video_info.width,
                "height": video_info.height,
                "fps": video_info.fps,
                "reported_frames": video_info.total_frames,
            },
            "max_frames": max_frames,
            "processed_frames": processed_frames,
            "total_detections": total_detections,
            "duration_s": duration_s,
            "processing_fps": processed_frames / duration_s,
        },
        allow_nan=False,
    )
    return output_dir
