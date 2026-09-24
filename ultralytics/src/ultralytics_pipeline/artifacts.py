"""
Ultralytics-owned conversion to the neutral prediction artifact schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from detection_evaluation import export_predictions
from PIL import Image


def predict_image(
    model: Any,
    image: Image.Image,
    *,
    category_id: int | None = None,
    category_ids: list[int] | None = None,
    resolution: int,
    device: str,
    score_floor: float = 0.001,
    nms_iou: float = 0.7,
    max_det: int = 300,
    agnostic_nms: bool = False,
) -> list[dict]:
    """
    Convert one Ultralytics result's xyxy tensors to COCO xywh rows.
    """
    if category_ids is None:
        category_ids = [] if category_id is None else [category_id]
    if not category_ids:
        raise ValueError("Category IDs are required")
    result = next(
        iter(
            model.predict(
                image,
                imgsz=resolution,
                device=device,
                conf=score_floor,
                iou=nms_iou,
                max_det=max_det,
                agnostic_nms=agnostic_nms,
                verbose=False,
                rect=True,
                stream=False,
            )
        ),
        None,
    )
    if result is None or result.boxes is None:
        raise RuntimeError("Ultralytics returned no detection boxes")
    boxes = result.boxes
    if not all(
        isinstance(value, torch.Tensor)
        for value in (boxes.xyxy, boxes.conf, boxes.cls)
    ):
        raise TypeError("Ultralytics result boxes must contain tensors")
    rows = []
    for box, score, label in zip(
        boxes.xyxy.cpu().tolist(),
        boxes.conf.cpu().tolist(),
        boxes.cls.cpu().tolist(),
        strict=True,
    ):
        if label != int(label) or not 0 <= int(label) < len(category_ids):
            raise ValueError("Unexpected Ultralytics class label")
        x1, y1, x2, y2 = box
        rows.append(
            {
                "category_id": category_ids[int(label)],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            }
        )
    return rows


def export_model_predictions(
    model: Any,
    destination: Path | str,
    annotation_path: Path | str,
    image_dir: Path | str,
    *,
    category_id: int | None = None,
    category_ids: list[int] | None = None,
    resolution: int,
    device: str,
    metadata: dict[str, Any],
) -> Path:
    """
    Export one complete split through Ultralytics' native prediction path.
    """
    postprocessing = metadata["postprocessing"]

    def predict_one(image: Image.Image) -> list[dict]:
        """
        Apply the selected native postprocessing settings to one image.
        """
        return predict_image(
            model,
            image,
            category_id=category_id,
            category_ids=category_ids,
            resolution=resolution,
            device=device,
            score_floor=postprocessing["score_floor"],
            nms_iou=postprocessing.get("nms_iou", 0.7),
            max_det=postprocessing.get("max_det", 300),
            agnostic_nms=postprocessing.get("agnostic_nms", False),
        )

    return export_predictions(
        destination, annotation_path, image_dir, predict_one, metadata=metadata
    )
