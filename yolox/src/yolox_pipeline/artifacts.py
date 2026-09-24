"""
YOLOX-owned conversion to the neutral prediction artifact schema.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from detection_evaluation import export_predictions
from PIL import Image

from . import yolox


def predict_image(
    model: torch.nn.Module,
    experiment: Any,
    image: Image.Image,
    *,
    device: torch.device,
    category_ids: Sequence[int],
    score_floor: float = 0.001,
) -> list[dict]:
    """
    Convert RGB input to YOLOX's BGR path and return COCO prediction rows.
    """
    bgr = np.asarray(image)[:, :, ::-1].copy()
    return yolox.predict_image(
        model,
        experiment,
        bgr,
        device,
        list(category_ids),
        conf_threshold=score_floor,
    )


def export_model_predictions(
    model: torch.nn.Module,
    experiment: Any,
    destination: Path | str,
    annotation_path: Path | str,
    image_dir: Path | str,
    *,
    device: torch.device,
    category_ids: Sequence[int],
    metadata: dict[str, Any],
) -> Path:
    """
    Export one complete split using YOLOX's native letterbox and CPU-NMS path.
    """
    score_floor = metadata["postprocessing"]["score_floor"]

    def predict_one(image: Image.Image) -> list[dict]:
        """
        Run the model-owned single-image prediction adapter.
        """
        return predict_image(
            model,
            experiment,
            image,
            device=device,
            category_ids=category_ids,
            score_floor=score_floor,
        )

    return export_predictions(
        destination, annotation_path, image_dir, predict_one, metadata=metadata
    )
