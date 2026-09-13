"""
Framework-neutral traversal of a COCO split for model-owned predictors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from detection_common.utils.json_io import read_json

from .metrics import write_prediction_artifact


def export_predictions(
    destination: Path | str,
    annotation_path: Path | str,
    image_dir: Path | str,
    predict_one: Callable[[Image.Image], Sequence[dict[str, Any]]],
    *,
    metadata: dict[str, Any],
) -> Path:
    """
    Visit every annotated image and save one versioned prediction artifact.
    """
    annotations = read_json(annotation_path)
    predictions = []
    for entry in annotations["images"]:
        image_path = Path(image_dir) / entry["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            predictions.extend(
                {"image_id": entry["id"], **row}
                for row in predict_one(image.convert("RGB"))
            )
    return write_prediction_artifact(
        destination, annotation_path, predictions, metadata=metadata
    )
