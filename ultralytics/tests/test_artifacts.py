"""
Tests for model-owned Ultralytics artifact conversion.
"""

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from detection_common.utils.json_io import write_json
from detection_evaluation import read_prediction_artifact
from ultralytics_pipeline.artifacts import export_model_predictions


def test_export_converts_xyxy_and_preserves_image_coverage(tmp_path: Path) -> None:
    """
    Convert native tensors to the shared versioned artifact schema.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_dir / "image.png")
    annotation = tmp_path / "annotations.json"
    write_json(
        annotation,
        {
            "images": [{"id": 2, "file_name": "image.png"}],
            "categories": [{"id": 7, "name": "ball"}],
            "annotations": [],
        },
    )
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")

    class FakeModel:
        """
        Minimal Ultralytics model returning a native-shaped result.
        """

        def predict(self, *_args, **_kwargs) -> list[SimpleNamespace]:
            """
            Return one xyxy detection.
            """
            boxes = SimpleNamespace(
                xyxy=torch.tensor([[1.0, 2.0, 4.0, 6.0]]),
                conf=torch.tensor([0.8]),
                cls=torch.tensor([0.0]),
            )
            return [SimpleNamespace(boxes=boxes)]

    artifact = export_model_predictions(
        FakeModel(),
        tmp_path / "predictions.json",
        annotation,
        image_dir,
        category_id=7,
        resolution=8,
        device="cpu",
        metadata={
            "model": "yolo11n",
            "split": "val",
            "run_dir": str(tmp_path),
            "checkpoint": str(checkpoint),
            "resolution": 8,
            "smoke_run": False,
            "postprocessing": {"score_floor": 0.001},
        },
    )
    document = read_prediction_artifact(artifact, annotation)
    assert document["image_ids"] == [2]
    assert document["predictions"][0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
