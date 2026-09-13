"""
Tests for model-owned YOLOX artifact conversion.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from detection_common.utils.json_io import write_json
from detection_evaluation import read_prediction_artifact
from yolox_pipeline import yolox
from yolox_pipeline.artifacts import export_model_predictions


def test_export_calls_native_predictor_for_every_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Feed native BGR input through the YOLOX predictor and save COCO rows.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(image_dir / "image.png")
    annotation = tmp_path / "annotations.json"
    write_json(
        annotation,
        {
            "images": [{"id": 2, "file_name": "image.png"}],
            "categories": [{"id": 7, "name": "ball"}],
            "annotations": [],
        },
    )
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"weights")
    observed = []

    def fake_predict(_model, _exp, image, _device, ids, **kwargs) -> list[dict]:
        """
        Record the converted pixel and selected category mapping.
        """
        observed.append((image[0, 0].tolist(), ids, kwargs["conf_threshold"]))
        return [{"category_id": 7, "bbox": [1, 2, 3, 4], "score": 0.8}]

    monkeypatch.setattr(yolox, "predict_image", fake_predict)
    artifact = export_model_predictions(
        torch.nn.Linear(1, 1),
        object(),
        tmp_path / "predictions.json",
        annotation,
        image_dir,
        device=torch.device("cpu"),
        category_ids=[7],
        metadata={
            "model": "yolox_tiny",
            "split": "val",
            "run_dir": str(tmp_path),
            "checkpoint": str(checkpoint),
            "resolution": 8,
            "smoke_run": False,
            "postprocessing": {"score_floor": 0.001},
        },
    )
    document = read_prediction_artifact(artifact, annotation)
    assert observed == [([0, 0, 255], [7], 0.001)]
    assert np.asarray(document["predictions"][0]["bbox"]).tolist() == [1, 2, 3, 4]
