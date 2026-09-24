"""
Tests for framework-neutral image coverage during artifact export.
"""

from pathlib import Path

from detection_common.utils.json_io import write_json
from PIL import Image

from detection_evaluation import export_predictions, read_prediction_artifact


def test_export_visits_positive_and_negative_images(tmp_path: Path) -> None:
    """
    Store a complete artifact even when one image has no detections.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("positive.png", "negative.png"):
        Image.new("RGB", (8, 8), "white").save(image_dir / name)
    annotation = tmp_path / "annotations.json"
    write_json(
        annotation,
        {
            "images": [
                {"id": 1, "file_name": "positive.png"},
                {"id": 2, "file_name": "negative.png"},
            ],
            "categories": [{"id": 7, "name": "ball"}],
            "annotations": [],
        },
    )
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls = 0

    def predict_one(image: Image.Image) -> list[dict]:
        """
        Return one detection on the first image and none on the second.
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            return []
        return [{"category_id": 7, "bbox": [1, 1, 2, 2], "score": 0.9}]

    artifact = export_predictions(
        tmp_path / "predictions.json",
        annotation,
        image_dir,
        predict_one,
        metadata={
            "model": "fake-model",
            "split": "val",
            "run_dir": str(tmp_path),
            "checkpoint": str(checkpoint),
            "resolution": 8,
            "smoke_run": False,
            "postprocessing": {"score_floor": 0.001},
        },
    )
    document = read_prediction_artifact(artifact, annotation)
    assert calls == 2
    assert document["image_ids"] == [1, 2]
    assert [row["image_id"] for row in document["predictions"]] == [1]
