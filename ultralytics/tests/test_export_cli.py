"""
Smoke-test the model-owned export command without a real checkpoint.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from PIL import Image

from ultralytics_pipeline import export_cli


def test_parse_export_args(tmp_path: Path) -> None:
    """
    Require explicit frozen data and an unused artifact destination.
    """
    args = export_cli.parse_args(
        ["--dataset-dir", str(tmp_path / "dataset"), "--output-dir", str(tmp_path)]
    )
    assert args.device == "cpu"
    assert args.split == "both"


def test_export_calls_native_adapter_and_neutral_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Keep checkpoint loading in this package and artifact writing in evaluation.
    """
    context = SimpleNamespace(
        checkpoint=tmp_path / "best.pt",
        category_id=7,
        resolution=640,
        device="cpu",
        training_settings={"iou": 0.6},
    )
    monkeypatch.setattr(export_cli, "prepare_baseline", lambda *a, **k: context)
    monkeypatch.setattr(export_cli, "configure_privacy", lambda: None)
    monkeypatch.setattr(export_cli, "framework_version", lambda: "test-version")
    rows = []

    def fake_writer(prepared, predict_one, *, metadata, benchmark):
        """
        Exercise the callback handed to the neutral artifact writer.
        """
        assert prepared is context
        assert metadata["postprocessing"]["nms_iou"] == 0.6
        assert not benchmark
        rows.extend(predict_one(Image.new("RGB", (8, 8))))
        return [tmp_path / "artifact.json"]

    monkeypatch.setattr(export_cli, "export_baseline", fake_writer)

    class FakeYOLO:
        """
        Return one native-looking Ultralytics result without model weights.
        """

        def __init__(self, checkpoint: Path) -> None:
            """
            Record the selected checkpoint and one-class model metadata.
            """
            assert checkpoint == context.checkpoint
            self.names = {0: "basketball"}
            self.model = torch.nn.Linear(1, 1)

        def predict(self, image: Image.Image, **kwargs):
            """
            Return original-pixel xyxy boxes after mock native prediction.
            """
            assert image.size == (8, 8)
            assert kwargs["iou"] == 0.6
            boxes = SimpleNamespace(
                xyxy=torch.tensor([[1.0, 2.0, 4.0, 6.0]]),
                conf=torch.tensor([0.9]),
                cls=torch.tensor([0.0]),
            )
            return [SimpleNamespace(boxes=boxes)]

    fake_module = ModuleType("ultralytics")
    fake_module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    args = export_cli.parse_args(
        ["--run-dir", str(tmp_path), "--dataset-dir", str(tmp_path),
         "--output-dir", str(tmp_path)]
    )
    assert export_cli.export(args) == [tmp_path / "artifact.json"]
    assert rows[0]["category_id"] == 7
    assert rows[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
