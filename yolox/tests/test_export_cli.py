"""
Smoke-test the YOLOX export command with a mocked native model.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from yolox_pipeline import export_cli, yolox


def test_parse_export_args(tmp_path: Path) -> None:
    """
    Keep Tiny/Nano choice explicit without importing a checkpoint.
    """
    args = export_cli.parse_args(["--model", "nano", "--dataset-dir", str(tmp_path)])
    assert args.model == "nano"
    assert args.device == "cpu"
    assert args.output_dir is None


def test_export_calls_native_adapter_and_neutral_writer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Load model settings in YOLOX and pass COCO rows to evaluation.
    """
    context = SimpleNamespace(
        run_dir=tmp_path / "run",
        dataset_dir=tmp_path / "dataset",
        checkpoint=tmp_path / "run" / "weights" / "best_ckpt.pth",
        category_id=7,
        resolution=640,
        device="cpu",
        training_settings={"classes": ["basketball"], "epochs": 5},
    )
    monkeypatch.setattr(export_cli, "prepare_baseline", lambda *a, **k: context)
    monkeypatch.setattr(export_cli, "version", lambda distribution: "test-version")

    class FakeExperiment:
        """
        Expose the inference settings expected by the native adapter.
        """

        def __init__(self, **kwargs) -> None:
            """
            Check that the selected dataset and run are passed through.
            """
            assert kwargs["dataset_dir"] == context.dataset_dir
            self.nmsthre = 0.65

    monkeypatch.setattr(yolox, "BasketballTinyExp", FakeExperiment)
    monkeypatch.setattr(
        yolox, "load_trained_model", lambda experiment, checkpoint, device:
        torch.nn.Linear(1, 1),
    )
    native_calls = []

    def fake_predict(model, experiment, bgr, device, ids, *, conf_threshold):
        """
        Confirm RGB-to-BGR conversion and category mapping.
        """
        native_calls.append((tuple(bgr[0, 0]), ids, conf_threshold))
        return [{"category_id": ids[0], "bbox": [1, 2, 3, 4], "score": 0.9}]

    monkeypatch.setattr(yolox, "predict_image", fake_predict)
    rows = []

    def fake_writer(prepared, predict_one, *, metadata, benchmark):
        """
        Exercise one model prediction at the neutral handoff boundary.
        """
        assert prepared is context
        assert metadata["postprocessing"]["nms_iou"] == 0.65
        assert not benchmark
        rows.extend(predict_one(Image.new("RGB", (8, 8), (10, 20, 30))))
        return [tmp_path / "artifact.json"]

    monkeypatch.setattr(export_cli, "export_baseline", fake_writer)
    args = export_cli.parse_args(
        ["--model", "tiny", "--run-dir", str(context.run_dir),
            "--dataset-dir", str(context.dataset_dir)]
    )
    assert export_cli.export(args) == [tmp_path / "artifact.json"]
    assert native_calls == [((30, 20, 10), [7], 0.001)]
    assert rows[0]["category_id"] == 7
