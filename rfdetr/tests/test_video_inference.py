"""
Test native video callbacks and records without a model or video codec.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import supervision as sv
from detection_common.utils.digest import file_digest
from detection_common.utils.json_io import read_json

from rfdetr_pipeline import video_inference


@pytest.fixture
def video_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """
    Supply existing input files and metadata without decoding a real video.
    """
    video = tmp_path / "source.mp4"
    checkpoint = tmp_path / "checkpoint_best_total.pth"
    video.write_bytes(b"source")
    checkpoint.write_bytes(b"weights")

    def video_info(path: str) -> sv.VideoInfo:
        """
        Return deliberately inaccurate container counts for callback-count tests.
        """
        assert Path(path) == video
        return sv.VideoInfo(width=60, height=40, fps=30.0, total_frames=999)

    monkeypatch.setattr(sv.VideoInfo, "from_video_path", video_info)
    return video, checkpoint


def test_video_records_preserve_geometry_empty_frames_and_previous_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_inputs: tuple[Path, Path],
) -> None:
    """
    Exercise RGB input, background filtering, overlays, JSONL, and fresh outputs.
    """
    video, checkpoint = video_inputs
    frames = [np.zeros((40, 60, 3), dtype=np.uint8) for _ in range(2)]
    frames[0][:, :, 0] = 20
    predictions = []
    returned_frames = []

    class FakeModel:
        """
        Return one foreground box and one no-object box, then an empty frame.
        """

        class_names = ("basketball",)
        model = SimpleNamespace(device="cpu")

        def predict(self, image, *, threshold, include_source_image):
            """
            Retain native prediction arguments and supply stable detections.
            """
            predictions.append(image.copy())
            assert threshold == 0.25
            assert include_source_image is False
            if len(predictions) == 1:
                return sv.Detections(
                    xyxy=np.array([[2.5, 4, 12.5, 14], [20, 20, 30, 30]]),
                    confidence=np.array([0.87, 0.99]),
                    class_id=np.array([0, 1]),
                )
            return sv.Detections.empty()

    def process_video(**kwargs):
        """
        Invoke the real callback while replacing Supervision's video I/O only.
        """
        assert kwargs["source_path"] == str(video)
        assert kwargs["max_frames"] is None
        assert kwargs["prefetch"] == kwargs["writer_buffer"] == 1
        assert kwargs["preserve_audio"] is False
        for index, frame in enumerate(frames):
            returned_frames.append(kwargs["callback"](frame, index))
        Path(kwargs["target_path"]).write_bytes(b"video")

    monkeypatch.setattr(sv, "process_video", process_video)
    model = FakeModel()
    output_dir = video_inference.run_video_inference(
        model, video, checkpoint, category_id=7, output_root=tmp_path
    )
    records = [
        json.loads(line)
        for line in (output_dir / "detections.jsonl").read_text().splitlines()
    ]
    assert predictions[0][0, 0].tolist() == [0, 0, 20]
    assert frames[0][0, 0].tolist() == [20, 0, 0]
    assert not np.array_equal(returned_frames[0], frames[0])
    np.testing.assert_array_equal(returned_frames[1], frames[1])
    assert [record["frame_index"] for record in records] == [0, 1]
    assert records[1]["timestamp_s"] == pytest.approx(1 / 30)
    assert records[1]["detections"] == []
    assert records[0]["detections"] == [
        {
            "category_id": 7,
            "class_name": "basketball",
            "bbox": [2.5, 4, 10, 10],
            "center_x": 7.5,
            "center_y": 9,
            "confidence": pytest.approx(0.87),
        }
    ]
    summary = read_json(output_dir / "summary.json")
    assert summary["status"] == "completed"
    assert summary["processed_frames"] == 2
    assert summary["total_detections"] == 1
    assert summary["checkpoint_sha256"] == file_digest(checkpoint)
    assert summary["device"] == "cpu"
    assert summary["processing_fps"] == pytest.approx(2 / summary["duration_s"])
    assert {path.name for path in output_dir.iterdir()} == {
        "annotated.mp4",
        "detections.jsonl",
        "summary.json",
    }
    second = video_inference.run_video_inference(
        model, video, checkpoint, output_root=tmp_path
    )
    assert second != output_dir
    assert read_json(output_dir / "summary.json") == summary


@pytest.mark.parametrize("failure", ["prediction", "video"])
def test_video_errors_leave_partial_artifacts_without_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    video_inputs: tuple[Path, Path],
    failure: str,
) -> None:
    """
    Preserve library errors and report callback failures with their frame index.
    """
    video, checkpoint = video_inputs
    original_error = RuntimeError("injected failure")

    class FailingModel:
        """
        Raise a prediction error without loading a native model.
        """

        class_names = ("basketball",)
        model = SimpleNamespace(device="cpu")

        def predict(self, *args, **kwargs):
            """
            Simulate a native model error for the failing frame.
            """
            raise original_error

    def process_video(**kwargs):
        """
        Simulate either a library failure or a failure inside the callback.
        """
        if failure == "video":
            raise original_error
        kwargs["callback"](np.zeros((40, 60, 3), dtype=np.uint8), 7)

    monkeypatch.setattr(sv, "process_video", process_video)
    with pytest.raises(RuntimeError) as caught:
        video_inference.run_video_inference(
            FailingModel(), video, checkpoint, output_root=tmp_path
        )
    if failure == "prediction":
        assert "frame 7" in str(caught.value)
        assert caught.value.__cause__ is original_error
    else:
        assert caught.value is original_error
    runs_root = tmp_path / "rfdetr" / "video_inference"
    assert len(list(runs_root.glob("*/detections.jsonl"))) == 1
    assert not list(runs_root.glob("*/summary.json"))
