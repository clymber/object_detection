"""
Test export discovery and full-video artifacts with CPU-only synthetic inputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from detection_common.utils.json_io import read_json, write_json

from detection_evaluation import video_inference as video
from detection_evaluation.onnx_inference import ModelFamilies, ProviderMode
from detection_evaluation.run_protocol import (
    capture_training_hardware,
    create_run_protocol,
    finalize_training_attempt,
    start_training_attempt,
)

DECLARATIONS = (("yolox_tiny", ModelFamilies.YOLOX, "yolox_tiny*/**/*.onnx"),)
CPU = ProviderMode("cpu_reference", ("CPUExecutionProvider",))


def _saved_export(
    runs_dir: Path,
    suffix: str = "a",
    *,
    name: str = "yolox_tiny",
    smoke: bool = False,
    settings: dict | None = None,
    status: str = "completed",
    fingerprint: str = "a" * 64,
) -> Path:
    """
    Write a tiny real run protocol and finalized training record without weights.
    """
    run_dir = runs_dir / f"{name}_{suffix}"
    create_run_protocol(
        run_dir,
        logical_dataset="basketball",
        model=name,
        source_notebook="test.py",
        original_utc="2026-11-01T00:00:00Z",
        training_settings=settings or {},
        smoke_run=smoke,
        canonical_dataset_identity={
            "source_fingerprint": fingerprint,
            "canonical_source": {
                "category_mapping": [{"id": 7, "name": "basketball"}],
            },
        },
        loader_dataset_identity={
            "source_fingerprint": fingerprint,
            "loader_fingerprint": "b" * 64,
        },
    )
    attempt = start_training_attempt(
        run_dir,
        resumed=False,
        training_hardware=capture_training_hardware(device="cpu"),
    )
    finalize_training_attempt(run_dir, attempt, outcome=status, completed_epochs=1)
    path = run_dir / "weights" / "model.onnx"
    path.parent.mkdir()
    path.write_bytes(b"fake ONNX")
    return path


def test_discovery_selects_latest_eligible_and_breaks_ties_by_path(tmp_path) -> None:
    """
    Ignore newer ineligible training and use nanosecond times with a path tie-break.
    """
    first = _saved_export(tmp_path, "a")
    last = _saved_export(tmp_path, "z")
    smoke = _saved_export(tmp_path, "smoke", smoke=True)
    limited = _saved_export(tmp_path, "limited", settings={"train_batch_limit": 1})
    fraction = _saved_export(tmp_path, "fraction", settings={"fraction": 0.5})
    failed = _saved_export(tmp_path, "failed", status="interrupted")
    for path in (first, last):
        os.utime(path, ns=(100, 100))
    for path in (smoke, limited, fraction, failed):
        os.utime(path, ns=(200, 200))
    models, excluded = video.discover_video_models(tmp_path, DECLARATIONS)
    assert models[0].spec.path == last
    assert len(excluded) == 4
    os.utime(first, ns=(101, 101))
    models, _ = video.discover_video_models(tmp_path, DECLARATIONS)
    assert models[0].spec.path == first


@pytest.mark.parametrize(
    "problem",
    [
        "missing_protocol",
        "invalid_protocol",
        "missing_training",
        "invalid_training",
        "wrong_model",
        "wrong_dataset",
        "missing_mapping",
        "invalid_mapping",
    ],
)
def test_discovery_reports_invalid_metadata(tmp_path, problem) -> None:
    """
    Consolidate missing models and candidate rejection reasons without dataset files.
    """
    path = _saved_export(tmp_path)
    run_dir = path.parent.parent
    protocol_path = run_dir / "run_protocol.json"
    training_path = run_dir / "training.json"
    if problem == "missing_protocol":
        protocol_path.unlink()
        # A protocol above RUNS_DIR must never be used for this candidate.
        write_json(tmp_path.parent / "run_protocol.json", {})
    elif problem == "invalid_protocol":
        protocol_path.write_text("{")
    elif problem == "missing_training":
        training_path.unlink()
    elif problem == "invalid_training":
        write_json(training_path, {})
    else:
        protocol = read_json(protocol_path)
        if problem == "wrong_model":
            protocol["model"] = "other"
        elif problem == "wrong_dataset":
            protocol["logical_dataset"] = "other"
        else:
            canonical = protocol["dataset_identity"]["canonical"]["canonical_source"]
            if problem == "missing_mapping":
                canonical.clear()
            else:
                canonical["category_mapping"][0]["name"] = "other"
        write_json(protocol_path, protocol)
        # Keep training provenance consistent to reach the mapping validation.
        from detection_common.utils.digest import json_digest

        training = read_json(training_path)
        training["run_protocol_sha256"] = json_digest(protocol)
        write_json(training_path, training)
    declarations = (
        *DECLARATIONS,
        ("yolo11n", ModelFamilies.YOLO11, "yolo11n*/**/*.onnx"),
    )
    with pytest.raises(FileNotFoundError) as error:
        video.discover_video_models(tmp_path, declarations)
    assert "yolox_tiny, yolo11n" in str(error.value)
    assert "Excluded candidates" in str(error.value)


def test_discovery_reads_saved_mapping_and_postprocessing(tmp_path) -> None:
    """
    Recover canonical IDs and model-owned NMS/limits with no training dataset.
    """
    path = _saved_export(tmp_path)
    run_dir = path.parent.parent
    (run_dir / "args.yaml").write_text("nms_threshold: 0.6\n")
    models, _ = video.discover_video_models(tmp_path, DECLARATIONS)
    assert models[0].category_id == 7
    assert models[0].spec.nms_threshold == 0.6
    write_json(
        run_dir / "test_predictions.json",
        {
            "metadata": {"postprocessing": {"nms_iou": 0.4, "max_det": 12}},
        },
    )
    models, _ = video.discover_video_models(tmp_path, DECLARATIONS)
    assert models[0].spec.nms_threshold == 0.4
    assert models[0].spec.max_detections == 12


def test_discovery_checks_rfdetr_manifest_and_shared_fingerprint(tmp_path) -> None:
    """
    Reject incompatible RF-DETR class mappings and different canonical sources.
    """
    _saved_export(tmp_path)
    path = _saved_export(tmp_path, name="rfdetr_small", fingerprint="c" * 64)
    declarations = (
        *DECLARATIONS,
        ("rfdetr_small", ModelFamilies.RFDETR, "rfdetr_small*/**/*.onnx"),
    )
    with pytest.raises(ValueError, match="different canonical"):
        video.discover_video_models(tmp_path, declarations)
    write_json(
        path.parent.parent / "dataset_manifest.json",
        {
            "category_mapping": {
                "class_names": ["basketball"],
                "prediction_to_source": {"0": 99},
            },
        },
    )
    with pytest.raises(FileNotFoundError, match="manifest disagrees"):
        video.discover_video_models(tmp_path, declarations)


class FakeSession:
    """
    Exercise the real YOLOX adapter with one detection followed by an empty frame.
    """

    def __init__(self) -> None:
        """
        Initialize synthetic graph metadata and call counters.
        """
        self.calls = 0
        self.contract_checks = 0
        self.fail_at = None
        self.providers = ["CPUExecutionProvider"]

    def get_inputs(self):
        """
        Count contract checks and expose a tiny static FP32 graph input.
        """
        self.contract_checks += 1
        return [
            SimpleNamespace(name="images", shape=[1, 3, 32, 32], type="tensor(float)")
        ]

    def get_outputs(self):
        """
        Return raw YOLOX output metadata for a single foreground class.
        """
        return [SimpleNamespace(name="output", shape=[1, 21, 6], type="tensor(float)")]

    def disable_fallback(self):
        """
        Support the runtime's explicit provider failure policy.
        """

    def get_providers(self):
        """
        Report the fake session's actual providers.
        """
        return self.providers

    def run(self, names, inputs):
        """
        Emit a source-pixel box on the first call and no boxes thereafter.
        """
        assert names is None
        # The blue BGR source frame must remain BGR in the YOLOX input tensor.
        assert inputs["images"][0, 0, 0, 0] == 255
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic inference failure")
        output = np.zeros((1, 21, 6), dtype=np.float32)
        if self.calls == 0:
            output[0, 0] = [1, 1, 0, 0, 0.9, 0.9]
        self.calls += 1
        return [output]


class FakeCapture:
    """
    Supply two blue BGR frames and advisory video metadata.
    """

    def __init__(self) -> None:
        """
        Initialize source geometry and resource state.
        """
        self.properties = {
            video.cv2.CAP_PROP_FRAME_WIDTH: 32,
            video.cv2.CAP_PROP_FRAME_HEIGHT: 32,
            video.cv2.CAP_PROP_FPS: 2,
            video.cv2.CAP_PROP_FRAME_COUNT: 2,
        }
        self.opened = True
        self.released = False
        self.index = 0
        self.frame_count = 2

    def isOpened(self):
        """
        Report whether input setup succeeded.
        """
        return self.opened

    def get(self, property_id):
        """
        Return configurable metadata for input validation.
        """
        return self.properties[property_id]

    def read(self):
        """
        Return every frame in order until EOF.
        """
        if self.index == self.frame_count:
            return False, None
        self.index += 1
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = 255
        return True, frame

    def release(self):
        """
        Mark the capture handle as closed.
        """
        self.released = True


class FakeWriter:
    """
    Count annotated frames while recording output closure.
    """

    def __init__(self) -> None:
        """
        Initialize the writer's resource and frame counters.
        """
        self.opened = True
        self.released = False
        self.frames = 0

    def isOpened(self):
        """
        Report whether the requested codec is available.
        """
        return self.opened

    def write(self, frame):
        """
        Count each correctly sized annotated frame, including empty detections.
        """
        assert frame.shape == (32, 32, 3)
        self.frames += 1

    def release(self):
        """
        Mark the output handle as closed.
        """
        self.released = True


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """
    Replace graph loading and video I/O while keeping real preprocessing/decoding.
    """
    _saved_export(tmp_path)
    models, _ = video.discover_video_models(tmp_path, DECLARATIONS)
    session, capture, writer = FakeSession(), FakeCapture(), FakeWriter()
    monkeypatch.setattr(video, "create_session", lambda *args: session)
    monkeypatch.setattr(video.onnx.checker, "check_model", lambda path: None)
    monkeypatch.setattr(video.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(video.cv2, "VideoWriter", lambda *args: writer)
    source = tmp_path / "input.webm"
    source.write_bytes(b"fake video")
    return SimpleNamespace(
        model=models[0],
        session=session,
        capture=capture,
        writer=writer,
        source=source,
        output=tmp_path / "output",
    )


def test_stream_retains_empty_frames_and_serializes_boxes(runtime) -> None:
    """
    Run once per frame with one cached contract and preserve the artifact fields.
    """
    summary = video.run_video_inference(
        runtime.model, runtime.source, runtime.output, CPU
    )
    records = [
        json.loads(line)
        for line in (runtime.output / "detections.jsonl").read_text().splitlines()
    ]
    assert [row["frame_index"] for row in records] == [0, 1]
    assert [row["timestamp_s"] for row in records] == [0.0, 0.5]
    assert records[1]["detections"] == []
    detection = records[0]["detections"][0]
    assert detection["bbox"] == [4.0, 4.0, 8.0, 8.0]
    assert (detection["center_x"], detection["center_y"]) == (8.0, 8.0)
    assert detection["category_id"] == 7
    assert detection["class_name"] == "basketball"
    assert detection["confidence"] == pytest.approx(0.81)
    assert runtime.session.calls == runtime.writer.frames == 2
    assert runtime.session.contract_checks == 1
    assert runtime.capture.released and runtime.writer.released
    assert summary["status"] == "completed"
    assert summary["processed_frames"] == 2
    assert (
        summary["frames_with_detections"] == summary["frames_without_detections"] == 1
    )
    assert summary["total_detections"] == 1
    assert summary["actual_providers"] == ["CPUExecutionProvider"]
    assert summary["processing_seconds"] > 0
    assert summary["mean_inference_ms"] > 0
    assert len(summary["onnx_sha256"]) == len(summary["video_sha256"]) == 64
    assert not Path(summary["onnx_path"]).is_absolute()
    assert read_json(runtime.output / "summary.json") == summary


def test_stream_records_partial_failure_and_closes_resources(runtime) -> None:
    """
    Retain completed records and report the exact failing zero-based frame.
    """
    runtime.session.fail_at = 1
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        video.run_video_inference(runtime.model, runtime.source, runtime.output, CPU)
    summary = read_json(runtime.output / "summary.json")
    assert summary["status"] == "partial"
    assert summary["processed_frames"] == runtime.writer.frames == 1
    assert summary["failing_frame"] == 1
    assert "synthetic inference failure" in summary["error"]
    assert runtime.capture.released and runtime.writer.released
    assert video.comparison_rows([summary]) == []


@pytest.mark.parametrize(
    "problem",
    [
        "missing_file",
        "capture",
        "width",
        "fps_zero",
        "fps_nan",
        "writer",
        "empty",
    ],
)
def test_stream_input_and_writer_failures_are_partial(runtime, problem) -> None:
    """
    Reject invalid input or unavailable MP4 output and release opened resources.
    """
    if problem == "missing_file":
        runtime.source.unlink()
    elif problem == "capture":
        runtime.capture.opened = False
    elif problem == "width":
        runtime.capture.properties[video.cv2.CAP_PROP_FRAME_WIDTH] = 0
    elif problem.startswith("fps"):
        runtime.capture.properties[video.cv2.CAP_PROP_FPS] = (
            float("nan") if problem == "fps_nan" else 0
        )
    elif problem == "writer":
        runtime.writer.opened = False
    elif problem == "empty":
        runtime.capture.frame_count = 0
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        video.run_video_inference(runtime.model, runtime.source, runtime.output, CPU)
    summary = read_json(runtime.output / "summary.json")
    assert summary["status"] == "partial"
    assert summary["processed_frames"] == 0
    if problem != "missing_file":
        assert runtime.capture.released
    if problem in {"writer", "empty"}:
        assert runtime.writer.released


def test_stream_flags_advisory_frame_count(runtime) -> None:
    """
    Process to EOF rather than stopping at an inaccurate reported frame count.
    """
    runtime.capture.properties[video.cv2.CAP_PROP_FRAME_COUNT] = 1
    summary = video.run_video_inference(
        runtime.model, runtime.source, runtime.output, CPU
    )
    assert summary["processed_frames"] == 2
    assert summary["warnings"] == ["Decoded 2 frames; source reported 1"]
    other = {**summary, "processed_frames": 3}
    with pytest.raises(ValueError, match="different numbers of frames"):
        video.comparison_rows([summary, other])


def test_stream_rejects_silent_provider_fallback(runtime) -> None:
    """
    Record CPU-only fallback as an error when CoreML was explicitly requested.
    """
    mode = ProviderMode(
        "coreml_first", ("CoreMLExecutionProvider", "CPUExecutionProvider")
    )
    with pytest.raises(RuntimeError, match="missing a requested execution provider"):
        video.run_video_inference(runtime.model, runtime.source, runtime.output, mode)
    summary = read_json(runtime.output / "summary.json")
    assert summary["status"] == "partial"
    assert summary["actual_providers"] == ["CPUExecutionProvider"]


def test_stream_graph_failure_does_not_choose_an_older_export(
    runtime, monkeypatch
) -> None:
    """
    Keep the selected graph failure visible in its partial summary.
    """

    def reject_graph(path):
        """
        Stand in for onnx.checker rejecting the selected export.
        """
        raise ValueError("incompatible selected graph")

    monkeypatch.setattr(video.onnx.checker, "check_model", reject_graph)
    with pytest.raises(ValueError, match="incompatible selected graph"):
        video.run_video_inference(runtime.model, runtime.source, runtime.output, CPU)
    summary = read_json(runtime.output / "summary.json")
    assert summary["status"] == "partial"
    assert summary["onnx_path"] == video.portable_path(runtime.model.spec.path)
    assert runtime.session.calls == 0
