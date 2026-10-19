"""
Test framework-neutral ONNX preprocessing, decoding, selection, and timing.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from detection_common.utils.json_io import read_json, write_json
from PIL import Image

from detection_evaluation.onnx_inference import (
    Detection,
    ImageTransform,
    ModelFamilies,
    OnnxModelSpec,
    benchmark_session,
    decode_rfdetr,
    decode_yolo11,
    decode_yolox,
    match_detections,
    preprocess_rfdetr,
    preprocess_yolo11,
    preprocess_yolox,
    provider_modes,
    select_next_test_image,
    validate_session_contract,
    xyxy_nms,
)


class FakeSession:
    """
    Provide the small ONNX Runtime interface exercised by the helpers.
    """

    def __init__(
        self,
        input_name: str,
        input_shape: list[int],
        outputs: list[tuple[str, list[int], np.ndarray]],
        *,
        input_type: str = "tensor(float)",
    ) -> None:
        """
        Store synthetic metadata and output arrays.
        """
        self._inputs = [
            SimpleNamespace(name=input_name, shape=input_shape, type=input_type)
        ]
        self._outputs = [
            SimpleNamespace(name=name, shape=shape, type="tensor(float)")
            for name, shape, _ in outputs
        ]
        self._arrays = [array for _, _, array in outputs]
        self.calls = 0

    def get_inputs(self) -> list[SimpleNamespace]:
        """
        Return synthetic graph input metadata.
        """
        return self._inputs

    def get_outputs(self) -> list[SimpleNamespace]:
        """
        Return synthetic graph output metadata.
        """
        return self._outputs

    def run(self, output_names, inputs) -> list[np.ndarray]:
        """
        Record one inference call and return stable arrays.
        """
        assert output_names is None
        assert set(inputs) == {self._inputs[0].name}
        self.calls += 1
        return [array.copy() for array in self._arrays]


def _annotation_file(path: Path) -> Path:
    """
    Write a tiny three-image COCO split for selection tests.
    """
    write_json(
        path,
        {
            "images": [
                {"id": image_id, "file_name": f"{image_id}.jpg"}
                for image_id in (1, 2, 3)
            ],
            "annotations": [
                {
                    "id": 10,
                    "image_id": 2,
                    "category_id": 7,
                    "bbox": [1, 2, 3, 4],
                }
            ],
            "categories": [{"id": 7, "name": "basketball"}],
        },
    )
    return path


def _square_transform() -> ImageTransform:
    """
    Return identity geometry for compact decoder tests.
    """
    return ImageTransform(
        tensor=np.zeros((1, 3, 32, 32), dtype=np.float32),
        original_width=32,
        original_height=32,
        input_width=32,
        input_height=32,
        scale=1.0,
    )


def test_selection_uses_every_image_before_repeating(tmp_path: Path) -> None:
    """
    Persist a shuffled queue and avoid a boundary repeat after a new cycle.
    """
    annotation_path = _annotation_file(tmp_path / "instances_test.json")
    state_path = tmp_path / "state.json"
    selected = [
        select_next_test_image(
            annotation_path,
            state_path,
            random_source=random.Random(42),
        ).image["id"]
        for _ in range(4)
    ]

    assert len(set(selected[:3])) == 3
    assert selected[3] != selected[2]
    state = read_json(state_path)
    assert state["last_image_id"] == selected[-1]
    assert selected[-1] not in state["remaining_image_ids"]


def test_selection_rebuilds_stale_state_and_resolves_annotations(
    tmp_path: Path,
) -> None:
    """
    Ignore incompatible persisted IDs and return annotations for the new image.
    """
    annotation_path = _annotation_file(tmp_path / "instances_test.json")
    state_path = tmp_path / "state.json"
    write_json(
        state_path,
        {
            "schema_version": 1,
            "annotation_sha256": "0" * 64,
            "last_image_id": 999,
            "remaining_image_ids": [999],
        },
    )

    selected = select_next_test_image(
        annotation_path,
        state_path,
        random_source=random.Random(1),
    )

    assert selected.image["id"] in {1, 2, 3}
    assert all(
        annotation["image_id"] == selected.image["id"]
        for annotation in selected.annotations
    )


def test_preprocessors_return_contiguous_fp32_nchw() -> None:
    """
    Preserve each family-specific numeric range and spatial transform.
    """
    image = Image.fromarray(np.full((10, 20, 3), 255, dtype=np.uint8))

    yolox = preprocess_yolox(image, 32, 32)
    yolo11 = preprocess_yolo11(image, 32, 32)
    rfdetr = preprocess_rfdetr(image, 32, 32)

    for transformed in (yolox, yolo11, rfdetr):
        assert transformed.tensor.shape == (1, 3, 32, 32)
        assert transformed.tensor.dtype == np.float32
        assert transformed.tensor.flags.c_contiguous
    assert yolox.pad_top == 0
    assert np.all(yolox.tensor[0, :, 20:, :] == 114)
    assert yolo11.pad_top == 8
    assert 0 <= float(yolo11.tensor.min()) <= float(yolo11.tensor.max()) <= 1
    assert float(rfdetr.tensor.max()) > 1


def test_yolox_decoder_restores_known_grid_box() -> None:
    """
    Decode one stride-eight prediction into source-image COCO coordinates.
    """
    output = np.zeros((1, 21, 6), dtype=np.float32)
    output[0, 5, 4:] = (0.9, 0.8)

    detections = decode_yolox(
        output,
        _square_transform(),
        [7],
        score_threshold=0.25,
        nms_threshold=0.65,
        max_detections=300,
    )

    assert len(detections) == 1
    assert detections[0].category_id == 7
    assert detections[0].bbox == pytest.approx((4, 4, 8, 8))
    assert detections[0].score == pytest.approx(0.72)


def test_yolo11_decoder_reverses_centered_letterbox() -> None:
    """
    Decode one exported YOLO11 candidate and remove centered padding.
    """
    output = np.zeros((1, 5, 21), dtype=np.float32)
    output[0, :4, 0] = (16, 16, 8, 8)
    output[0, 4, 0] = 0.9
    transform = ImageTransform(
        tensor=np.zeros((1, 3, 32, 32), dtype=np.float32),
        original_width=16,
        original_height=8,
        input_width=32,
        input_height=32,
        scale=2.0,
        pad_top=8,
    )

    detections = decode_yolo11(
        output,
        transform,
        [7],
        score_threshold=0.25,
        nms_threshold=0.7,
        max_detections=300,
    )

    assert len(detections) == 1
    assert detections[0].bbox == pytest.approx((6, 2, 4, 4))


def test_rfdetr_decoder_removes_background_and_scales_box() -> None:
    """
    Select foreground sigmoid logits and scale normalized boxes directly.
    """
    boxes = np.array([[[0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.1, 0.1]]])
    logits = np.array([[[3.0, -3.0], [-10.0, 10.0]]])

    detections = decode_rfdetr(
        boxes,
        logits,
        _square_transform(),
        [7],
        score_threshold=0.25,
        max_detections=300,
    )

    assert len(detections) == 1
    assert detections[0].bbox == pytest.approx((8, 8, 16, 16))
    assert detections[0].score == pytest.approx(0.952574, rel=1e-5)


def test_nms_keeps_disjoint_boxes_and_suppresses_duplicate() -> None:
    """
    Prefer the highest-scoring member of an overlapping pair.
    """
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]])
    scores = np.array([0.9, 0.8, 0.7])

    assert xyxy_nms(boxes, scores, 0.5) == [0, 2]


def test_matching_scores_duplicates_misses_and_negative_images() -> None:
    """
    Use score-ordered one-to-one IoU matching for selected-image diagnostics.
    """
    ground_truth = [{"category_id": 7, "bbox": [0, 0, 10, 10]}]
    predictions = [
        Detection(7, (0, 0, 10, 10), 0.9),
        Detection(7, (0, 0, 10, 10), 0.8),
    ]

    metrics = match_detections(ground_truth, predictions)
    negative = match_detections([], predictions[:1])

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert negative["precision"] == 0
    assert negative["recall"] == 0


def test_validate_session_contract_accepts_and_rejects_shapes() -> None:
    """
    Require the known fixed YOLO11 graph contract and FP32 input.
    """
    raw = np.zeros((1, 5, 21), dtype=np.float32)
    session = FakeSession("images", [1, 3, 32, 32], [("output0", [1, 5, 21], raw)])

    contract = validate_session_contract(session, ModelFamilies.YOLO11)
    assert contract.input_name == "images"
    assert contract.output_names == ("output0",)

    invalid = FakeSession(
        "images",
        [1, 3, 32, 32],
        [("output0", [1, 5, 21], raw)],
        input_type="tensor(float16)",
    )
    with pytest.raises(ValueError, match="FP32"):
        validate_session_contract(invalid, ModelFamilies.YOLO11)


def test_provider_modes_are_explicit_and_exclude_azure(tmp_path: Path) -> None:
    """
    Declare CPU reference and CoreML-first fallback with a writable cache.
    """
    cpu, coreml = provider_modes(
        ["CoreMLExecutionProvider", "AzureExecutionProvider", "CPUExecutionProvider"],
        tmp_path,
        "yolo11n",
    )

    assert cpu.providers == ("CPUExecutionProvider",)
    assert [
        provider[0] if isinstance(provider, tuple) else provider
        for provider in coreml.providers
    ] == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    options = coreml.providers[0][1]
    assert options["ModelFormat"] == "MLProgram"
    assert Path(options["ModelCacheDirectory"]).is_dir()


def test_provider_modes_report_missing_coreml(tmp_path: Path) -> None:
    """
    Keep the CoreML comparison row declarable when the provider is unavailable.
    """
    _, coreml = provider_modes(["CPUExecutionProvider"], tmp_path, "model")

    assert not coreml.available
    assert "unavailable" in str(coreml.unavailable_reason)


def test_benchmark_counts_calls_and_returns_latency() -> None:
    """
    Execute correctness, warm-up, and timed end-to-end calls serially.
    """
    raw = np.zeros((1, 5, 21), dtype=np.float32)
    raw[0, :4, 0] = (16, 16, 8, 8)
    raw[0, 4, 0] = 0.9
    session = FakeSession("images", [1, 3, 32, 32], [("output0", [1, 5, 21], raw)])
    spec = OnnxModelSpec(
        "yolo11n",
        ModelFamilies.YOLO11,
        Path("model.onnx"),
        nms_threshold=0.7,
    )

    result = benchmark_session(
        session,
        spec,
        Image.new("RGB", (32, 32)),
        [7],
        warmup=2,
        samples=3,
    )

    assert session.calls == 6
    assert len(result.predictions) == 1
    assert result.latency_ms >= 0


@pytest.mark.parametrize(
    ("family", "outputs"),
    [
        (
            ModelFamilies.YOLOX,
            [("output", [1, 21, 6], np.zeros((1, 21, 6)))],
        ),
        (
            ModelFamilies.RFDETR,
            [
                ("dets", [1, 300, 4], np.zeros((1, 300, 4))),
                ("labels", [1, 300, 2], np.zeros((1, 300, 2))),
            ],
        ),
    ],
)
def test_other_graph_contracts(family: ModelFamilies, outputs: list[tuple]) -> None:
    """
    Accept the current YOLOX and RF-DETR export layouts.
    """
    input_name = "images" if family is ModelFamilies.YOLOX else "input"
    session = FakeSession(input_name, [1, 3, 32, 32], outputs)

    assert validate_session_contract(session, family).input_name == input_name
