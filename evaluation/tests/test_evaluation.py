"""
Verify shared detector metrics and portable artifact provenance on small COCO fixtures.
"""

from pathlib import Path

import pytest
from detection_common.utils.json_io import read_json, write_json

from detection_evaluation import (
    evaluate_predictions,
    file_sha256,
    read_prediction_artifact,
    write_comparison,
    write_prediction_artifact,
)


@pytest.fixture
def annotations(tmp_path: Path) -> Path:
    """
    Create two positive images and a negative with non-contiguous image/category IDs.
    """
    path = tmp_path / "instances_test.json"
    write_json(
        path,
        {
            "images": [
                {"id": index, "file_name": f"{index}.png", "width": 64, "height": 64}
                for index in (10, 20, 30)
            ],
            "categories": [{"id": 7, "name": "basketball"}],
            "annotations": [
                {
                    "id": index,
                    "image_id": index,
                    "category_id": 7,
                    "bbox": [2, 3, 10, 12],
                    "area": 120,
                    "iscrowd": 0,
                }
                for index in (10, 20)
            ],
        },
    )
    return path


def prediction(image_id: int = 10, score: float = 0.9) -> dict:
    """
    Predict the fixture's basketball box in original-image COCO coordinates.
    """
    return {
        "image_id": image_id,
        "category_id": 7,
        "bbox": [2, 3, 10, 12],
        "score": score,
    }


@pytest.fixture
def metadata(tmp_path: Path) -> dict:
    """
    Describe a selected checkpoint without importing any detector framework.
    """
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"local test weights")
    return {
        "model": "rfdetr_small",
        "split": "test",
        "run_dir": str(tmp_path),
        "checkpoint": str(checkpoint),
        "resolution": 640,
        "smoke_run": False,
        "postprocessing": {"score_floor": 0.001},
    }


def benchmark(*, median: float = 10.0, **overrides: object) -> dict:
    """
    Describe a comparable batch-one FP32 benchmark over two fixture images.
    """
    record = {
        "protocol": "pil-to-cpu-coco-predictions-v1",
        "batch_size": 1,
        "precision": "float32",
        "warmup": 5,
        "samples": 2,
        "end_to_end_ms_mean": median + 1.0,
        "end_to_end_ms_median": median,
        "device": "cuda:0",
        "host": "renku-test-host",
        "torch": "2.9.1",
        "cuda_runtime": "12.8",
        "image_sha256": ["a" * 64, "b" * 64],
        "gpu": "NVIDIA Test GPU",
        "includes": (
            "resize/normalize, transfer, forward, native postprocess, CPU boxes"
        ),
        "excludes": "file I/O, drawing, model loading",
    }
    record.update(overrides)
    return record


def test_perfect_detections_include_negative_image(annotations: Path) -> None:
    """
    Perfect detections score one while a no-prediction negative stays in the split.
    """
    metrics = evaluate_predictions(annotations, [prediction(10), prediction(20)])
    for metric in ("ap50", "ap50_95", "ar100", "precision", "recall", "f1"):
        assert metrics[metric] == pytest.approx(1)
    assert metrics["image_count"] == 3
    assert metrics["negative_image_count"] == 1
    assert metrics["false_positives_per_negative_image"] == 0
    assert metrics["negative_image_false_positive_fraction"] == 0
    assert metrics["evaluator"]["max_dets"] == [1, 10, 100]


def test_empty_predictions_are_zero_with_all_missed_boxes(annotations: Path) -> None:
    """
    Handle COCO's empty detection edge case without dropping any source images.
    """
    metrics = evaluate_predictions(annotations, [])
    for metric in ("ap50", "ap50_95", "ar100", "precision", "recall", "f1"):
        assert metrics[metric] == 0
    assert metrics["false_negatives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["image_count"] == 3


def test_duplicate_miss_and_negative_false_positive(annotations: Path) -> None:
    """
    Match a box once, count duplicate/negative detections, and retain the missed box.
    """
    metrics = evaluate_predictions(
        annotations, [prediction(10, 0.9), prediction(10, 0.8), prediction(30, 0.7)]
    )
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == pytest.approx(0.4)
    assert metrics["false_positives_per_negative_image"] == 1
    assert metrics["negative_image_false_positive_fraction"] == 1


def test_ap_floor_is_separate_from_operating_threshold(annotations: Path) -> None:
    """
    Low-score true positives enter AP but not fixed-threshold precision/recall.
    """
    metrics = evaluate_predictions(
        annotations,
        [prediction(10, 0.001), prediction(20, 0.1), prediction(30, 0.0009)],
    )
    assert metrics["ap50_95"] == pytest.approx(1)
    assert metrics["precision"] == metrics["recall"] == 0
    assert metrics["prediction_count"] == 2
    assert metrics["negative_false_positives"] == 0


def test_operating_threshold_is_inclusive_and_detections_are_capped(
    annotations: Path,
) -> None:
    """
    Apply the inclusive operating threshold and the common 100-box per-image cap.
    """
    predictions = [prediction(10, 0.25)] + [prediction(30, 0.3)] * 101
    metrics = evaluate_predictions(annotations, predictions)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 100
    assert metrics["prediction_count"] == 101
    assert metrics["false_positives_per_negative_image"] == 100


def test_all_negative_split_has_undefined_ap(annotations: Path) -> None:
    """
    Preserve undefined COCO AP as None while counting background false detections.
    """
    ground_truth = read_json(annotations)
    ground_truth["annotations"] = []
    write_json(annotations, ground_truth)
    metrics = evaluate_predictions(annotations, [prediction(10), prediction(10)])
    assert metrics["ap50"] is None
    assert metrics["ap50_95"] is None
    assert metrics["ar100"] is None
    assert metrics["negative_image_count"] == 3
    assert metrics["false_positives_per_negative_image"] == pytest.approx(2 / 3)
    assert metrics["negative_image_false_positive_fraction"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    "override",
    [
        {"image_id": 999},
        {"category_id": 0},
        {"bbox": [2, 3, -1, 2]},
        {"bbox": [2, 3, 4]},
        {"bbox": None},
        {"score": float("nan")},
        {"score": 1.1},
    ],
)
def test_invalid_predictions_fail_before_evaluation(
    annotations: Path,
    override: dict,
) -> None:
    """
    Reject unknown IDs and invalid geometry/confidence instead of scoring bad exports.
    """
    with pytest.raises(ValueError):
        evaluate_predictions(annotations, [{**prediction(), **override}])


def test_crowd_annotations_require_a_different_matching_policy(
    annotations: Path,
) -> None:
    """
    Avoid reporting misleading fixed-threshold counts for unsupported crowd boxes.
    """
    ground_truth = read_json(annotations)
    ground_truth["annotations"][0]["iscrowd"] = 1
    write_json(annotations, ground_truth)
    with pytest.raises(ValueError, match="crowd/ignore"):
        evaluate_predictions(annotations, [])


def test_artifact_round_trip_preserves_empty_image_coverage(
    annotations: Path,
    metadata: dict,
    tmp_path: Path,
) -> None:
    """
    Store original annotation/checkpoint identities and all images for portable reuse.
    """
    path = write_prediction_artifact(
        tmp_path / "export.json",
        annotations,
        [prediction()],
        metadata=metadata,
    )
    artifact = read_prediction_artifact(path, annotations)
    assert artifact["image_ids"] == [10, 20, 30]
    assert artifact["predictions"] == [prediction()]
    assert artifact["annotation_sha256"] == file_sha256(annotations)
    assert artifact["metadata"]["checkpoint_sha256"] == file_sha256(
        metadata["checkpoint"]
    )
    # Inference weights need not exist in the receiving environment.
    Path(metadata["checkpoint"]).unlink()
    assert read_prediction_artifact(path, annotations) == artifact


@pytest.mark.parametrize("change", ["hash", "coverage", "prediction"])
def test_artifact_rejects_changed_annotation_or_image_provenance(
    annotations: Path,
    metadata: dict,
    tmp_path: Path,
    change: str,
) -> None:
    """
    Reject changed ground truth, missing image coverage, and unknown prediction IDs.
    """
    path = write_prediction_artifact(
        tmp_path / "export.json",
        annotations,
        [],
        metadata=metadata,
    )
    artifact = read_json(path)
    if change == "hash":
        annotations.write_text(annotations.read_text() + "\n")
    elif change == "coverage":
        artifact["image_ids"] = [10, 20]
    else:
        artifact["predictions"] = [prediction(999)]
    write_json(path, artifact)
    with pytest.raises(ValueError):
        read_prediction_artifact(path, annotations)


def test_smoke_artifact_requires_explicit_inspection_override(
    annotations: Path,
    metadata: dict,
    tmp_path: Path,
) -> None:
    """
    Allow smoke inspection while excluding smoke results from full comparisons.
    """
    metadata["smoke_run"] = True
    path = write_prediction_artifact(
        tmp_path / "smoke.json",
        annotations,
        [],
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="Smoke"):
        read_prediction_artifact(path, annotations)
    assert read_prediction_artifact(path, annotations, allow_smoke=True)
    with pytest.raises(ValueError, match="Smoke"):
        write_comparison({"rfdetr_small": path}, annotations, tmp_path, split="test")


@pytest.mark.parametrize(
    "override",
    [
        {"smoke_run": "false"},
        {"resolution": 0},
        {"postprocessing": {}},
        {"split": "train"},
        {"checkpoint": ""},
    ],
)
def test_artifact_requires_unambiguous_metadata(
    annotations: Path,
    metadata: dict,
    tmp_path: Path,
    override: dict,
) -> None:
    """
    Reject missing or malformed run metadata before saving an export.
    """
    with pytest.raises(ValueError):
        write_prediction_artifact(
            tmp_path / "export.json",
            annotations,
            [],
            metadata={**metadata, **override},
        )


def test_artifact_requires_checkpoint_content_identity(
    annotations: Path,
    metadata: dict,
    tmp_path: Path,
) -> None:
    """
    Reject a missing checkpoint unless a portable export carries its valid digest.
    """
    Path(metadata["checkpoint"]).unlink()
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        write_prediction_artifact(
            tmp_path / "export.json", annotations, [], metadata=metadata
        )


def test_comparison_marks_missing_baselines_unavailable(
    annotations: Path,
    metadata: dict,
    tmp_path: Path,
) -> None:
    """
    Persist real shared metrics and explicit missing rows without invented timing ranks.
    """
    path = write_prediction_artifact(
        tmp_path / "export.json",
        annotations,
        [prediction(10), prediction(20)],
        metadata=metadata,
    )
    destination = tmp_path / "comparison"
    frame = write_comparison(
        {
            "yolo11n": None,
            "yolox_tiny": tmp_path / "missing.json",
            "rfdetr_small": path,
        },
        annotations,
        destination,
        split="test",
    )
    assert frame["status"].tolist() == ["unavailable", "unavailable", "available"]
    assert frame.iloc[2]["ap50_95"] == pytest.approx(1)
    report = read_json(destination / "comparison_test.json")
    assert report["unavailable"] == ["yolo11n", "yolox_tiny"]
    assert report["models"]["rfdetr_small"]["metrics"]["image_count"] == 3
    assert (destination / "comparison_test.csv").is_file()
    assert "0.001" in (destination / "comparison_test.md").read_text()
    assert frame["latency_ms_median"].isna().all()
    assert frame["batch_one_images_per_second"].isna().all()


def test_comparison_reports_compatible_latency_and_inverse_median_throughput(
    annotations: Path,
    metadata: dict,
    tmp_path: Path,
) -> None:
    """
    Compare compatible end-to-end latency records and derive batch-one images/s.
    """
    paths = {}
    for model, median in (("yolo11n", 20.0), ("rfdetr_small", 10.0)):
        timing = benchmark(
            median=median,
            device="cuda" if model == "rfdetr_small" else "cuda:0",
        )
        paths[model] = write_prediction_artifact(
            tmp_path / f"{model}.json",
            annotations,
            [prediction(10), prediction(20)],
            metadata={
                **metadata,
                "model": model,
                "benchmark": timing,
            },
        )
    destination = tmp_path / "comparison"
    frame = write_comparison(paths, annotations, destination, split="test")
    by_model = frame.set_index("model")
    assert by_model.loc["yolo11n", "latency_ms_median"] == 20.0
    assert by_model.loc["rfdetr_small", "latency_ms_mean"] == 11.0
    assert by_model.loc["yolo11n", "batch_one_images_per_second"] == 50.0
    assert by_model.loc["rfdetr_small", "batch_one_images_per_second"] == 100.0
    report = read_json(destination / "comparison_test.json")
    assert report["models"]["rfdetr_small"]["timing"] == {
        "benchmark_samples": 2,
        "latency_ms_median": 10.0,
        "latency_ms_mean": 11.0,
        "batch_one_images_per_second": 100.0,
    }
    markdown = (destination / "comparison_test.md").read_text()
    assert "Median latency (ms/image)" in markdown
    assert "| rfdetr_small | available |" in markdown
    assert "| 10.00 | 11.00 | 100.00 |" in markdown


@pytest.mark.parametrize(
    ("override", "difference"),
    [
        ({"host": "another-host"}, "host"),
        ({"gpu": "Another GPU"}, "gpu"),
        ({"image_sha256": ["a" * 64, "c" * 64]}, "image_sha256"),
        ({"precision": "float16"}, "batch-one FP32"),
    ],
)
def test_comparison_rejects_incompatible_benchmarks(
    annotations: Path,
    metadata: dict,
    tmp_path: Path,
    override: dict,
    difference: str,
) -> None:
    """
    Refuse latency comparisons produced with incompatible timing conditions.
    """
    reference = write_prediction_artifact(
        tmp_path / "reference.json",
        annotations,
        [],
        metadata={**metadata, "model": "yolo11n", "benchmark": benchmark()},
    )
    candidate_metadata = {
        **metadata,
        "model": "rfdetr_small",
        "benchmark": benchmark(**override),
    }
    if difference == "batch-one FP32":
        with pytest.raises(ValueError, match=difference):
            write_prediction_artifact(
                tmp_path / "candidate.json",
                annotations,
                [],
                metadata=candidate_metadata,
            )
        return
    candidate = write_prediction_artifact(
        tmp_path / "candidate.json",
        annotations,
        [],
        metadata=candidate_metadata,
    )
    with pytest.raises(ValueError, match=difference):
        write_comparison(
            {"yolo11n": reference, "rfdetr_small": candidate},
            annotations,
            tmp_path / "comparison",
            split="test",
        )


@pytest.mark.parametrize(
    "override",
    [
        {"model": "yolo11n"},
        {"split": "val"},
        {"postprocessing": {"score_floor": 0.25}},
    ],
)
def test_comparison_rejects_mismatched_models_splits_and_ap_floors(
    annotations: Path,
    metadata: dict,
    tmp_path: Path,
    override: dict,
) -> None:
    """
    Require the selected model/split and low-score export policy to match the report.
    """
    path = write_prediction_artifact(
        tmp_path / "export.json",
        annotations,
        [],
        metadata={**metadata, **override},
    )
    with pytest.raises(ValueError):
        write_comparison({"rfdetr_small": path}, annotations, tmp_path, split="test")
