"""
Verify framework-neutral discovery and reporting of published comparison bundles.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from detection_common.utils.json_io import read_json, write_json
from detection_evaluation import (
    capture_training_hardware,
    create_run_protocol,
    discover_bundles,
    finalize_training_attempt,
    publish_bundle,
    read_training_record,
    start_training_attempt,
    write_bundle_comparisons,
    write_prediction_artifact,
)

MODELS = ("yolo11n", "yolox_tiny", "yolox_nano", "rfdetr_small")


def _annotations(root: Path, split: str) -> Path:
    """
    Write a compact basketball COCO split with one positive and one negative image.
    """
    path = root / f"instances_{split}.json"
    write_json(
        path,
        {
            "images": [
                {"id": 1, "file_name": "positive.png", "width": 64, "height": 64},
                {"id": 2, "file_name": "negative.png", "width": 64, "height": 64},
            ],
            "categories": [{"id": 7, "name": "basketball"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 7,
                    "bbox": [2, 3, 10, 12],
                    "area": 120,
                    "iscrowd": 0,
                }
            ],
        },
    )
    return path


def _benchmark(median: float, *, host: str = "renku-test-host") -> dict:
    """
    Return a valid comparable batch-one benchmark record with a known median.
    """
    return {
        "protocol": "pil-to-cpu-coco-predictions-v1",
        "batch_size": 1,
        "precision": "float32",
        "warmup": 5,
        "samples": 2,
        "end_to_end_ms_mean": median + 1,
        "end_to_end_ms_median": median,
        "device": "cuda:0",
        "host": host,
        "torch": "2.9.1",
        "cuda_runtime": "12.8",
        "image_sha256": ["a" * 64, "b" * 64],
        "gpu": "NVIDIA Test GPU",
        "includes": "resize/normalize, transfer, forward, native postprocess, CPU boxes",
        "excludes": "file I/O, drawing, model loading",
    }


def _publish_bundle(
    tmp_path: Path,
    output_root: Path,
    annotations: dict[str, Path],
    *,
    model: str,
    run_name: str,
    hour: int,
    source_fingerprint: str = "a" * 64,
    smoke_run: bool = False,
    train_batch_limit: int | None = None,
    predictions: list[dict] | None = None,
    median: float = 10.0,
    host: str = "renku-test-host",
    incomplete_timing: bool = False,
) -> Path:
    """
    Publish one full Stage 3 fixture bundle with deterministic provenance and timing.
    """
    run_dir = tmp_path / "runs" / model / run_name
    settings = {"epochs": 10, "batch_size": 4, "train_batch_limit": train_batch_limit}
    create_run_protocol(
        run_dir,
        logical_dataset="basketball",
        model=model,
        source_notebook=f"{model}.ipynb",
        original_utc=datetime(2026, 9, 19, hour, tzinfo=UTC),
        training_settings=settings,
        smoke_run=smoke_run,
        canonical_dataset_identity={"source_fingerprint": source_fingerprint},
        loader_dataset_identity={
            "source_fingerprint": source_fingerprint,
            "loader_fingerprint": (str(MODELS.index(model) + 1) * 64),
        },
    )
    attempt = start_training_attempt(
        run_dir,
        resumed=False,
        training_hardware=capture_training_hardware(device="cuda:0"),
        monotonic_clock=lambda: 100.0,
    )
    finalize_training_attempt(
        run_dir,
        attempt,
        outcome="completed",
        completed_epochs=3,
        monotonic_clock=lambda: 700.0,
    )
    if incomplete_timing:
        record = read_training_record(run_dir)
        record["attempts"][0].update(status="incomplete", duration_seconds=None)
        record["summary"].update(
            status="incomplete", timing_available=False, total_seconds=None,
            total_hours=None, amortized_seconds_per_completed_epoch=None,
        )
        write_json(run_dir / "training.json", record)
        resumed = start_training_attempt(
            run_dir, resumed=True, training_hardware={"device": "cpu"},
            monotonic_clock=lambda: 1.0,
        )
        finalize_training_attempt(
            run_dir, resumed, outcome="completed", completed_epochs=1,
            monotonic_clock=lambda: 2.0,
        )
    checkpoint = run_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"{model}-{run_name}".encode())
    common_metadata = {
        "model": model,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "resolution": 640,
        "smoke_run": smoke_run,
        "parameters": 1000,
        "epochs_completed": 3,
        "postprocessing": {"score_floor": 0.001},
        "benchmark": _benchmark(median, host=host),
    }
    generated = (
        predictions
        if predictions is not None
        else [
            {
                "image_id": 1,
                "category_id": 7,
                "bbox": [2, 3, 10, 12],
                "score": 0.9,
            }
        ]
    )
    paths = {
        split: write_prediction_artifact(
            run_dir / "evaluation" / f"{split}_predictions.json",
            annotation_path,
            generated,
            metadata={**common_metadata, "split": split},
        )
        for split, annotation_path in annotations.items()
    }
    bundle_dir = output_root / "evaluation" / "basketball_large_dataset" / model / run_name
    publish_bundle(
        bundle_dir,
        run_dir=run_dir,
        val_predictions=paths["val"],
        test_predictions=paths["test"],
        selected_checkpoint=checkpoint,
        resolution=640,
        parameter_count=1000,
    )
    return bundle_dir


@pytest.fixture
def comparison_inputs(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    """
    Materialize evaluation roots and matching validation/test annotations.
    """
    annotations = {split: _annotations(tmp_path, split) for split in ("val", "test")}
    return tmp_path / "outputs", annotations


def test_discovery_orders_explicitly_selects_and_writes_full_reports(
    tmp_path: Path,
    comparison_inputs: tuple[Path, dict[str, Path]],
) -> None:
    """
    Select four full bundles by immutable time and report known metrics and timing.
    """
    output_root, annotations = comparison_inputs
    _publish_bundle(
        tmp_path,
        output_root,
        annotations,
        model="yolo11n",
        run_name="older",
        hour=9,
        median=40,
    )
    latest = _publish_bundle(
        tmp_path,
        output_root,
        annotations,
        model="yolo11n",
        run_name="latest",
        hour=10,
        median=30,
    )
    for index, model in enumerate(MODELS[1:], start=1):
        _publish_bundle(
            tmp_path,
            output_root,
            annotations,
            model=model,
            run_name=f"run-{model}",
            hour=10 + index,
            predictions=[] if model == "yolox_nano" else None,
            median=10.0 * index,
        )
    selections = discover_bundles(output_root)
    assert selections["yolo11n"].bundle_dir == latest
    reports = write_bundle_comparisons(
        selections,
        annotations,
        output_root / "comparisons" / "basketball_large_dataset",
    )
    test = reports["test"].set_index("model")
    assert test.loc["yolo11n", "ap50_95"] == pytest.approx(1)
    assert test.loc["yolox_nano", "ap50_95"] == pytest.approx(0)
    assert test.loc["rfdetr_small", "latency_ms_median"] == 30
    assert test.loc["yolo11n", "training_total_hours"] == pytest.approx(600 / 3600)
    report_dir = output_root / "comparisons" / "basketball_large_dataset"
    for split in ("val", "test"):
        for name in ("metrics", "inference", "training"):
            for suffix in ("csv", "json", "md"):
                assert (report_dir / f"{name}_{split}.{suffix}").is_file()
    selection = read_json(report_dir / "selection.json")["models"]["yolo11n"]
    assert selection["bundle_dir"] == str(latest)
    assert selection["protocol"]["training_settings"]["epochs"] == 10
    assert selection["training"]["attempts"][0]["timing_protocol"] == "monotonic-synchronized-v1"
    explicit = discover_bundles(output_root, selections={"yolo11n": "older"})
    assert explicit["yolo11n"].bundle_dir.name == "older"


def test_discovery_marks_missing_and_skips_smoke_or_truncated_candidates(
    tmp_path: Path,
    comparison_inputs: tuple[Path, dict[str, Path]],
) -> None:
    """
    Retain missing models as recoverable while excluding automatic non-full candidates.
    """
    output_root, annotations = comparison_inputs
    smoke = _publish_bundle(
        tmp_path,
        output_root,
        annotations,
        model="yolo11n",
        run_name="smoke",
        hour=10,
        smoke_run=True,
    )
    _publish_bundle(
        tmp_path,
        output_root,
        annotations,
        model="yolox_tiny",
        run_name="truncated",
        hour=10,
        train_batch_limit=1,
    )
    selections = discover_bundles(output_root, models=("yolo11n", "yolox_tiny"))
    assert not selections["yolo11n"].available
    assert "recover" in selections["yolo11n"].unavailable_reason
    assert not selections["yolox_tiny"].available
    with pytest.raises(ValueError, match="Smoke"):
        discover_bundles(output_root, selections={"yolo11n": smoke})
    with pytest.raises(FileNotFoundError, match="explicitly selected"):
        discover_bundles(output_root, selections={"rfdetr_small": "missing"})


def test_discovery_rejects_invalid_incomplete_and_mixed_source_bundles(
    tmp_path: Path,
    comparison_inputs: tuple[Path, dict[str, Path]],
) -> None:
    """
    Refuse malformed published state and incompatible canonical source identities.
    """
    output_root, annotations = comparison_inputs
    invalid = output_root / "evaluation" / "basketball_large_dataset" / "yolo11n" / "bad"
    invalid.mkdir(parents=True)
    write_json(invalid / "manifest.json", {})
    with pytest.raises(ValueError, match="Unsupported bundle manifest"):
        discover_bundles(output_root, models=("yolo11n",))
    (invalid / "manifest.json").unlink()
    _publish_bundle(
        tmp_path,
        output_root,
        annotations,
        model="yolo11n",
        run_name="one",
        hour=10,
    )
    _publish_bundle(
        tmp_path,
        output_root,
        annotations,
        model="yolox_tiny",
        run_name="two",
        hour=11,
        source_fingerprint="b" * 64,
    )
    with pytest.raises(ValueError, match="different canonical sources"):
        discover_bundles(output_root, models=("yolo11n", "yolox_tiny"))
    run_dir = tmp_path / "incomplete"
    create_run_protocol(
        run_dir,
        logical_dataset="basketball",
        model="rfdetr_small",
        source_notebook="rfdetr_small.ipynb",
        original_utc=datetime(2026, 9, 19, 12, tzinfo=UTC),
        training_settings={"epochs": 10},
        smoke_run=False,
        canonical_dataset_identity={"source_fingerprint": "c" * 64},
        loader_dataset_identity={"source_fingerprint": "c" * 64, "loader_fingerprint": "d" * 64},
    )
    start_training_attempt(
        run_dir,
        resumed=False,
        training_hardware=capture_training_hardware(device="cuda:0"),
        monotonic_clock=lambda: 1.0,
    )
    checkpoint = run_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"incomplete")
    artifacts = {
        split: write_prediction_artifact(
            run_dir / f"{split}_predictions.json",
            annotation_path,
            [],
            metadata={
                "model": "rfdetr_small",
                "split": split,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "resolution": 640,
                "smoke_run": False,
                "postprocessing": {"score_floor": 0.001},
            },
        )
        for split, annotation_path in annotations.items()
    }
    with pytest.raises(ValueError, match="unfinished training attempt"):
        publish_bundle(
            output_root / "incomplete",
            run_dir=run_dir,
            val_predictions=artifacts["val"],
            test_predictions=artifacts["test"],
            selected_checkpoint=checkpoint,
            resolution=640,
            parameter_count=1,
        )


def test_selected_snapshot_survives_republication_and_rejects_mixed_benchmarks(
    tmp_path: Path,
    comparison_inputs: tuple[Path, dict[str, Path]],
) -> None:
    """
    Keep an already-read snapshot stable and reject incompatible latency records.
    """
    output_root, annotations = comparison_inputs
    first = _publish_bundle(
        tmp_path,
        output_root,
        annotations,
        model="yolo11n",
        run_name="one",
        hour=10,
    )
    _publish_bundle(
        tmp_path,
        output_root,
        annotations,
        model="yolox_tiny",
        run_name="two",
        hour=11,
        host="another-host",
    )
    selected = discover_bundles(output_root, models=("yolo11n",))
    original_artifact = selected["yolo11n"].artifact_path("test")
    assert original_artifact is not None and original_artifact.is_file()
    manifest = read_json(first / "manifest.json")
    assert original_artifact.parts[-2] == manifest["generation"]
    write_bundle_comparisons(selected, annotations, tmp_path / "snapshot-reports")
    all_models = discover_bundles(output_root, models=("yolo11n", "yolox_tiny"))
    with pytest.raises(ValueError, match="Incomparable benchmarks"):
        write_bundle_comparisons(all_models, annotations, tmp_path / "mixed-reports")


def test_consumer_imports_no_producer_modules_or_subprocesses(
    tmp_path: Path,
    comparison_inputs: tuple[Path, dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Confirm discovery needs neither model-framework imports nor producer processes.
    """
    output_root, annotations = comparison_inputs
    _publish_bundle(
        tmp_path,
        output_root,
        annotations,
        model="yolo11n",
        run_name="one",
        hour=10,
    )
    before = set(sys.modules)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("consumer must not start subprocesses"),
    )
    discover_bundles(output_root, models=("yolo11n",))
    added = set(sys.modules) - before
    assert not any(name.startswith(("ultralytics", "yolox", "rfdetr")) for name in added)

def test_completed_recovery_reports_unknown_training_time(
    tmp_path: Path, comparison_inputs: tuple[Path, dict[str, Path]],
) -> None:
    """
    Compare completed predictions after a crash while keeping run timing unavailable.
    """
    import pandas as pd

    output_root, annotations = comparison_inputs
    _publish_bundle(
        tmp_path, output_root, annotations, model="yolo11n", run_name="resumed",
        hour=10, incomplete_timing=True,
    )
    selections = discover_bundles(output_root, models=("yolo11n",))
    assert selections["yolo11n"].available
    reports = write_bundle_comparisons(
        selections, annotations, output_root / "comparison",
    )
    result = reports["test"].set_index("model").loc["yolo11n"]
    assert result["ap50_95"] == pytest.approx(1)
    assert pd.isna(result["training_total_hours"])
    assert result["training_completed_epochs"] == 4
    assert result["training_status"] == "incomplete"
