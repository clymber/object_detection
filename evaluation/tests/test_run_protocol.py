"""
Verify framework-neutral run timing and immutable evaluation bundle publication.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from detection_common.utils.json_io import read_json, write_json

from detection_evaluation import (
    capture_training_hardware,
    create_run_protocol,
    finalize_training_attempt,
    publish_bundle,
    read_bundle,
    read_run_protocol,
    read_training_record,
    start_training_attempt,
    write_prediction_artifact,
)


class FakeClock:
    """
    Return deterministic monotonic readings supplied by a test.
    """

    def __init__(self, readings: list[float]) -> None:
        """
        Store the readings consumed at attempt timing boundaries.
        """
        self._readings = iter(readings)

    def __call__(self) -> float:
        """
        Return the next monotonic reading.
        """
        return next(self._readings)


@pytest.fixture
def identities() -> tuple[dict, dict]:
    """
    Return compatible, recorded canonical and loader identity payloads.
    """
    source_fingerprint = "a" * 64
    return (
        {
            "source_fingerprint": source_fingerprint,
            "canonical_source": {"schema_version": 1},
        },
        {
            "source_fingerprint": source_fingerprint,
            "loader_fingerprint": "b" * 64,
            "loader": {"schema_version": 1},
        },
    )


def create_protocol(
    run_dir: Path,
    identities: tuple[dict, dict],
    *,
    smoke_run: bool = False,
) -> dict:
    """
    Capture the immutable protocol shared by timing and bundle test cases.
    """
    canonical, loader = identities
    return create_run_protocol(
        run_dir,
        logical_dataset="basketball",
        model="yolo11n",
        source_notebook="nb02.02-ultralytics-yolo11n.ipynb",
        original_utc=datetime(2026, 9, 19, 12, tzinfo=UTC),
        training_settings={"epochs": 10, "batch": 8},
        smoke_run=smoke_run,
        canonical_dataset_identity=canonical,
        loader_dataset_identity=loader,
    )


def test_completed_and_interrupted_attempts_are_aggregated_once(
    tmp_path: Path, identities: tuple[dict, dict]
) -> None:
    """
    Include finalized interruption work in one synchronized timing summary.
    """
    run_dir = tmp_path / "run"
    create_protocol(run_dir, identities)
    clock = FakeClock([10, 15, 20, 23])
    synchronizations: list[str] = []
    hardware = capture_training_hardware(device="cuda:0", details={"gpu": "test"})
    first = start_training_attempt(
        run_dir,
        resumed=False,
        training_hardware=hardware,
        monotonic_clock=clock,
        synchronize=lambda: synchronizations.append("sync"),
    )
    finalize_training_attempt(
        run_dir,
        first,
        outcome="completed",
        completed_epochs=2,
        monotonic_clock=clock,
        synchronize=lambda: synchronizations.append("sync"),
    )
    second = start_training_attempt(
        run_dir,
        resumed=True,
        training_hardware=hardware,
        monotonic_clock=clock,
        synchronize=lambda: synchronizations.append("sync"),
    )
    summary = finalize_training_attempt(
        run_dir,
        second,
        outcome="interrupted",
        completed_epochs=1,
        monotonic_clock=clock,
        synchronize=lambda: synchronizations.append("sync"),
    )
    assert synchronizations == ["sync"] * 4
    assert summary == {
        "status": "interrupted",
        "timing_available": True,
        "total_seconds": 8.0,
        "total_hours": pytest.approx(8 / 3600),
        "completed_epochs": 3,
        "amortized_seconds_per_completed_epoch": pytest.approx(8 / 3),
    }
    with pytest.raises(ValueError, match="already"):
        finalize_training_attempt(
            run_dir,
            second,
            outcome="interrupted",
            completed_epochs=1,
            monotonic_clock=FakeClock([24]),
        )
    assert read_training_record(run_dir)["summary"] == summary


def test_unfinished_attempt_has_no_reconstructed_timing(
    tmp_path: Path, identities: tuple[dict, dict]
) -> None:
    """
    Treat a hard-terminated attempt as incomplete rather than guessing its duration.
    """
    run_dir = tmp_path / "run"
    create_protocol(run_dir, identities)
    start_training_attempt(
        run_dir,
        resumed=False,
        training_hardware=capture_training_hardware(device="cuda:0"),
        monotonic_clock=FakeClock([100]),
    )
    summary = read_training_record(run_dir)["summary"]
    assert summary["status"] == "incomplete"
    assert summary["timing_available"] is False
    assert summary["total_seconds"] is None
    assert summary["total_hours"] is None
    assert summary["amortized_seconds_per_completed_epoch"] is None
    with pytest.raises(ValueError, match="unfinished"):
        start_training_attempt(
            run_dir,
            resumed=True,
            training_hardware=capture_training_hardware(device="cuda:0"),
            monotonic_clock=FakeClock([101]),
        )


def test_run_provenance_is_immutable_including_smoke_identity(
    tmp_path: Path, identities: tuple[dict, dict]
) -> None:
    """
    Reject changed metadata when resume or recovery tries to recapture a run.
    """
    run_dir = tmp_path / "run"
    protocol = create_protocol(run_dir, identities, smoke_run=True)
    canonical, loader = identities
    with pytest.raises(ValueError, match="Immutable"):
        create_run_protocol(
            run_dir,
            logical_dataset="basketball",
            model="yolo11n",
            source_notebook="nb02.02-ultralytics-yolo11n.ipynb",
            original_utc="2026-09-19T12:00:00Z",
            training_settings={"epochs": 10, "batch": 8},
            smoke_run=False,
            canonical_dataset_identity=canonical,
            loader_dataset_identity=loader,
        )
    assert read_run_protocol(run_dir) == protocol


def _completed_run(
    tmp_path: Path,
    identities: tuple[dict, dict],
    *,
    smoke_run: bool = False,
) -> tuple[Path, Path, Path, Path]:
    """
    Materialize a completed run and schema-v1 prediction artifacts for a bundle.
    """
    run_dir = tmp_path / "run"
    create_protocol(run_dir, identities, smoke_run=smoke_run)
    attempt = start_training_attempt(
        run_dir,
        resumed=False,
        training_hardware=capture_training_hardware(device="cuda:0"),
        monotonic_clock=FakeClock([0]),
    )
    finalize_training_attempt(
        run_dir,
        attempt,
        outcome="completed",
        completed_epochs=2,
        monotonic_clock=FakeClock([4]),
    )
    checkpoint = run_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    annotations = tmp_path / "annotations.json"
    write_json(
        annotations,
        {
            "images": [{"id": 1, "file_name": "one.png"}],
            "categories": [{"id": 7, "name": "basketball"}],
            "annotations": [],
        },
    )
    common_metadata = {
        "model": "yolo11n",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "resolution": 640,
        "smoke_run": smoke_run,
        "postprocessing": {"score_floor": 0.001},
    }
    val = write_prediction_artifact(
        tmp_path / "val_predictions.json",
        annotations,
        [],
        metadata={**common_metadata, "split": "val"},
    )
    test = write_prediction_artifact(
        tmp_path / "test_predictions.json",
        annotations,
        [],
        metadata={**common_metadata, "split": "test"},
    )
    return run_dir, val, test, checkpoint


def test_bundle_failure_preserves_old_manifest_and_old_snapshot(
    tmp_path: Path, identities: tuple[dict, dict]
) -> None:
    """
    Keep a published generation readable when a later publication fails.
    """
    run_dir, val, test, checkpoint = _completed_run(tmp_path, identities)
    bundle_dir = tmp_path / "bundle"
    failed_first_bundle = tmp_path / "failed-first-bundle"

    def fail_after_staging(phase: str) -> None:
        """
        Stop a first publication before any generation is published.
        """
        if phase == "after_staging":
            raise RuntimeError("simulated publication failure")

    with pytest.raises(RuntimeError, match="simulated"):
        publish_bundle(
            failed_first_bundle,
            run_dir=run_dir,
            val_predictions=val,
            test_predictions=test,
            selected_checkpoint=checkpoint,
            resolution=640,
            parameter_count=1,
            failpoint=fail_after_staging,
        )
    assert not (failed_first_bundle / "manifest.json").exists()
    first = publish_bundle(
        bundle_dir,
        run_dir=run_dir,
        val_predictions=val,
        test_predictions=test,
        selected_checkpoint=checkpoint,
        resolution=640,
        parameter_count=1,
    )
    old_snapshot = read_bundle(bundle_dir)

    def fail_after_generation(phase: str) -> None:
        """
        Stop publication after its immutable generation exists.
        """
        if phase == "after_generation":
            raise RuntimeError("simulated publication failure")

    def fail_before_manifest(phase: str) -> None:
        """
        Stop publication immediately before replacing the manifest pointer.
        """
        if phase == "before_manifest_replace":
            raise RuntimeError("simulated publication failure")

    with pytest.raises(RuntimeError, match="simulated"):
        publish_bundle(
            bundle_dir,
            run_dir=run_dir,
            val_predictions=val,
            test_predictions=test,
            selected_checkpoint=checkpoint,
            resolution=640,
            parameter_count=2,
            failpoint=fail_after_generation,
        )
    assert read_bundle(bundle_dir)["manifest"] == first
    assert old_snapshot["manifest"] == first
    with pytest.raises(RuntimeError, match="simulated"):
        publish_bundle(
            bundle_dir,
            run_dir=run_dir,
            val_predictions=val,
            test_predictions=test,
            selected_checkpoint=checkpoint,
            resolution=640,
            parameter_count=2,
            failpoint=fail_before_manifest,
        )
    assert read_bundle(bundle_dir)["manifest"] == first
    second = publish_bundle(
        bundle_dir,
        run_dir=run_dir,
        val_predictions=val,
        test_predictions=test,
        selected_checkpoint=checkpoint,
        resolution=640,
        parameter_count=2,
    )
    assert read_bundle(bundle_dir)["manifest"] == second
    assert (bundle_dir / "generations" / first["generation"]).is_dir()
    assert old_snapshot["manifest"]["parameter_count"] == 1


@pytest.mark.parametrize("tampering", ["missing", "modified", "mixed"])
def test_bundle_rejects_modified_or_mixed_artifacts(
    tmp_path: Path, identities: tuple[dict, dict], tampering: str
) -> None:
    """
    Verify artifact hashes and split provenance after one manifest snapshot.
    """
    run_dir, val, test, checkpoint = _completed_run(tmp_path, identities)
    bundle_dir = tmp_path / "bundle"
    manifest = publish_bundle(
        bundle_dir,
        run_dir=run_dir,
        val_predictions=val,
        test_predictions=test,
        selected_checkpoint=checkpoint,
        resolution=640,
        parameter_count=1,
    )
    generation = bundle_dir / "generations" / manifest["generation"]
    if tampering == "missing":
        (generation / "val_predictions.json").unlink()
    elif tampering == "modified":
        (generation / "val_predictions.json").write_text("{}\n", encoding="utf-8")
    else:
        mixed = read_json(generation / "val_predictions.json")
        write_json(generation / "test_predictions.json", mixed)
        manifest_path = bundle_dir / "manifest.json"
        replacement = read_json(manifest_path)
        replacement["artifacts"]["test_predictions"]["sha256"] = hashlib.sha256(
            (generation / "test_predictions.json").read_bytes()
        ).hexdigest()
        write_json(manifest_path, replacement)
    with pytest.raises(ValueError):
        read_bundle(bundle_dir)


def test_smoke_bundle_requires_explicit_allowance(
    tmp_path: Path, identities: tuple[dict, dict]
) -> None:
    """
    Permit smoke bundle inspection only when the caller opts in explicitly.
    """
    run_dir, val, test, checkpoint = _completed_run(tmp_path, identities, smoke_run=True)
    bundle_dir = tmp_path / "smoke-bundle"
    publish_bundle(
        bundle_dir,
        run_dir=run_dir,
        val_predictions=val,
        test_predictions=test,
        selected_checkpoint=checkpoint,
        resolution=640,
        parameter_count=1,
    )
    with pytest.raises(ValueError, match="Smoke"):
        read_bundle(bundle_dir)
    assert read_bundle(bundle_dir, allow_smoke=True)["manifest"]["provenance"][
        "smoke_run"
    ]

def test_killed_trainer_can_resume_without_fabricating_duration(
    tmp_path: Path, identities: tuple[dict, dict]
) -> None:
    """
    Release a killed process's lock and publish resumed results with unknown timing.
    """
    import subprocess
    import sys

    run_dir, val_path, test_path, checkpoint = _completed_run(tmp_path, identities)
    child = subprocess.run(
        [
            sys.executable, "-c",
            "import os, signal, sys; "
            "from detection_evaluation import start_training_attempt; "
            "start_training_attempt(sys.argv[1], resumed=True, "
            "training_hardware={'device': 'cpu'}); "
            "os.kill(os.getpid(), signal.SIGKILL)",
            str(run_dir),
        ],
        check=False,
    )
    assert child.returncode == -9
    attempt = start_training_attempt(
        run_dir, resumed=True, completed_epochs_before=3,
        training_hardware={"device": "cpu"}, monotonic_clock=FakeClock([1]),
    )
    summary = finalize_training_attempt(
        run_dir, attempt, outcome="completed", completed_epochs=1,
        monotonic_clock=FakeClock([3]),
    )
    record = read_training_record(run_dir)
    assert record["attempts"][1]["status"] == "incomplete"
    assert record["attempts"][1]["duration_seconds"] is None
    assert record["attempts"][1]["completed_epochs"] == 1
    assert summary["completed_epochs"] == 4
    assert summary["status"] == "incomplete"
    assert summary["total_seconds"] is None
    assert summary["amortized_seconds_per_completed_epoch"] is None
    bundle_dir = tmp_path / "resumed_bundle"
    publish_bundle(
        bundle_dir, run_dir=run_dir, val_predictions=val_path,
        test_predictions=test_path, selected_checkpoint=checkpoint,
        resolution=640, parameter_count=123,
    )
    assert read_bundle(bundle_dir)["training"] == record


@pytest.mark.parametrize("reading", [float("nan"), float("inf"), -float("inf")])
def test_timing_rejects_nonfinite_clocks(
    tmp_path: Path, identities: tuple[dict, dict], reading: float
) -> None:
    """
    Refuse nonfinite timing values and release the lock after start failures.
    """
    run_dir = tmp_path / "run"
    create_protocol(run_dir, identities)
    with pytest.raises(ValueError, match="finite"):
        start_training_attempt(
            run_dir, resumed=False, training_hardware={"device": "cpu"},
            monotonic_clock=FakeClock([reading]),
        )
    attempt = start_training_attempt(
        run_dir, resumed=False, training_hardware={"device": "cpu"},
        monotonic_clock=FakeClock([0]),
    )
    with pytest.raises(ValueError, match="finite"):
        finalize_training_attempt(
            run_dir, attempt, outcome="completed", completed_epochs=1,
            monotonic_clock=FakeClock([reading]),
        )


def test_publication_revalidates_copied_artifact_provenance(
    tmp_path: Path, identities: tuple[dict, dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Preserve a published generation if a source artifact changes during copying.
    """
    from detection_evaluation import run_protocol

    run_dir, val_path, test_path, checkpoint = _completed_run(tmp_path, identities)
    bundle_dir = tmp_path / "bundle"
    arguments = dict(
        run_dir=run_dir, val_predictions=val_path, test_predictions=test_path,
        selected_checkpoint=checkpoint, resolution=640, parameter_count=123,
    )
    publish_bundle(bundle_dir, **arguments)
    before = read_bundle(bundle_dir)
    original_copy = run_protocol._copy_fsynced

    def copy_changed(source: Path, destination: Path) -> None:
        """
        Emulate a competing writer changing otherwise valid prediction metadata.
        """
        if source == val_path:
            artifact = read_json(source)
            artifact["metadata"]["model"] = "yolox_tiny"
            write_json(source, artifact)
        original_copy(source, destination)

    monkeypatch.setattr(run_protocol, "_copy_fsynced", copy_changed)
    with pytest.raises(ValueError, match="model differs"):
        publish_bundle(bundle_dir, **arguments)
    assert read_bundle(bundle_dir) == before
