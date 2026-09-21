"""
Mock-only tests for the YOLOX Stage 5 producer lifecycle.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch

from detection_evaluation import read_bundle, read_training_record, write_prediction_artifact
from yolox_pipeline import producer, yolox


def identities() -> dict[str, dict[str, Any]]:
    """
    Return a compact valid identity pair for protocol lifecycle mocks.
    """
    digest = "0" * 64
    return {
        "canonical": {
            "source_fingerprint": digest,
            "canonical_source": {
                "category_mapping": [{"id": 1, "name": "basketball"}],
            },
        },
        "loader": {"source_fingerprint": digest, "loader_fingerprint": "1" * 64},
    }


def settings(*, smoke_run: bool = False, benchmark: bool = False) -> producer.ProducerSettings:
    """
    Build no-GPU native settings for producer lifecycle tests.
    """
    return producer.ProducerSettings(
        yolox.TrainingSettings(
            epochs=2 if smoke_run else 5,
            batch_size=2,
            train_batch_limit=None,
            image_size=640,
            seed=42,
            smoke_run=smoke_run,
            show_progress=False,
            verbose_output=False,
        ),
        benchmark=benchmark,
    )


def patch_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace filesystem dataset identity work with deterministic fixtures.
    """
    monkeypatch.setattr(producer, "_dataset_identity", lambda _paths: identities())


@pytest.mark.parametrize("model_name", ["tiny", "nano"])
def test_fresh_and_resumed_runs_append_timed_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
) -> None:
    """
    Keep original provenance while appending exactly one fresh and resume attempt.
    """
    patch_identity(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls: list[bool] = []

    def fake_fit(*args: Any, **kwargs: Any) -> object:
        """
        Record resume mode and create one completed native history row.
        """
        calls.append(kwargs["resume"])
        with (run_dir / "results.csv").open("a", encoding="utf-8") as stream:
            if len(calls) == 1:
                stream.write("epoch\n")
            stream.write(f"{len(calls)}\n")
        return object()

    clock = iter((1.0, 4.0, 10.0, 16.0))
    common = dict(
        run_dir=run_dir,
        paths=producer.DatasetPaths(tmp_path, tmp_path),
        settings=settings(),
        model=producer.variant(model_name),
        exp=object(),
        checkpoint_path=tmp_path / "pretrained.pth",
        device=torch.device("cpu"),
        project_root=tmp_path,
        fit=fake_fit,
        synchronize=lambda: None,
        monotonic_clock=lambda: next(clock),
    )
    producer.fit_run(
        **common,
        original_utc=datetime(2026, 9, 19, tzinfo=UTC),
    )
    producer.fit_run(**common, resumed=True)

    protocol = producer.read_run_protocol(run_dir)
    attempts = read_training_record(run_dir)["attempts"]
    assert protocol["model"] == producer.variant(model_name).name
    assert protocol["original_utc"] == "2026-09-19T00:00:00Z"
    assert [attempt["kind"] for attempt in attempts] == ["fresh", "resume"]
    assert [attempt["duration_seconds"] for attempt in attempts] == [3.0, 6.0]
    assert [attempt["completed_epochs"] for attempt in attempts] == [1, 1]
    assert calls == [False, True]


@pytest.mark.parametrize("model_name", ["tiny", "nano"])
def test_interrupted_fit_finalizes_without_publishing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
) -> None:
    """
    Record timing when the native fit raises before any post-training work.
    """
    patch_identity(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def fail_fit(*_args: Any, **_kwargs: Any) -> None:
        """
        Simulate a catchable native training failure.
        """
        raise RuntimeError("training failed")

    clock = iter((10.0, 15.0))
    with pytest.raises(RuntimeError, match="training failed"):
        producer.fit_run(
            run_dir,
            producer.DatasetPaths(tmp_path, tmp_path),
            settings(),
            producer.variant(model_name),
            object(),
            tmp_path / "pretrained.pth",
            torch.device("cpu"),
            tmp_path,
            original_utc=datetime(2026, 9, 19, tzinfo=UTC),
            fit=fail_fit,
            synchronize=lambda: None,
            monotonic_clock=lambda: next(clock),
        )

    attempt = read_training_record(run_dir)["attempts"][0]
    assert attempt["status"] == "interrupted"
    assert attempt["duration_seconds"] == 5.0
    assert attempt["completed_epochs"] == 0


def test_smoke_paths_request_the_complete_stage_two_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Materialize the shared smoke layout with the required 16/8/8 selection.
    """
    captured: dict[str, Any] = {}
    smoke_root = tmp_path / "processed" / "coco_basketball_smoke"

    def fake_prepare(source: Path, destination: Path) -> dict[str, Any]:
        """
        Record the source and return the exact Stage 2 smoke record.
        """
        captured["source"] = source
        captured["destination"] = destination
        return {"smoke_limits": {"train": 16, "val": 8, "test": 8}}

    monkeypatch.setattr(producer, "prepare_smoke_layouts", fake_prepare)
    monkeypatch.setattr(producer, "smoke_layout_directory", lambda _source: smoke_root)
    paths = producer.resolve_dataset_paths(
        tmp_path / "coco_basketball",
        tmp_path / "yolo_basketball",
        smoke_run=True,
    )

    assert captured == {
        "source": (tmp_path / "coco_basketball").resolve(),
        "destination": smoke_root,
    }
    assert paths == producer.DatasetPaths(smoke_root / "coco", smoke_root / "yolo")


def test_smoke_settings_require_complete_layout_and_no_batch_limit() -> None:
    """
    Enforce exact two-epoch Stage 2 smoke training controls.
    """
    smoke = settings(smoke_run=True)
    smoke.validate()
    assert smoke.training.epochs == 2
    assert smoke.training.train_batch_limit is None

    invalid = producer.ProducerSettings(
        yolox.TrainingSettings(
            epochs=2,
            batch_size=2,
            train_batch_limit=1,
            image_size=640,
            seed=42,
            smoke_run=True,
            show_progress=False,
            verbose_output=False,
        )
    )
    with pytest.raises(ValueError, match="without train batch limits"):
        invalid.validate()


def test_prepare_run_rejects_changed_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Refuse resume or recovery after a loader identity changes.
    """
    patch_identity(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths = producer.DatasetPaths(tmp_path, tmp_path)
    current = settings()
    producer.prepare_run(
        run_dir,
        paths,
        current,
        producer.variant("tiny"),
        original_utc=datetime(2026, 9, 19, tzinfo=UTC),
    )
    changed = identities()
    changed["loader"]["loader_fingerprint"] = "2" * 64
    monkeypatch.setattr(producer, "_dataset_identity", lambda _paths: changed)

    with pytest.raises(ValueError, match="Dataset identity mismatch"):
        producer.prepare_run(run_dir, paths, current, producer.variant("tiny"))


def test_recovery_republishes_without_training_after_a_failed_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Republish a completed run without changing attempts or invoking native fit.
    """
    patch_identity(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "weights" / "best_ckpt.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    paths = producer.DatasetPaths(tmp_path, tmp_path)
    current = settings(smoke_run=True)
    model = producer.variant("nano")
    producer.prepare_run(
        run_dir,
        paths,
        current,
        model,
        original_utc=datetime(2026, 9, 19, tzinfo=UTC),
    )
    attempt = producer.start_training_attempt(
        run_dir,
        resumed=False,
        training_hardware={"device": "cpu"},
        monotonic_clock=lambda: 1.0,
    )
    producer.finalize_training_attempt(
        run_dir,
        attempt,
        outcome="completed",
        completed_epochs=2,
        monotonic_clock=lambda: 3.0,
    )
    exported: list[str] = []

    def fake_export(*args: Any, **kwargs: Any) -> Path:
        """
        Write minimal valid artifacts through the native export handoff.
        """
        destination = args[2]
        split = kwargs["metadata"]["split"]
        annotation = tmp_path / f"{split}.json"
        annotation.write_text(
            '{"images":[{"id":1,"file_name":"image.jpg"}],'
            '"annotations":[],"categories":[{"id":1,"name":"basketball"}]}',
            encoding="utf-8",
        )
        exported.append(split)
        return write_prediction_artifact(destination, annotation, [], metadata=kwargs["metadata"])

    monkeypatch.setattr(producer.artifacts, "export_model_predictions", fake_export)
    monkeypatch.setattr(producer, "version", lambda _distribution: "test")
    monkeypatch.setattr(
        producer,
        "publish_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("publish failed")),
    )

    with pytest.raises(RuntimeError, match="publish failed"):
        producer.recover_and_publish(
            run_dir,
            paths,
            current,
            model,
            torch.device("cpu"),
            load_model=lambda *_args: torch.nn.Linear(1, 1),
            bundle_root=tmp_path / "bundle",
        )
    assert len(read_training_record(run_dir)["attempts"]) == 1

    monkeypatch.undo()
    patch_identity(monkeypatch)
    monkeypatch.setattr(producer.artifacts, "export_model_predictions", fake_export)
    monkeypatch.setattr(producer, "version", lambda _distribution: "test")
    first = producer.recover_and_publish(
        run_dir,
        paths,
        current,
        model,
        torch.device("cpu"),
        load_model=lambda *_args: torch.nn.Linear(1, 1),
        bundle_root=tmp_path / "bundle",
    )
    second = producer.recover_and_publish(
        run_dir,
        paths,
        current,
        model,
        torch.device("cpu"),
        load_model=lambda *_args: torch.nn.Linear(1, 1),
        bundle_root=tmp_path / "bundle",
    )

    assert exported == ["val", "test", "val", "test", "val", "test"]
    assert len(read_training_record(run_dir)["attempts"]) == 1
    assert read_bundle(tmp_path / "bundle", allow_smoke=True)["manifest"] == second
    assert first["provenance"]["model"] == "yolox_nano"


@pytest.mark.parametrize(
    ("model_name", "expected_class"),
    [("tiny", yolox.BasketballTinyExp), ("nano", yolox.BasketballNanoExp)],
)
def test_recovery_uses_the_variant_experiment_and_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
    expected_class: type[yolox.BasketballTinyExp],
) -> None:
    """
    Reconstruct the requested variant and only load its selected best checkpoint.
    """
    patch_identity(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "weights" / "best_ckpt.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    paths = producer.DatasetPaths(tmp_path, tmp_path)
    current = settings()
    model = producer.variant(model_name)
    producer.prepare_run(
        run_dir,
        paths,
        current,
        model,
        original_utc=datetime(2026, 9, 19, tzinfo=UTC),
    )
    attempt = producer.start_training_attempt(
        run_dir,
        resumed=False,
        training_hardware={"device": "cpu"},
        monotonic_clock=lambda: 1.0,
    )
    producer.finalize_training_attempt(
        run_dir,
        attempt,
        outcome="completed",
        completed_epochs=5,
        monotonic_clock=lambda: 2.0,
    )
    captured: dict[str, Any] = {}

    def fake_load(exp: Any, path: Path, device: torch.device) -> torch.nn.Module:
        """
        Capture the variant experiment and selected checkpoint.
        """
        captured.update({"exp": exp, "path": path, "device": device})
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(producer, "version", lambda _distribution: "test")
    monkeypatch.setattr(
        producer.artifacts,
        "export_model_predictions",
        lambda *_args, **_kwargs: tmp_path / "prediction.json",
    )
    monkeypatch.setattr(producer, "publish_bundle", lambda *_args, **_kwargs: {})

    producer.recover_and_publish(
        run_dir,
        paths,
        current,
        model,
        torch.device("cpu"),
        load_model=fake_load,
    )

    assert isinstance(captured["exp"], expected_class)
    assert captured["path"] == checkpoint
    assert captured["device"] == torch.device("cpu")

@pytest.mark.parametrize("model_name", ["tiny", "nano"])
def test_resume_settings_inherit_persisted_smoke_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_name: str,
) -> None:
    """
    Resume either variant with its saved smoke controls and original epoch budget.
    """
    patch_identity(monkeypatch)
    saved = settings(smoke_run=True)
    producer.prepare_run(
        tmp_path, producer.DatasetPaths(tmp_path, tmp_path), saved,
        producer.variant(model_name), original_utc=datetime(2026, 9, 19, tzinfo=UTC),
    )
    restored = producer.settings_from_run(tmp_path)
    assert restored.training.smoke_run is True
    assert restored.training.resolved_resume_run_dir == tmp_path
    assert restored.protocol_settings() == saved.protocol_settings()
