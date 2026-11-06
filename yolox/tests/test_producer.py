"""
Mock-only tests for the YOLOX Stage 5 producer lifecycle.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch
from detection_common.utils.json_io import read_json, write_json
from detection_evaluation import (
    read_bundle,
    read_training_record,
    write_prediction_artifact,
)
from PIL import Image

from yolox_pipeline import producer, yolox


def test_direct_coco_loader_identity_and_resume_reject_changed_source(tmp_path):
    """Verify native ball targets and reject drift without any YOLO export."""
    source = tmp_path / "coco"
    for split in ("train", "val", "test"):
        images = []
        for index in (1, 2):
            path = source / "images" / split / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 20), (index * 50, 20, 40)).save(path)
            images.append(
                {"id": index, "file_name": path.name, "width": 40, "height": 20}
            )
        path = source / "annotations" / f"instances_{split}.json"
        path.parent.mkdir(exist_ok=True)
        write_json(
            path,
            {
                "images": images,
                "categories": [{"id": 7, "name": "ball"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 7,
                        "bbox": [4, 2, 20, 10],
                        "area": 200,
                        "iscrowd": 0,
                    }
                ],
            },
        )
    paths = producer.resolve_dataset_paths(source, smoke_run=False)
    identity = producer._dataset_identity(paths)
    assert identity["loader"]["loader"]["format"] == "coco"
    assert not (tmp_path / "yolo").exists()
    exp = yolox.YOLOXNanoExp(
        source, tmp_path / "runs", 2, 640, "ball", class_names=("ball",)
    )
    assert yolox.verify_coco_loaders(exp, ("ball",)) == {
        split: {"images": 2, "backgrounds": 1} for split in ("train", "val", "test")
    }
    run_dir = tmp_path / "run"
    model = producer.variant(
        "nano",
        logical_dataset="ball",
        source_notebook="nb03.04-yolox_nano_sport_object.ipynb",
    )
    producer.prepare_run(
        run_dir,
        paths,
        settings(),
        model,
        original_utc=datetime(2026, 11, 5, tzinfo=UTC),
    )
    path = source / "annotations/instances_train.json"
    document = read_json(path)
    document["annotations"][0]["bbox"][0] += 1
    write_json(path, document)
    with pytest.raises(ValueError, match="Dataset identity mismatch"):
        producer.prepare_run(run_dir, paths, settings(), model)


def identities(class_name: str = "basketball") -> dict[str, dict[str, Any]]:
    """
    Return a compact valid identity pair for protocol lifecycle mocks.
    """
    digest = "0" * 64
    return {
        "canonical": {
            "source_fingerprint": digest,
            "canonical_source": {
                "category_mapping": [{"id": 1, "name": class_name}],
            },
        },
        "loader": {"source_fingerprint": digest, "loader_fingerprint": "1" * 64},
    }


def settings(
    *, smoke_run: bool = False, benchmark: bool = False
) -> producer.ProducerSettings:
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


def patch_identity(
    monkeypatch: pytest.MonkeyPatch, class_name: str = "basketball"
) -> None:
    """
    Replace filesystem dataset identity work with deterministic fixtures.
    """
    monkeypatch.setattr(
        producer, "_dataset_identity", lambda _paths: identities(class_name)
    )


@pytest.mark.parametrize("class_name", ["basketball", "football", "ball"])
@pytest.mark.parametrize("model_name", ["tiny", "nano"])
def test_fresh_and_resumed_runs_append_timed_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
    class_name: str,
) -> None:
    """
    Keep original provenance while appending exactly one fresh and resume attempt.
    """
    patch_identity(monkeypatch, class_name)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls: list[bool] = []

    def fake_fit(*args: Any, **kwargs: Any) -> object:
        """
        Record resume mode and create one completed native history row.
        """
        assert kwargs["class_names"] == (class_name,)
        calls.append(kwargs["resume"])
        with (run_dir / "results.csv").open("a", encoding="utf-8") as stream:
            if len(calls) == 1:
                stream.write("epoch\n")
            stream.write(f"{len(calls)}\n")
        return object()

    clock = iter((1.0, 4.0, 10.0, 16.0))
    common = {
        "run_dir": run_dir,
        "paths": producer.DatasetPaths(tmp_path),
        "settings": settings(),
        "model": producer.variant(
            model_name, logical_dataset=class_name, source_notebook="training.ipynb"
        ),
        "exp": object(),
        "checkpoint_path": tmp_path / "pretrained.pth",
        "device": torch.device("cpu"),
        "project_root": tmp_path,
        "fit": fake_fit,
        "synchronize": lambda: None,
        "monotonic_clock": lambda: next(clock),
    }
    producer.fit_run(
        **common,
        original_utc=datetime(2026, 9, 19, tzinfo=UTC),
    )
    producer.fit_run(**common, resumed=True)

    protocol = producer.read_run_protocol(run_dir)
    attempts = read_training_record(run_dir)["attempts"]
    assert protocol["model"] == producer.variant(model_name).name
    assert protocol["logical_dataset"] == class_name
    assert protocol["source_notebook"] == "training.ipynb"
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
            producer.DatasetPaths(tmp_path),
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

    def fake_prepare(
        source: Path, destination: Path, *, formats: tuple[str, ...]
    ) -> dict[str, Any]:
        """
        Record the source and return the exact Stage 2 smoke record.
        """
        captured["source"] = source
        captured["destination"] = destination
        captured["formats"] = formats
        return {"smoke_limits": {"train": 16, "val": 8, "test": 8}}

    monkeypatch.setattr(producer, "prepare_smoke_layouts", fake_prepare)
    monkeypatch.setattr(producer, "smoke_layout_directory", lambda _source: smoke_root)
    paths = producer.resolve_dataset_paths(
        tmp_path / "coco_basketball",
        smoke_run=True,
    )

    assert captured == {
        "source": (tmp_path / "coco_basketball").resolve(),
        "destination": smoke_root,
        "formats": ("coco",),
    }
    assert paths == producer.DatasetPaths(smoke_root / "coco")


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
    paths = producer.DatasetPaths(tmp_path)
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


@pytest.mark.parametrize("class_name", ["basketball", "football", "ball"])
def test_recovery_republishes_without_training_after_a_failed_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, class_name: str
) -> None:
    """
    Republish a completed run without changing attempts or invoking native fit.
    """
    patch_identity(monkeypatch, class_name)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "weights" / "best_ckpt.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    paths = producer.DatasetPaths(tmp_path)
    current = settings(smoke_run=True)
    model = producer.variant(
        "nano",
        logical_dataset=class_name,
        source_notebook="nb03.05-yolox_nano_football.ipynb"
        if class_name == "football"
        else "nb03.03-yolox_nano_large_basketball.ipynb",
    )
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
    bundle = (
        tmp_path
        / "output"
        / "evaluation"
        / ("basketball_large_dataset" if class_name == "basketball" else class_name)
        / "yolox_nano"
        / "run"
    )
    exported: list[str] = []

    def fake_export(*args: Any, **kwargs: Any) -> Path:
        """
        Write minimal valid artifacts through the native export handoff.
        """
        assert kwargs["category_ids"] == [1]
        destination = args[2]
        split = kwargs["metadata"]["split"]
        annotation = tmp_path / f"{split}.json"
        write_json(
            annotation,
            {
                "images": [{"id": 1, "file_name": "image.jpg"}],
                "annotations": [],
                "categories": [{"id": 1, "name": class_name}],
            },
        )
        exported.append(split)
        return write_prediction_artifact(
            destination, annotation, [], metadata=kwargs["metadata"]
        )

    monkeypatch.setattr(producer.artifacts, "export_model_predictions", fake_export)
    monkeypatch.setattr(producer, "version", lambda _distribution: "test")
    monkeypatch.setattr(producer, "OUTPUT_ROOT", tmp_path / "output")
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
        )
    assert len(read_training_record(run_dir)["attempts"]) == 1

    monkeypatch.undo()
    patch_identity(monkeypatch, class_name)
    monkeypatch.setattr(producer.artifacts, "export_model_predictions", fake_export)
    monkeypatch.setattr(producer, "version", lambda _distribution: "test")
    monkeypatch.setattr(producer, "OUTPUT_ROOT", tmp_path / "output")
    first = producer.recover_and_publish(
        run_dir,
        paths,
        current,
        model,
        torch.device("cpu"),
        load_model=lambda *_args: torch.nn.Linear(1, 1),
    )
    second = producer.recover_and_publish(
        run_dir,
        paths,
        current,
        model,
        torch.device("cpu"),
        load_model=lambda *_args: torch.nn.Linear(1, 1),
    )

    assert exported == ["val", "test", "val", "test", "val", "test"]
    assert len(read_training_record(run_dir)["attempts"]) == 1
    assert read_bundle(bundle, allow_smoke=True)["manifest"] == second
    assert first["provenance"]["model"] == "yolox_nano"
    assert producer.read_run_protocol(run_dir)["logical_dataset"] == class_name


@pytest.mark.parametrize(
    ("model_name", "expected_class"),
    [("tiny", yolox.YOLOXTinyExp), ("nano", yolox.YOLOXNanoExp)],
)
@pytest.mark.parametrize("class_count", [1, 2])
def test_recovery_uses_the_variant_experiment_and_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
    expected_class: type[yolox.YOLOXTinyExp],
    class_count: int,
) -> None:
    """
    Reconstruct the requested variant and only load its selected best checkpoint.
    """
    identity = identities()
    if class_count == 2:
        identity["canonical"]["canonical_source"]["category_mapping"].append(
            {"id": 19, "name": "football"}
        )
    monkeypatch.setattr(producer, "_dataset_identity", lambda paths: identity)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "weights" / "best_ckpt.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    paths = producer.DatasetPaths(tmp_path)
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

    def fake_export(*args: Any, **kwargs: Any) -> Path:
        assert kwargs["category_ids"] == ([1, 19] if class_count == 2 else [1])
        return tmp_path / "prediction.json"

    monkeypatch.setattr(producer.artifacts, "export_model_predictions", fake_export)
    monkeypatch.setattr(producer, "publish_bundle", lambda *_args, **_kwargs: {})

    producer.recover_and_publish(
        run_dir,
        paths,
        current,
        model,
        torch.device("cpu"),
        load_model=fake_load,
    )

    assert captured["exp"].num_classes == class_count
    assert isinstance(captured["exp"], expected_class)
    assert captured["path"] == checkpoint
    assert captured["device"] == torch.device("cpu")


@pytest.mark.parametrize("model_name", ["tiny", "nano"])
def test_resume_settings_inherit_persisted_smoke_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_name: str,
) -> None:
    """
    Resume either variant with its saved smoke controls and original epoch budget.
    """
    patch_identity(monkeypatch)
    saved = settings(smoke_run=True)
    producer.prepare_run(
        tmp_path,
        producer.DatasetPaths(tmp_path),
        saved,
        producer.variant(model_name),
        original_utc=datetime(2026, 9, 19, tzinfo=UTC),
    )
    restored = producer.settings_from_run(tmp_path)
    assert restored.training.smoke_run is True
    assert restored.training.resolved_resume_run_dir == tmp_path
    assert restored.protocol_settings() == saved.protocol_settings()
