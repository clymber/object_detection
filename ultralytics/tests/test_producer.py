"""
Mock-only tests for the Ultralytics Stage 4 producer lifecycle.
"""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from detection_evaluation import read_bundle, read_training_record
from ultralytics_pipeline import producer


class FakeModel:
    """
    Minimal model that records training kwargs and can fail before returning.
    """

    def __init__(self, run_dir: Path, *, fail: bool = False) -> None:
        """
        Initialize the fake model for a reserved output directory.
        """
        self.run_dir = run_dir
        self.fail = fail
        self.callbacks: list[tuple[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def add_callback(self, event: str, callback: Any) -> None:
        """
        Record a registered callback.
        """
        self.callbacks.append((event, callback))

    def train(self, **kwargs: Any) -> Any:
        """
        Raise before returning or report the reserved training directory.
        """
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("training failed")
        return SimpleNamespace(save_dir=self.run_dir)


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


def patch_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Replace filesystem dataset identity work with valid deterministic fixtures.
    """
    monkeypatch.setattr(producer, "_dataset_identity", lambda _paths: identities())


def test_train_failure_finalizes_interrupted_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Persist an interrupted attempt when native train raises before returning.
    """
    patch_identity(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data_yaml = tmp_path / "dataset" / "data.yaml"
    data_yaml.parent.mkdir()
    data_yaml.write_text("path: .\n", encoding="utf-8")
    model = FakeModel(run_dir, fail=True)
    clock = iter((10.0, 14.0))

    with pytest.raises(RuntimeError, match="training failed"):
        producer.train_pinned_run(
            model,
            run_dir,
            producer.DatasetPaths(tmp_path, data_yaml.parent),
            producer.TrainingSettings(device="cpu"),
            original_utc=datetime(2026, 9, 19, tzinfo=UTC),
            synchronize=lambda: None,
            monotonic_clock=lambda: next(clock),
        )

    attempt = read_training_record(run_dir)["attempts"][0]
    assert attempt["status"] == "interrupted"
    assert attempt["duration_seconds"] == 4.0
    assert attempt["completed_epochs"] == 0


def test_callback_rejects_pinned_directory_mismatch(tmp_path: Path) -> None:
    """
    Reject native callbacks that try to relocate the reserved run.
    """
    callback = producer.reserved_run_callback(tmp_path / "expected")

    with pytest.raises(RuntimeError, match="Trainer directory mismatch"):
        callback(SimpleNamespace(save_dir=tmp_path / "other"))


def test_smoke_settings_use_exact_epochs_and_no_truncation() -> None:
    """
    Keep smoke at two complete-layout epochs without unsupported truncation keys.
    """
    settings = producer.settings_from_environment(
        workers=0, environment={"ULTRALYTICS_SMOKE": "1"}
    )
    kwargs = settings.train_kwargs(Path("data.yaml"), Path("run"), resume=False)

    assert settings.smoke_run
    assert kwargs["epochs"] == 2
    assert "fraction" not in kwargs
    assert "train_batch_limit" not in kwargs
    assert "smoke_run" not in kwargs


def test_prepare_run_rejects_changed_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Refuse resume or recovery when the saved dataset identity has changed.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths = producer.DatasetPaths(tmp_path, tmp_path)
    monkeypatch.setattr(producer, "_dataset_identity", lambda _paths: identities())
    settings = producer.TrainingSettings(device="cpu")
    producer.prepare_run(
        run_dir, paths, settings, original_utc=datetime(2026, 9, 19, tzinfo=UTC)
    )
    changed = identities()
    changed["loader"]["loader_fingerprint"] = "2" * 64
    monkeypatch.setattr(producer, "_dataset_identity", lambda _paths: changed)

    with pytest.raises(ValueError, match="Dataset identity mismatch"):
        producer.prepare_run(run_dir, paths, settings)


def test_train_timing_synchronizes_both_boundaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Synchronize and clock immediately around only the native training invocation.
    """
    patch_identity(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    data_yaml = tmp_path / "dataset" / "data.yaml"
    data_yaml.parent.mkdir()
    data_yaml.write_text("path: .\n", encoding="utf-8")
    synchronized: list[str] = []
    clock = iter((10.0, 15.0))

    producer.train_pinned_run(
        FakeModel(run_dir),
        run_dir,
        producer.DatasetPaths(tmp_path, data_yaml.parent),
        producer.TrainingSettings(device="cpu"),
        original_utc=datetime(2026, 9, 19, tzinfo=UTC),
        synchronize=lambda: synchronized.append("sync"),
        monotonic_clock=lambda: next(clock),
    )

    assert synchronized == ["sync", "sync"]
    assert read_training_record(run_dir)["summary"]["total_seconds"] == 5.0


def test_recovery_publishes_bundle_without_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """
    Recover a finished run without registering another train call or attempt.
    """
    patch_identity(monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    settings = producer.TrainingSettings(
        device="cpu", smoke_run=True, benchmark=False
    )
    paths = producer.DatasetPaths(tmp_path, tmp_path)
    producer.prepare_run(
        run_dir, paths, settings, original_utc=datetime(2026, 9, 19, tzinfo=UTC)
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
    artifacts = {}

    def fake_export(
        _model: Any,
        destination: Path,
        *_args: Any,
        **kwargs: Any,
    ) -> Path:
        """
        Write syntactically valid minimal shared artifacts without model frameworks.
        """
        from detection_evaluation import write_prediction_artifact

        annotation = tmp_path / f"{kwargs['metadata']['split']}.json"
        annotation.write_text(
            "{"
            '\"images\":[{\"id\":1,\"file_name\":\"image.jpg\"}],'
            '\"annotations\":[],\"categories\":[{\"id\":1,\"name\":\"basketball\"}]'
            "}",
            encoding="utf-8",
        )
        artifacts[kwargs["metadata"]["split"]] = write_prediction_artifact(
            destination, annotation, [], metadata=kwargs["metadata"]
        )
        return artifacts[kwargs["metadata"]["split"]]

    monkeypatch.setattr(producer, "export_model_predictions", fake_export)
    model = SimpleNamespace(
        names={0: "basketball"},
        model=SimpleNamespace(parameters=lambda: [SimpleNamespace(numel=lambda: 1)]),
    )
    manifest = producer.recover_and_publish(
        run_dir,
        paths,
        settings,
        load_model=lambda path: model,
        bundle_root=tmp_path / "bundle",
    )

    assert manifest["provenance"]["smoke_run"]
    assert len(read_training_record(run_dir)["attempts"]) == 1
    assert read_bundle(tmp_path / "bundle", allow_smoke=True)["manifest"] == manifest

def test_resume_settings_inherit_persisted_smoke_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """
    Restore smoke controls without requiring the original shell environment.
    """
    patch_identity(monkeypatch)
    saved = producer.TrainingSettings(epochs=2, smoke_run=True, batch_size=3)
    producer.prepare_run(
        tmp_path, producer.DatasetPaths(tmp_path, tmp_path), saved,
        original_utc=datetime(2026, 9, 19, tzinfo=UTC),
    )
    assert producer.settings_from_run(tmp_path) == saved
