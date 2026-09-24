"""
Test RF-DETR run configuration, recovery, history, and prediction conversion.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
import torch
from detection_common.utils.json_io import read_json, write_json
from detection_evaluation import (
    file_sha256,
    finalize_training_attempt,
    read_bundle,
    read_training_record,
    start_training_attempt,
    write_prediction_artifact,
)
from PIL import Image

from rfdetr_pipeline import rfdetr


def _identity() -> dict[str, dict[str, object]]:
    """
    Return a compact valid dataset identity pair for run protocol fixtures.
    """
    digest = "0" * 64
    return {
        "canonical": {
            "source_fingerprint": digest,
            "canonical_source": {"category_mapping": [{"id": 1, "name": "basketball"}]},
        },
        "loader": {"source_fingerprint": digest, "loader_fingerprint": "1" * 64},
    }


def _run_config(path: Path, settings: rfdetr.TrainingSettings) -> None:
    """
    Write the saved settings needed by evaluation and checkpoint recovery.
    """
    values = vars(settings).copy()
    values.pop("mode")
    values.pop("run_dir")
    write_json(path / rfdetr.RUN_CONFIG, {"settings": values})


def test_settings_support_smoke_and_removed_evaluate_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Bound a new smoke run and direct removed evaluate requests to recovery.
    """
    monkeypatch.delenv("RFDETR_EPOCHS", raising=False)
    smoke = rfdetr.settings_from_env(overrides={"smoke_run": True})
    assert smoke.epochs == 2
    assert smoke.batch_size == 4
    assert smoke.grad_accum_steps == 1
    assert smoke.samples_per_optimizer_step == 4
    assert smoke.early_stopping is True
    assert smoke.early_stopping_patience == 10
    assert smoke.early_stopping_min_delta == 0.001
    assert smoke.early_stopping_use_ema is True
    assert smoke.mode is rfdetr.RunMode.FRESH

    with pytest.raises(ValueError, match="postprocess_cli"):
        rfdetr.settings_from_env(
            overrides={"mode": "evaluate", "run_dir": str(tmp_path)},
        )


def test_resume_of_a_preprotocol_run_requires_a_fresh_experiment(tmp_path: Path) -> None:
    """
    Reject historical runs rather than attempting an unsafe protocol conversion.
    """
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="fresh experiment"):
        rfdetr.settings_from_env(
            overrides={"mode": "resume", "run_dir": str(run_dir)},
        )


def test_fresh_run_uses_configured_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Place new RF-DETR runs below OUTPUT_ROOT, including mounted overrides.
    """
    output_root = tmp_path / "mounted-output"
    monkeypatch.setattr(rfdetr, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(rfdetr, "_dataset_identity", lambda _manifest: _identity())
    settings = rfdetr.TrainingSettings(epochs=2, smoke_run=True)
    manifest = {"fingerprint": "fixture", "source_dir": "fixture"}
    run_dir = rfdetr.prepare_run(settings, manifest, {})
    assert run_dir.parent == output_root / "runs" / "basketball"
    protocol = read_json(run_dir / "run_protocol.json")
    assert protocol["smoke_run"] is True


def test_prepare_run_rejects_changed_dataset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Refuse a resume when the current canonical or loader identity has changed.
    """
    monkeypatch.setattr(rfdetr, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(rfdetr, "_dataset_identity", lambda _manifest: _identity())
    settings = rfdetr.TrainingSettings(epochs=2, smoke_run=True)
    manifest = {"fingerprint": "fixture", "source_dir": "fixture"}
    run_dir = rfdetr.prepare_run(settings, manifest, {})
    changed = _identity()
    changed["loader"]["loader_fingerprint"] = "2" * 64
    monkeypatch.setattr(rfdetr, "_dataset_identity", lambda _manifest: changed)

    with pytest.raises(ValueError, match="Dataset identity mismatch"):
        rfdetr.prepare_run(
            rfdetr.TrainingSettings(
                epochs=2,
                smoke_run=True,
                mode="resume",
                run_dir=str(run_dir),
            ),
            manifest,
            {},
        )


@pytest.mark.parametrize(
    "override",
    [
        {"resolution": 641},
        {"batch_size": True},
        {"num_workers": 1.5},
        {"lr": float("nan")},
        {"amp": 1},
        {"early_stopping_patience": 0},
        {"early_stopping_min_delta": -0.001},
        {"mode": "invalid"},
    ],
)
def test_settings_reject_ambiguous_or_invalid_values(override: dict) -> None:
    """
    Fail before training when numeric settings cannot define the requested run.
    """
    with pytest.raises(ValueError):
        rfdetr.TrainingSettings(**override)


def test_training_uses_supported_single_gpu_device_form(tmp_path: Path) -> None:
    """
    Avoid RF-DETR 1.10.1's indexed-device list failure in its trainer helper.
    """
    kwargs = rfdetr.train_kwargs(rfdetr.TrainingSettings(), tmp_path, tmp_path)
    assert kwargs["device"] == "cuda"
    assert kwargs["devices"] == 1
    assert kwargs["progress_bar"] == "tqdm"
    assert "eval_ema_only" not in kwargs
    assert kwargs["early_stopping"] is True
    assert kwargs["early_stopping_patience"] == 10
    assert kwargs["early_stopping_min_delta"] == 0.001
    assert kwargs["early_stopping_use_ema"] is True


def test_resume_checkpoint_requires_full_state_and_remaining_budget(
    tmp_path: Path,
) -> None:
    """
    Accept only a full, incomplete Lightning checkpoint for true recovery.
    """
    checkpoint = tmp_path / "last.ckpt"
    state = {
        "state_dict": {"model.w": torch.tensor([1.0])},
        "optimizer_states": [{"state": {}}],
        "lr_schedulers": [{"last_epoch": 2}],
        "epoch": 2,
        "global_step": 9,
    }
    torch.save(state, checkpoint)
    assert rfdetr.validate_resume_checkpoint(checkpoint, 5) == {
        "completed_epochs": 3,
        "global_step": 9,
    }
    with pytest.raises(ValueError, match="already reached"):
        rfdetr.validate_resume_checkpoint(checkpoint, 3)
    state["optimizer_states"] = []
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="optimizer_states"):
        rfdetr.validate_resume_checkpoint(checkpoint, 5)


def test_sparse_history_combines_training_and_validation_rows(tmp_path: Path) -> None:
    """
    Merge Lightning's sparse rows without losing metrics from the same epoch.
    """
    pd.DataFrame(
        [
            {"epoch": 0, "train/loss": 4.0, "val/mAP_50_95": None},
            {"epoch": 0, "train/loss": None, "val/mAP_50_95": 0.2},
            {"epoch": 1, "train/loss": 3.0, "val/mAP_50_95": None},
            {"epoch": 1, "train/loss": None, "val/mAP_50_95": 0.3},
        ]
    ).to_csv(tmp_path / "metrics.csv", index=False)
    history = rfdetr.read_training_history(tmp_path)
    assert history["epoch"].tolist() == [1, 2]
    assert history["train/loss"].tolist() == [4.0, 3.0]
    assert history["val/mAP_50_95"].tolist() == [0.2, 0.3]
    assert (tmp_path / "results.csv").is_file()


def test_best_checkpoint_recovers_stripped_metadata_and_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verify selected weights and restore 640 pixels from the unstripped source.
    """
    settings = rfdetr.TrainingSettings()
    _run_config(tmp_path, settings)
    weights = {"w": torch.tensor([1.0, 2.0])}
    total = tmp_path / rfdetr.BEST_CHECKPOINT
    source = tmp_path / "checkpoint_best_ema.pth"
    torch.save({"model": weights, "args": {}, "best_total_source": "ema"}, total)
    torch.save(
        {"model": weights, "epoch": 6, "model_config": {"resolution": 640}},
        source,
    )
    calls = []

    class FakeRFDETRSmall:
        """
        Record checkpoint reload settings and expose the adapter's model shape.
        """

        @classmethod
        def from_checkpoint(cls, path: Path, **kwargs):
            """
            Return a one-class model while retaining the requested constructor values.
            """
            calls.append((path, kwargs))
            return SimpleNamespace(
                class_names=["basketball"],
                model_config=SimpleNamespace(num_classes=1),
                model=SimpleNamespace(model=torch.nn.Linear(1, 1)),
            )

    module = ModuleType("rfdetr")
    module.RFDETRSmall = FakeRFDETRSmall
    monkeypatch.setitem(sys.modules, "rfdetr", module)
    _, metadata = rfdetr.load_best_model(tmp_path)
    assert metadata["best_epoch"] == 7
    assert metadata["selected_weights"] == "ema"
    assert metadata["source_checkpoint"] == str(source)
    assert calls[0][0] == total
    assert calls[0][1]["resolution"] == 640
    assert calls[0][1]["positional_encoding_size"] == 40
    assert calls[0][1]["trust_checkpoint"] is True
    assert read_json(tmp_path / "best_checkpoint.json") == metadata


def test_best_checkpoint_rejects_a_mismatched_selected_source(
    tmp_path: Path,
) -> None:
    """
    Stop if the promoted total weights do not equal their named source checkpoint.
    """
    _run_config(tmp_path, rfdetr.TrainingSettings())
    torch.save(
        {
            "model": {"w": torch.tensor([1.0])},
            "args": {},
            "best_total_source": "regular",
        },
        tmp_path / rfdetr.BEST_CHECKPOINT,
    )
    torch.save(
        {
            "model": {"w": torch.tensor([2.0])},
            "epoch": 0,
            "model_config": {"resolution": 640},
        },
        tmp_path / "checkpoint_best_regular.pth",
    )
    with pytest.raises(ValueError, match="differ"):
        rfdetr.load_best_model(tmp_path)


def test_export_onnx_model_uses_static_best_checkpoint_name(tmp_path: Path) -> None:
    """
    Export the verified model at its trained resolution with a stable artifact name.
    """
    calls = []

    def fake_export(**kwargs):
        """
        Record native export arguments and create its declared artifact.
        """
        calls.append(kwargs)
        output = Path(kwargs["output_dir"]) / f"{kwargs['output_name']}.onnx"
        output.touch()
        return output

    model = SimpleNamespace(export=fake_export)
    exported = rfdetr.export_onnx_model(model, tmp_path, resolution=640)
    assert exported == tmp_path / rfdetr.BEST_ONNX_MODEL
    assert calls == [
        {
            "format": "onnx",
            "output_dir": str(tmp_path),
            "output_name": "checkpoint_best_total",
            "shape": (640, 640),
            "batch_size": 1,
            "dynamic_batch": False,
            "verbose": False,
        }
    ]


def test_export_onnx_model_rejects_an_unexpected_native_path(tmp_path: Path) -> None:
    """
    Fail when the native exporter does not return the requested artifact path.
    """

    def fake_export(**kwargs):
        """
        Return a different ONNX filename to exercise path validation.
        """
        output = Path(kwargs["output_dir"]) / "unexpected.onnx"
        output.touch()
        return output

    model = SimpleNamespace(export=fake_export)
    with pytest.raises(RuntimeError, match="unexpected path"):
        rfdetr.export_onnx_model(model, tmp_path, resolution=640)


def test_export_onnx_model_requires_the_declared_file(tmp_path: Path) -> None:
    """
    Fail when the native exporter reports success without writing its model.
    """

    def fake_export(**kwargs):
        """
        Return the declared path without creating the ONNX artifact.
        """
        return Path(kwargs["output_dir"]) / f"{kwargs['output_name']}.onnx"

    model = SimpleNamespace(export=fake_export)
    with pytest.raises(FileNotFoundError, match="did not produce"):
        rfdetr.export_onnx_model(model, tmp_path, resolution=640)


def test_ensure_onnx_model_reuses_a_valid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Avoid checkpoint loading and export when the existing graph validates.
    """
    expected = tmp_path / rfdetr.BEST_ONNX_MODEL
    expected.write_bytes(b"valid")
    validated = []

    def fake_validate(path: Path) -> Path:
        """
        Record validation of the existing artifact.
        """
        validated.append(path)
        return path

    monkeypatch.setattr(rfdetr, "validate_onnx_model", fake_validate)
    monkeypatch.setattr(
        rfdetr,
        "load_best_model",
        lambda *_: pytest.fail("valid ONNX should not reload the checkpoint"),
    )

    assert rfdetr.ensure_onnx_model(tmp_path) == expected
    assert validated == [expected]


def test_ensure_onnx_model_atomically_replaces_an_invalid_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Keep the old path in place until a staged replacement passes validation.
    """
    expected = tmp_path / rfdetr.BEST_ONNX_MODEL
    expected.write_bytes(b"invalid")
    model = SimpleNamespace(model_config=SimpleNamespace(resolution=640))
    validated = []

    def fake_validate(path: Path) -> Path:
        """
        Reject the old artifact and accept the newly staged one.
        """
        validated.append(path)
        if path == expected:
            raise ValueError("invalid graph")
        assert path.parent != tmp_path
        assert path.read_bytes() == b"replacement"
        return path

    def fake_export(model_arg, output_dir: Path, resolution: int) -> Path:
        """
        Write a replacement into the worker's temporary directory.
        """
        assert model_arg is model
        assert resolution == 640
        staged = output_dir / rfdetr.BEST_ONNX_MODEL
        staged.write_bytes(b"replacement")
        assert expected.read_bytes() == b"invalid"
        return staged

    monkeypatch.setattr(rfdetr, "validate_onnx_model", fake_validate)
    monkeypatch.setattr(rfdetr, "load_best_model", lambda *_: (model, {}))
    monkeypatch.setattr(rfdetr, "export_onnx_model", fake_export)

    assert rfdetr.ensure_onnx_model(tmp_path) == expected
    assert expected.read_bytes() == b"replacement"
    assert validated[0] == expected
    assert validated[1].name == rfdetr.BEST_ONNX_MODEL


def test_ensure_onnx_model_preserves_existing_file_after_staging_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Leave an existing artifact untouched when its replacement fails validation.
    """
    expected = tmp_path / rfdetr.BEST_ONNX_MODEL
    expected.write_bytes(b"old")
    model = SimpleNamespace(model_config=SimpleNamespace(resolution=640))

    def fake_validate(path: Path) -> Path:
        """
        Reject both the existing and staged fixture artifacts.
        """
        raise ValueError(f"invalid graph: {path}")

    def fake_export(*_args) -> Path:
        """
        Write a staged artifact that will fail validation.
        """
        output_dir = _args[1]
        staged = output_dir / rfdetr.BEST_ONNX_MODEL
        staged.write_bytes(b"bad replacement")
        return staged

    monkeypatch.setattr(rfdetr, "validate_onnx_model", fake_validate)
    monkeypatch.setattr(rfdetr, "load_best_model", lambda *_: (model, {}))
    monkeypatch.setattr(rfdetr, "export_onnx_model", fake_export)

    with pytest.raises(ValueError, match="invalid graph"):
        rfdetr.ensure_onnx_model(tmp_path)
    assert expected.read_bytes() == b"old"


def test_ensure_onnx_model_in_subprocess_uses_active_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Launch the standalone worker with the notebook kernel's Python executable.
    """
    project_root = tmp_path / "project"
    run_dir = project_root / "outputs" / "run"
    worker = project_root / "scripts" / "export_rfdetr_onnx.py"
    worker.parent.mkdir(parents=True)
    worker.touch()
    run_dir.mkdir(parents=True)
    calls = []

    def fake_run(command, **kwargs) -> None:
        """
        Record the worker command and emulate its successful artifact creation.
        """
        calls.append((command, kwargs))
        (run_dir / rfdetr.BEST_ONNX_MODEL).touch()

    monkeypatch.setattr(rfdetr.subprocess, "run", fake_run)

    expected = rfdetr.ensure_onnx_model_in_subprocess(project_root, run_dir)
    assert expected == run_dir / rfdetr.BEST_ONNX_MODEL
    assert calls == [
        (
            [
                sys.executable,
                "-I",
                str(worker),
                "--run-dir",
                str(run_dir),
            ],
            {"cwd": project_root, "check": True},
        )
    ]


def test_predictions_skip_background_and_convert_xyxy_to_coco() -> None:
    """
    Remove the explicit no-object slot while retaining basketball detections.
    """
    detections = SimpleNamespace(
        xyxy=[[1.0, 2.0, 11.0, 22.0], [0.0, 0.0, 3.0, 3.0]],
        confidence=[0.8, 0.7],
        class_id=[0, 1],
    )
    model = SimpleNamespace(predict=lambda *args, **kwargs: detections)
    rows = rfdetr.predictions_for_image(
        model, Image.new("RGB", (32, 32)), image_id=9, category_id=7
    )
    assert rows == [
        {
            "image_id": 9,
            "category_id": 7,
            "bbox": [1.0, 2.0, 10.0, 20.0],
            "score": 0.8,
        }
    ]


def _completed_protocol_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, rfdetr.TrainingSettings]:
    """
    Create a completed smoke run with mockable model-owned postprocessing inputs.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    settings = rfdetr.TrainingSettings(epochs=2, smoke_run=True, benchmark=False)
    monkeypatch.setattr(rfdetr, "_dataset_identity", lambda _manifest: _identity())
    write_json(run_dir / "dataset_manifest.json", {"source_dir": "source"})
    rfdetr.capture_dataset_identity(run_dir / "dataset_identity.json", _identity())
    rfdetr._verify_protocol(
        run_dir,
        _identity(),
        settings,
        datetime(2026, 9, 19, tzinfo=UTC),
    )
    attempt = start_training_attempt(
        run_dir,
        resumed=False,
        training_hardware={"device": "cuda:0"},
        monotonic_clock=lambda: 1.0,
    )
    finalize_training_attempt(
        run_dir,
        attempt,
        outcome="completed",
        completed_epochs=2,
        monotonic_clock=lambda: 3.0,
    )
    pd.DataFrame([{"epoch": 1, "train/loss": 1.0}]).to_csv(
        run_dir / "metrics.csv", index=False
    )
    (run_dir / rfdetr.BEST_CHECKPOINT).write_bytes(b"selected weights")
    return run_dir, settings


def test_fit_closes_synchronized_timing_before_postprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Time exactly model.train and finalize its record before callers can evaluate.
    """
    run_dir, settings = _completed_protocol_run(tmp_path, monkeypatch)
    (run_dir / "training.json").unlink()
    calls: list[str] = []
    clock = iter((10.0, 16.0))

    class FakeTrainingModel:
        """
        Provide the small native RF-DETR surface used by the fitting wrapper.
        """

        model_config = SimpleNamespace(model_dump=lambda **_kwargs: {})

        def get_train_config(self, **_kwargs: object) -> SimpleNamespace:
            """
            Return a serializable resolved training configuration.
            """
            return SimpleNamespace(model_dump=lambda **_kwargs: {})

        def train(self, **_kwargs: object) -> None:
            """
            Write native outputs while the shared attempt is still running.
            """
            calls.append("train")
            assert read_training_record(run_dir)["summary"]["status"] == "incomplete"
            pd.DataFrame([{"epoch": 0, "train/loss": 1.0}]).to_csv(
                run_dir / "metrics.csv", index=False
            )
            (run_dir / rfdetr.BEST_CHECKPOINT).write_bytes(b"selected weights")

    history = rfdetr.fit_model(
        FakeTrainingModel(),
        settings,
        {
            "dataset_dir": str(tmp_path),
            "category_mapping": {"class_names": ["basketball"]},
        },
        run_dir,
        synchronize=lambda: calls.append("sync"),
        monotonic_clock=lambda: next(clock),
    )

    assert calls == ["sync", "train", "sync"]
    assert history["epoch"].tolist() == [1]
    assert read_training_record(run_dir)["summary"]["total_seconds"] == 6.0


def test_postprocess_failure_recovers_and_republishes_smoke_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Recover a successful fit after postprocessing fails without another attempt.
    """
    run_dir, _settings = _completed_protocol_run(tmp_path, monkeypatch)
    checkpoint = run_dir / rfdetr.BEST_CHECKPOINT

    class FakeFigure:
        """
        Create the expected plot paths without importing a GUI backend.
        """

        def savefig(self, path: Path, **_kwargs: object) -> None:
            """
            Materialize a tiny plot placeholder.
            """
            Path(path).touch()

    monkeypatch.setattr(rfdetr, "plot_history", lambda _history: (FakeFigure(), FakeFigure()))
    monkeypatch.setattr(
        rfdetr,
        "ensure_onnx_model_in_subprocess",
        lambda _project, path: path / rfdetr.BEST_ONNX_MODEL,
    )
    monkeypatch.setattr(
        rfdetr,
        "load_best_model",
        lambda _path: (
            SimpleNamespace(model_config=SimpleNamespace(resolution=640)),
            {"parameters": 1, "checkpoint_sha256": file_sha256(checkpoint)},
        ),
    )
    monkeypatch.setattr(
        rfdetr,
        "fit_model",
        lambda *_args, **_kwargs: pytest.fail("postprocessing must not fit"),
    )

    with pytest.raises(RuntimeError, match="postprocess failed"):
        monkeypatch.setattr(
            rfdetr,
            "evaluate_split",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("postprocess failed")
            ),
        )
        rfdetr.postprocess_run(run_dir, bundle_root=tmp_path / "bundle")
    assert len(read_training_record(run_dir)["attempts"]) == 1

    def fake_evaluate(
        _model: object,
        _manifest: dict,
        _run_dir: Path,
        split: str,
        _best_metadata: dict,
        **_kwargs: object,
    ) -> tuple[dict, Path]:
        """
        Write a minimal shared prediction artifact for each recovered split.
        """
        annotation = tmp_path / f"{split}.json"
        annotation.write_text(
            '{"images":[{"id":1,"file_name":"image.jpg"}],'
            '"annotations":[],"categories":[{"id":1,"name":"basketball"}]}',
            encoding="utf-8",
        )
        artifact = write_prediction_artifact(
            run_dir / f"{split}_predictions.json",
            annotation,
            [],
            metadata={
                "model": rfdetr.MODEL_NAME,
                "split": split,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "smoke_run": True,
                "resolution": 640,
                "checkpoint_sha256": file_sha256(checkpoint),
                "postprocessing": {
                    "score_floor": 0.001,
                    "precision": "float32",
                },
            },
        )
        return {}, artifact

    monkeypatch.setattr(rfdetr, "evaluate_split", fake_evaluate)
    first = rfdetr.postprocess_run(run_dir, bundle_root=tmp_path / "bundle")
    second = rfdetr.postprocess_run(run_dir, bundle_root=tmp_path / "bundle")

    assert len(read_training_record(run_dir)["attempts"]) == 1
    assert first["bundle"]["provenance"]["smoke_run"] is True
    assert read_bundle(tmp_path / "bundle", allow_smoke=True)["manifest"] == second["bundle"]


def test_multiclass_predictions_keep_second_class_and_skip_background() -> None:
    """Label one is a real class; only label num_classes is the no-object slot."""
    detections = SimpleNamespace(
        xyxy=[[0.0, 0.0, 2.0, 2.0]] * 3,
        confidence=[0.9, 0.8, 0.1],
        class_id=[0, 1, 2],
    )
    model = SimpleNamespace(predict=lambda *args, **kwargs: detections)
    rows = rfdetr.predictions_for_image(model, None, 42, [7, 19])
    assert [row["category_id"] for row in rows] == [7, 19]
    assert all(row["image_id"] == 42 for row in rows)


def test_multiclass_model_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build a two-output head without downloading pretrained weights."""
    module = ModuleType("rfdetr")
    module.RFDETRSmall = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "rfdetr", module)
    model = rfdetr.build_model(
        rfdetr.TrainingSettings(), class_names=["football", "basketball"]
    )
    assert model["num_classes"] == 2
    kwargs = rfdetr.train_kwargs(
        rfdetr.TrainingSettings(),
        Path("dataset"),
        Path("run"),
        class_names=["football", "basketball"],
    )
    assert kwargs["class_names"] == ["football", "basketball"]
