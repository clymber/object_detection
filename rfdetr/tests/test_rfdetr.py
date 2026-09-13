"""
Test RF-DETR run configuration, recovery, history, and prediction conversion.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest
import torch
from PIL import Image

from detection_common.utils.json_io import read_json, write_json
from rfdetr_pipeline import rfdetr


def _run_config(path: Path, settings: rfdetr.TrainingSettings) -> None:
    """
    Write the saved settings needed by evaluation and checkpoint recovery.
    """
    values = vars(settings).copy()
    values.pop("mode")
    values.pop("run_dir")
    write_json(path / rfdetr.RUN_CONFIG, {"settings": values})


def test_settings_support_smoke_and_saved_evaluation_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Bound a new smoke run and inherit an existing run before explicit overrides.
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

    run_dir = tmp_path / "saved"
    run_dir.mkdir()
    saved = rfdetr.TrainingSettings(epochs=17, batch_size=2)
    _run_config(run_dir, saved)
    evaluation = rfdetr.settings_from_env(
        overrides={"mode": "evaluate", "run_dir": str(run_dir)},
    )
    assert evaluation.epochs == 17
    assert evaluation.batch_size == 2
    assert evaluation.mode is rfdetr.RunMode.EVALUATE
    assert evaluation.run_dir == str(run_dir.resolve())


def test_legacy_run_retains_disabled_early_stopping(tmp_path: Path) -> None:
    """
    Keep runs created before the new policy evaluable without changing identity.
    """
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    legacy = rfdetr.TrainingSettings(
        early_stopping=False,
        early_stopping_use_ema=False,
    )
    values = vars(legacy).copy()
    values.pop("mode")
    values.pop("run_dir")
    for name in rfdetr.LEGACY_EARLY_STOPPING:
        values.pop(name)
    write_json(
        run_dir / rfdetr.RUN_CONFIG,
        {
            "settings": values,
            "dataset_fingerprint": "fixture",
            "model": "rfdetr_small",
            "rfdetr_version": rfdetr.RFDETR_VERSION,
        },
    )
    (run_dir / rfdetr.BEST_CHECKPOINT).touch()

    settings = rfdetr.settings_from_env(
        overrides={"mode": "evaluate", "run_dir": str(run_dir)},
    )
    assert settings.early_stopping is False
    assert settings.early_stopping_use_ema is False
    assert (
        rfdetr.prepare_run(
            settings,
            {"fingerprint": "fixture", "source_dir": "fixture"},
            {},
        )
        == run_dir
    )


def test_fresh_run_uses_configured_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Place new RF-DETR runs below OUTPUT_ROOT, including mounted overrides.
    """
    output_root = tmp_path / "mounted-output"
    monkeypatch.setattr(rfdetr, "OUTPUT_ROOT", output_root)
    settings = rfdetr.TrainingSettings(smoke_run=True)
    manifest = {"fingerprint": "fixture", "source_dir": "fixture"}
    run_dir = rfdetr.prepare_run(settings, manifest, {})
    assert run_dir.parent == output_root / "runs" / "basketball"
    (run_dir / rfdetr.BEST_CHECKPOINT).touch()
    evaluation = rfdetr.settings_from_env(
        overrides={
            "mode": "evaluate",
            "run_dir": str(run_dir.relative_to(output_root)),
        }
    )
    assert evaluation.run_dir == str(run_dir)


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
