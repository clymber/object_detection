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
from dataset_builder import Dataset as VersionedDataset
from detection_common.utils.digest import file_digest
from detection_common.utils.json_io import read_json, write_json
from detection_evaluation import (
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


@pytest.mark.parametrize("smoke_run", [False, True])
def test_prepare_dataset_derives_inventory_and_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smoke_run: bool
) -> None:
    """Prepare arbitrary split sizes and category IDs without experiment assertions."""
    data_root = tmp_path / "data"
    monkeypatch.setattr(rfdetr, "DATA_ROOT", data_root)
    monkeypatch.setattr("dataset_builder.identity.DATA_ROOT", data_root)
    dataset = VersionedDataset(data_root / "composed" / "widgets-v2")
    (dataset.root / "coco/annotations").mkdir(parents=True)
    counts = {"train": 17, "val": 9, "test": 10}
    for split_index, (split, count) in enumerate(counts.items()):
        images = []
        image_dir = dataset.root / "coco/images" / split
        image_dir.mkdir(parents=True)
        for image_id in range(1, count + 1):
            file_name = f"{image_id}.png"
            Image.new("RGB", (16, 16), (split_index * 50, image_id, 30)).save(
                image_dir / file_name
            )
            images.append(
                {"id": image_id, "file_name": file_name, "width": 16, "height": 16}
            )
        write_json(
            dataset.root / "coco/annotations" / f"instances_{split}.json",
            {
                "images": images,
                "categories": [
                    {"id": 7, "name": "widget"},
                    {"id": 19, "name": "gizmo"},
                ],
                "annotations": [
                    {
                        "id": index,
                        "image_id": 1,
                        "category_id": category_id,
                        "bbox": [index, index, 4, 4],
                        "area": 16,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                    for index, category_id in enumerate((7, 19), start=1)
                ],
            },
        )

    manifest, summary = rfdetr.prepare_dataset(
        dataset, rfdetr.TrainingSettings(epochs=2, smoke_run=smoke_run)
    )

    assert summary["dataset_version"] == "widgets-v2"
    assert summary["splits"] == {
        split: {"images": count, "annotations": 2, "backgrounds": count - 1}
        for split, count in counts.items()
    }
    assert manifest["category_mapping"]["class_names"] == ["widget", "gizmo"]
    assert manifest["category_mapping"]["prediction_to_source"] == {"0": 7, "1": 19}
    loader_counts = {"train": 16, "val": 8, "test": 8} if smoke_run else counts
    for split, count in loader_counts.items():
        assert manifest["splits"][split]["images"] == count
        assert manifest["splits"][split]["annotations"] == 2
        assert manifest["splits"][split]["negative_images"] == count - 1
    assert not (dataset.root / "yolo").exists()


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


def test_resume_of_a_preprotocol_run_requires_a_fresh_experiment(
    tmp_path: Path,
) -> None:
    """
    Reject historical runs rather than attempting an unsafe protocol conversion.
    """
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="fresh experiment"):
        rfdetr.settings_from_env(
            overrides={"mode": "resume", "run_dir": str(run_dir)},
        )


@pytest.mark.parametrize("model_size", ["small", "nano"])
def test_fresh_run_uses_configured_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_size: str
) -> None:
    """
    Place new RF-DETR runs below OUTPUT_ROOT, including mounted overrides.
    """
    output_root = tmp_path / "mounted-output"
    monkeypatch.setattr(rfdetr, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(rfdetr, "_dataset_identity", lambda _manifest: _identity())
    settings = rfdetr.TrainingSettings(epochs=2, smoke_run=True, model_size=model_size)
    manifest = {"fingerprint": "fixture", "source_dir": "fixture"}
    run_dir = rfdetr.prepare_run(settings, manifest, {})
    assert run_dir.parent == output_root / "runs" / "basketball"
    assert run_dir.name.startswith(f"rfdetr_{model_size}_")
    assert read_json(run_dir / rfdetr.RUN_CONFIG)["model"] == settings.model_name
    protocol = read_json(run_dir / "run_protocol.json")
    assert protocol["smoke_run"] is True
    assert protocol["model"] == settings.model_name
    assert protocol["source_notebook"] == rfdetr.SOURCE_NOTEBOOKS[settings.model_name]
    resumed = rfdetr.settings_from_env(
        overrides={"mode": "resume", "run_dir": str(run_dir)}
    )
    assert resumed.model_size == model_size
    assert resumed.smoke_run is True
    assert rfdetr._protocol_settings(resumed) == protocol["training_settings"]


def test_model_size_environment_and_notebook_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Parse a variant from the shell while allowing the Nano notebook to pin it.
    """
    monkeypatch.setenv("RFDETR_MODEL_SIZE", "small")
    assert rfdetr.settings_from_env().model_size == "small"
    nano = rfdetr.settings_from_env(overrides={"model_size": "nano"})
    assert nano.model_size == "nano"
    monkeypatch.setenv("RFDETR_MODEL_SIZE", "nano")
    assert rfdetr.settings_from_env().model_size == "nano"


def test_existing_small_protocol_preserves_original_notebook_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Keep historical Small protocols resumable after adding the Nano variant.
    """
    monkeypatch.setattr(
        rfdetr,
        "SOURCE_NOTEBOOKS",
        {
            **rfdetr.SOURCE_NOTEBOOKS,
            "rfdetr_small": "nb04.02-rfdetr_small_large_basketball.ipynb",
        },
    )
    settings = rfdetr.TrainingSettings()
    original_utc = datetime(2026, 9, 19, tzinfo=UTC)
    original = rfdetr._verify_protocol(tmp_path, _identity(), settings, original_utc)
    assert "model_size" not in original["training_settings"]
    monkeypatch.setattr(
        rfdetr,
        "SOURCE_NOTEBOOKS",
        {
            **rfdetr.SOURCE_NOTEBOOKS,
            "rfdetr_small": "nb04.02-rfdetr_small-basketball.ipynb",
        },
    )
    assert rfdetr._verify_protocol(tmp_path, _identity(), settings, original_utc) == original
    recovered = rfdetr._settings_from_protocol(original, tmp_path)
    assert recovered.model_size == "small"
    with pytest.raises(ValueError, match="Immutable protocol differs"):
        rfdetr._verify_protocol(
            tmp_path,
            _identity(),
            rfdetr.TrainingSettings(model_size="nano"),
            original_utc,
        )


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
        {"model_size": "large"},
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
    assert kwargs["lr_scheduler_kwargs"] == {"lr_drop": 100}
    assert "lr_drop" not in kwargs
    assert "do_random_resize_via_padding" not in kwargs


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


@pytest.mark.parametrize("model_size", ["small", "nano"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize(
    ("total_prefix", "source_prefix"),
    [("", ""), ("_orig_mod.", "_orig_mod."), ("_orig_mod.", ""), ("", "_orig_mod.")],
)
def test_best_checkpoint_recovers_stripped_metadata_and_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    total_prefix: str,
    source_prefix: str,
    device: str,
    model_size: str,
) -> None:
    """
    Reload compiled or eager selected weights without changing the saved checkpoints.
    """
    settings = rfdetr.TrainingSettings(model_size=model_size)
    _run_config(tmp_path, settings)
    if model_size == "small":
        # Existing run configs predate the explicit variant setting.
        saved = read_json(tmp_path / rfdetr.RUN_CONFIG)
        saved["settings"].pop("model_size")
        write_json(tmp_path / rfdetr.RUN_CONFIG, saved)
    bias = torch.tensor([1.0, 2.0])
    total = tmp_path / rfdetr.BEST_CHECKPOINT
    source = tmp_path / "checkpoint_best_ema.pth"
    torch.save(
        {
            "model": {f"{total_prefix}class_embed.bias": bias},
            "args": {"class_names": ["basketball"]},
            "best_total_source": "ema",
        },
        total,
    )
    torch.save(
        {
            "model": {f"{source_prefix}class_embed.bias": bias},
            "epoch": 6,
            "model_config": {"resolution": 640, "num_select": 125},
        },
        source,
    )
    original_digests = {path: file_digest(path) for path in (total, source)}
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
            state = torch.load(path, map_location="cpu", weights_only=False)
            assert torch.equal(state["model"]["class_embed.bias"], bias)
            assert state["args"]["class_names"] == ["basketball"]
            assert state["best_total_source"] == "ema"
            return SimpleNamespace(
                class_names=["basketball"],
                model_config=SimpleNamespace(num_classes=1),
                model=SimpleNamespace(model=torch.nn.Linear(1, 1)),
            )

    module = ModuleType("rfdetr")
    setattr(module, f"RFDETR{model_size.capitalize()}", FakeRFDETRSmall)
    monkeypatch.setitem(sys.modules, "rfdetr", module)
    if device == "cpu":
        _, metadata = rfdetr.load_best_model(tmp_path, device=device)
    else:
        _, metadata = rfdetr.load_best_model(tmp_path)
    assert metadata["best_epoch"] == 7
    assert metadata["selected_weights"] == "ema"
    assert metadata["source_checkpoint"] == str(source)
    assert metadata["checkpoint"] == str(total)
    assert metadata["checkpoint_sha256"] == original_digests[total]
    assert {path: file_digest(path) for path in (total, source)} == original_digests
    if total_prefix:
        assert calls[0][0] != total
        assert not calls[0][0].exists()
    else:
        assert calls[0][0] == total
    assert calls[0][1]["num_select"] == 125
    assert calls[0][1]["resolution"] == 640
    assert calls[0][1]["positional_encoding_size"] == 40
    assert calls[0][1]["trust_checkpoint"] is True
    assert calls[0][1]["device"] == device
    assert calls[0][1]["amp"] is False
    assert calls[0][1]["compile"] is False
    assert read_json(tmp_path / "best_checkpoint.json") == metadata


@pytest.mark.parametrize("prefix", ["", "_orig_mod."])
def test_best_checkpoint_rejects_a_mismatched_selected_source(
    tmp_path: Path, prefix: str
) -> None:
    """
    Stop if the promoted total weights do not equal their named source checkpoint.
    """
    _run_config(tmp_path, rfdetr.TrainingSettings())
    torch.save(
        {
            "model": {f"{prefix}w": torch.tensor([1.0])},
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


@pytest.mark.parametrize("conflicting_checkpoint", ["total", "source"])
def test_best_checkpoint_rejects_conflicting_compiled_weight_names(
    tmp_path: Path, conflicting_checkpoint: str
) -> None:
    """Reject ambiguous names rather than silently replacing a trained tensor."""
    _run_config(tmp_path, rfdetr.TrainingSettings())
    states = {
        "total": {
            "model": {"w": torch.tensor([1.0])},
            "best_total_source": "ema",
        },
        "source": {
            "model": {"w": torch.tensor([1.0])},
            "epoch": 0,
            "model_config": {"resolution": 640},
        },
    }
    states[conflicting_checkpoint]["model"]["_orig_mod.w"] = torch.tensor([2.0])
    torch.save(states["total"], tmp_path / rfdetr.BEST_CHECKPOINT)
    torch.save(states["source"], tmp_path / "checkpoint_best_ema.pth")

    with pytest.raises(ValueError, match="conflicting compiled and uncompiled"):
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
    expected = tmp_path / "weights" / rfdetr.BEST_ONNX_MODEL
    expected.parent.mkdir(parents=True, exist_ok=True)
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
    expected = tmp_path / "weights" / rfdetr.BEST_ONNX_MODEL
    expected.parent.mkdir(parents=True, exist_ok=True)
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
    expected = tmp_path / "weights" / rfdetr.BEST_ONNX_MODEL
    expected.parent.mkdir(parents=True, exist_ok=True)
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
        (run_dir / "weights").mkdir(exist_ok=True)
        (run_dir / "weights" / rfdetr.BEST_ONNX_MODEL).touch()

    monkeypatch.setattr(rfdetr.subprocess, "run", fake_run)

    expected = rfdetr.ensure_onnx_model_in_subprocess(project_root, run_dir)
    assert expected == run_dir / "weights" / rfdetr.BEST_ONNX_MODEL
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, model_size: str = "small"
) -> tuple[Path, rfdetr.TrainingSettings]:
    """
    Create a completed smoke run with mockable model-owned postprocessing inputs.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    settings = rfdetr.TrainingSettings(
        epochs=2, smoke_run=True, benchmark=False, model_size=model_size
    )
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


@pytest.mark.parametrize("model_size", ["small", "nano"])
@pytest.mark.parametrize("publish", [False, True])
def test_postprocess_failure_recovers_with_optional_bundle_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publish: bool, model_size: str
) -> None:
    """
    Recover all run outputs without another fit, publishing only when requested.
    """
    run_dir, settings = _completed_protocol_run(
        tmp_path, monkeypatch, model_size=model_size
    )
    assert rfdetr._postprocess_state(run_dir)[2].model_size == model_size
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

    monkeypatch.setattr(
        rfdetr, "plot_history", lambda _history: (FakeFigure(), FakeFigure())
    )
    monkeypatch.setattr(
        rfdetr,
        "ensure_onnx_model_in_subprocess",
        lambda _project, path: path / rfdetr.BEST_ONNX_MODEL,
    )
    monkeypatch.setattr(
        rfdetr,
        "load_best_model",
        lambda _path, **_kwargs: (
            SimpleNamespace(model_config=SimpleNamespace(resolution=640)),
            {"parameters": 1, "checkpoint_sha256": file_digest(checkpoint)},
        ),
    )
    monkeypatch.setattr(
        rfdetr,
        "fit_model",
        lambda *_args, **_kwargs: pytest.fail("postprocessing must not fit"),
    )
    options = {"bundle_root": tmp_path / "bundle"} if publish else {"publish": False}
    if not publish:
        monkeypatch.setattr(
            rfdetr,
            "publish_bundle",
            lambda *_args, **_kwargs: pytest.fail("notebook must not publish a bundle"),
        )

    with pytest.raises(RuntimeError, match="postprocess failed"):
        monkeypatch.setattr(
            rfdetr,
            "evaluate_split",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("postprocess failed")
            ),
        )
        rfdetr.postprocess_run(run_dir, **options)
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
        write_json(
            annotation,
            {
                "images": [{"id": 1, "file_name": "image.jpg"}],
                "annotations": [],
                "categories": [{"id": 1, "name": "basketball"}],
            },
        )
        artifact = write_prediction_artifact(
            run_dir / f"{split}_predictions.json",
            annotation,
            [],
            metadata={
                "model": settings.model_name,
                "split": split,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "smoke_run": True,
                "resolution": 640,
                "checkpoint_sha256": file_digest(checkpoint),
                "postprocessing": {
                    "score_floor": 0.001,
                    "precision": "float32",
                },
            },
        )
        return {}, artifact

    monkeypatch.setattr(rfdetr, "evaluate_split", fake_evaluate)
    first = rfdetr.postprocess_run(run_dir, **options)
    second = rfdetr.postprocess_run(run_dir, **options)

    assert len(read_training_record(run_dir)["attempts"]) == 1
    assert first["split_metrics"] == second["split_metrics"] == {"val": {}, "test": {}}
    assert first["prediction_artifacts"] == second["prediction_artifacts"]
    if publish:
        assert first["bundle"]["provenance"]["smoke_run"] is True
        assert (
            read_bundle(tmp_path / "bundle", allow_smoke=True)["manifest"]
            == second["bundle"]
        )
    else:
        assert first["bundle"] is second["bundle"] is None
        assert not (tmp_path / "bundle").exists()


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


@pytest.mark.parametrize("model_size", ["small", "nano"])
@pytest.mark.parametrize("cuda_available", [False, True])
def test_multiclass_model_construction(
    monkeypatch: pytest.MonkeyPatch, cuda_available: bool, model_size: str
) -> None:
    """
    Build the selected variant's class head and enable compilation only with CUDA.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    module = ModuleType("rfdetr")
    setattr(module, f"RFDETR{model_size.capitalize()}", lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "rfdetr", module)
    model = rfdetr.build_model(
        rfdetr.TrainingSettings(model_size=model_size),
        class_names=["football", "basketball"],
    )
    assert model["num_classes"] == 2
    assert model["compile"] is cuda_available
    assert model["resolution"] == 640
    assert model["positional_encoding_size"] == 40
    kwargs = rfdetr.train_kwargs(
        rfdetr.TrainingSettings(),
        Path("dataset"),
        Path("run"),
        class_names=["football", "basketball"],
    )
    assert kwargs["class_names"] == ["football", "basketball"]


def test_checkpoint_report_never_calls_exporter(tmp_path, monkeypatch):
    """
    Produce training checkpoint reports through a path with no export side effect.
    """
    settings = rfdetr.TrainingSettings(benchmark=False)
    monkeypatch.setattr(rfdetr, "_postprocess_state",
                        lambda _: ({}, {}, settings))
    monkeypatch.setattr(rfdetr, "read_training_history", lambda _: "history")
    figure = SimpleNamespace(savefig=lambda *a, **k: None)
    monkeypatch.setattr(rfdetr, "plot_history", lambda _: (figure, figure))
    monkeypatch.setattr(rfdetr, "load_best_model", lambda *a, **k: ("model", {}))
    monkeypatch.setattr(rfdetr, "evaluate_split", lambda *a, **k: ({}, tmp_path))

    def forbidden(*args, **kwargs):
        """
        Reject any attempt to fit or export from checkpoint reporting.
        """
        pytest.fail("Checkpoint reporting must not fit or export")

    monkeypatch.setattr(rfdetr, "ensure_onnx_model_in_subprocess", forbidden)
    monkeypatch.setattr(rfdetr, "export_onnx_model", forbidden)
    monkeypatch.setattr(rfdetr, "fit_model", forbidden)
    result = rfdetr.evaluate_checkpoint_run(tmp_path, device="cpu")
    assert result["history"] == "history"
    assert set(result["split_metrics"]) == {"val", "test"}
    assert "onnx_path" not in result


def test_archived_paths_resolve_locally_without_rewriting_provenance(
        tmp_path, monkeypatch):
    """
    Recover an archived run's layout in memory and preserve its saved Linux paths.
    """
    from dataset_builder import rfdetr as dataset_rfdetr

    run = tmp_path / "run"
    run.mkdir()
    settings = rfdetr.TrainingSettings()
    original = datetime(2026, 11, 5, tzinfo=UTC)
    protocol = rfdetr._verify_protocol(run, _identity(), settings, original)
    write_json(run / "dataset_identity.json", _identity())
    saved = {"source_dir": "/archived/linux/coco", "dataset_dir": "/archived/linux/rfdetr",
             "mode": "full", "fingerprint": "same-layout"}
    write_json(run / "dataset_manifest.json", saved)
    local = tmp_path / "data/composed/basketball-v0.1/coco"
    local.mkdir(parents=True)
    monkeypatch.setattr(rfdetr, "DATA_ROOT", tmp_path / "data")
    expected = {**saved, "source_dir": str(local), "dataset_dir": str(tmp_path / "derived")}
    monkeypatch.setattr(dataset_rfdetr, "prepare_coco_dataset", lambda *a, **k: expected)
    monkeypatch.setattr(rfdetr, "_dataset_identity", lambda manifest: _identity())
    monkeypatch.setattr(rfdetr, "verify_dataset_identity", lambda *a: None)
    monkeypatch.setattr(rfdetr, "read_training_record", lambda _: {
        "attempts": [{"status": "completed"}]})
    digest = file_digest(run / "dataset_manifest.json")
    actual_protocol, manifest, actual_settings = rfdetr._postprocess_state(run)
    assert actual_protocol == protocol and manifest == expected
    assert actual_settings.model_size == "small"
    assert file_digest(run / "dataset_manifest.json") == digest
    assert read_json(run / "dataset_manifest.json")["source_dir"] == saved["source_dir"]


def test_existing_root_onnx_moves_into_weights_without_reexport(tmp_path, monkeypatch):
    """
    Consolidate a legacy graph under weights without loading or exporting again.
    """
    legacy = tmp_path / rfdetr.BEST_ONNX_MODEL
    legacy.write_bytes(b"valid historical graph")
    monkeypatch.setattr(rfdetr, "validate_onnx_model", lambda path: path)
    monkeypatch.setattr(rfdetr, "load_best_model",
                        lambda *a: pytest.fail("valid legacy graph must not reload"))
    path = rfdetr.ensure_onnx_model(tmp_path)
    assert path == tmp_path / "weights" / rfdetr.BEST_ONNX_MODEL
    assert path.read_bytes() == b"valid historical graph"
    assert not legacy.exists()


def test_onnx_recovery_replaces_link_target_without_duplicate(tmp_path, monkeypatch):
    """
    Repair the canonical artifact while preserving the uniform weights link.
    """
    canonical = tmp_path / "exports/onnx/checkpoint_best_total.onnx"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"invalid")
    alias = tmp_path / "weights/checkpoint_best_total.onnx"
    alias.parent.mkdir()
    alias.symlink_to(canonical)
    model = SimpleNamespace(model_config=SimpleNamespace(resolution=640))

    def validate(path):
        """
        Accept only the staged replacement graph.
        """
        if path.read_bytes() != b"replacement":
            raise ValueError("invalid graph")
        return path

    def export(_model, directory, resolution):
        """
        Write a replacement into the helper's temporary directory.
        """
        staged = directory / rfdetr.BEST_ONNX_MODEL
        staged.write_bytes(b"replacement")
        return staged

    monkeypatch.setattr(rfdetr, "load_best_model", lambda _: (model, {}))
    monkeypatch.setattr(rfdetr, "validate_onnx_model", validate)
    monkeypatch.setattr(rfdetr, "export_onnx_model", export)
    assert rfdetr.ensure_onnx_model(tmp_path) == alias
    assert alias.is_symlink()
    assert canonical.read_bytes() == b"replacement"
