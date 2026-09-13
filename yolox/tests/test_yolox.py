from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest
import torch

from yolox_pipeline import yolox


class TinyHead(torch.nn.Module):
    """
    Minimal head carrying the YOLOX L1 phase flag.
    """

    def __init__(self) -> None:
        """
        Initialize the head with L1 loss disabled.
        """
        super().__init__()
        self.use_l1 = False


class TinyModel(torch.nn.Module):
    """
    Minimal trainable model compatible with the recovery helpers.
    """

    def __init__(self) -> None:
        """
        Initialize one parameter and a phase-aware head.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.head = TinyHead()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """
        Scale a tensor by the model's single parameter.
        """
        return value * self.weight


class TinyExportHead(TinyHead):
    """
    Minimal YOLOX head carrying the deploy-time decoding flag.
    """

    def __init__(self) -> None:
        """
        Initialize the head with inference decoding enabled.
        """
        super().__init__()
        self.decode_in_inference = True


class TinyExportModel(TinyModel):
    """
    Minimal model containing the SiLU module replaced during ONNX export.
    """

    def __init__(self) -> None:
        """
        Initialize the tiny model with an export-sensitive activation.
        """
        super().__init__()
        self.head = TinyExportHead()
        self.activation = torch.nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """
        Scale and activate an input tensor.
        """
        return self.activation(super().forward(value))


class FakeExportExp:
    """
    Small experiment object exposing the model and ONNX test input size.
    """

    test_size = (24, 32)

    def get_model(self) -> TinyExportModel:
        """
        Return the experiment's cached tiny export model.
        """
        if not hasattr(self, "model"):
            self.model = TinyExportModel()
        return self.model


class FakeScheduler:
    """
    Deterministic scheduler used to verify restored positions.
    """

    def update_lr(self, position: int) -> float:
        """
        Return a distinct learning rate for each scheduler position.
        """
        return 0.01 * (position + 1)


class FakeLoader:
    """
    Minimal loader exposing length and mosaic phase control.
    """

    def __init__(self) -> None:
        """
        Initialize with mosaic augmentation enabled.
        """
        self.mosaic_closed = False

    def __len__(self) -> int:
        """
        Return one training iteration per epoch.
        """
        return 1

    def close_mosaic(self) -> None:
        """
        Record that mosaic augmentation was disabled.
        """
        self.mosaic_closed = True


class FakeExp:
    """
    Small experiment object implementing the trainer's required API.
    """

    def __init__(self) -> None:
        """
        Configure a two-epoch run with a one-epoch final phase.
        """
        self.max_epoch = 2
        self.no_aug_epochs = 1
        self.ema = True
        self.seed = 42
        self.basic_lr_per_img = 0.1

    def get_model(self) -> TinyModel:
        """
        Return the experiment's cached tiny model.
        """
        if not hasattr(self, "model"):
            self.model = TinyModel()
        return self.model

    def get_optimizer(self, batch_size: int) -> torch.optim.Optimizer:
        """
        Return an optimizer over the cached model.
        """
        del batch_size
        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.SGD(
                self.get_model().parameters(),
                lr=0.5,
                momentum=0.9,
            )
        return self.optimizer


def make_settings(resume_run_dir: Path | None = None) -> yolox.TrainingSettings:
    """
    Build lightweight settings for recovery tests.
    """
    return yolox.TrainingSettings(
        epochs=2,
        batch_size=2,
        train_batch_limit=None,
        image_size=32,
        seed=42,
        smoke_run=True,
        show_progress=False,
        verbose_output=False,
        resume_run_dir=resume_run_dir,
    )


def test_onnx_export_uses_fresh_cpu_ema_model_and_validates_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Export EMA weights with YOLOX deploy settings and validate the saved graph.
    """
    checkpoint_path = tmp_path / "weights" / "best_ckpt.pth"
    checkpoint_path.parent.mkdir()
    raw_model = TinyExportModel()
    ema_model = TinyExportModel()
    with torch.no_grad():
        raw_model.weight.fill_(2.0)
        ema_model.weight.fill_(3.0)
    torch.save(
        {
            "model": raw_model.state_dict(),
            "ema_model": ema_model.state_dict(),
        },
        checkpoint_path,
    )

    captured: dict[str, Any] = {}

    def fake_export(
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        path: Path,
        **kwargs: Any,
    ) -> None:
        """
        Capture the public ONNX export call and create its expected artifact.
        """
        captured["model"] = model
        captured["dummy_input"] = dummy_input
        captured["export_path"] = path
        captured["export_kwargs"] = kwargs
        Path(path).write_bytes(b"mock ONNX graph")

    checked_model = object()

    def fake_load(path: Path) -> object:
        """
        Capture the ONNX artifact loaded for validation.
        """
        captured["loaded_path"] = path
        return checked_model

    def fake_check_model(model: object) -> None:
        """
        Capture the model passed to the ONNX checker.
        """
        captured["checked_model"] = model

    fake_onnx: Any = ModuleType("onnx")
    fake_checker: Any = ModuleType("onnx.checker")
    fake_onnx.load = fake_load
    fake_checker.check_model = fake_check_model
    fake_onnx.checker = fake_checker
    monkeypatch.setitem(sys.modules, "onnx", fake_onnx)
    monkeypatch.setattr(torch.onnx, "export", fake_export)

    exp = FakeExportExp()
    stale_model = exp.get_model()
    output_path = tmp_path / "exports" / "best_ckpt.onnx"

    exported_path = yolox.export_trained_model_to_onnx(
        exp,
        checkpoint_path,
        output_path,
    )

    export_model = captured["model"]
    assert exported_path == output_path
    assert captured["export_path"] == output_path
    assert captured["loaded_path"] == output_path
    assert captured["checked_model"] is checked_model
    assert export_model is not stale_model
    assert export_model.weight.item() == pytest.approx(3.0)
    assert next(export_model.parameters()).device.type == "cpu"
    assert not export_model.training
    assert not export_model.head.decode_in_inference
    assert type(export_model.activation) is yolox.YOLOXSiLU
    assert captured["dummy_input"].shape == (1, 3, 24, 32)
    assert captured["dummy_input"].device.type == "cpu"
    assert captured["export_kwargs"] == {
        "input_names": ["images"],
        "output_names": ["output"],
        "opset_version": 11,
        "dynamo": False,
    }
    assert not hasattr(exp, "model")


def test_onnx_export_rejects_a_non_onnx_output_path(tmp_path: Path) -> None:
    """
    Reject an incorrectly named artifact before loading a checkpoint.
    """
    with pytest.raises(ValueError, match=r"must end with \.onnx"):
        yolox.export_trained_model_to_onnx(
            FakeExportExp(),
            tmp_path / "missing_checkpoint.pth",
            tmp_path / "best_ckpt.pt",
        )


def test_training_settings_read_explicit_resume_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Read the explicit resume directory without changing fresh-run defaults.
    """
    monkeypatch.setattr(yolox, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setenv("YOLOX_TINY_RESUME_RUN", "runs/existing")

    settings = yolox.training_settings_from_env()

    assert settings.resume_run_dir == Path("runs/existing")
    assert settings.run_mode is yolox.RunMode.RESUME
    assert settings.resolved_resume_run_dir == tmp_path / "outputs/runs/existing"


def test_training_settings_default_to_fresh_run() -> None:
    """
    Derive fresh mode when no resume directory is configured.
    """
    assert make_settings().run_mode is yolox.RunMode.FRESH


def test_nano_training_settings_use_an_independent_environment_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Read Nano controls without inheriting values intended for Tiny runs.
    """
    monkeypatch.setenv("YOLOX_TINY_EPOCHS", "99")
    monkeypatch.setenv("YOLOX_NANO_EPOCHS", "12")
    monkeypatch.setenv("YOLOX_NANO_BATCH_SIZE", "4")
    monkeypatch.setenv("YOLOX_NANO_SMOKE", "0")

    settings = yolox.training_settings_from_env(env_prefix="YOLOX_NANO")

    assert settings.epochs == 12
    assert settings.batch_size == 4
    assert not settings.smoke_run


def test_basketball_nano_uses_official_nano_model_geometry(tmp_path: Path) -> None:
    """
    Configure Nano with its narrower depthwise architecture and shared image size.
    """
    exp = yolox.BasketballNanoExp(
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "outputs",
        max_epoch=100,
        image_size=640,
        project_name="nano-test",
    )

    assert exp.depth == pytest.approx(0.33)
    assert exp.width == pytest.approx(0.25)
    assert exp.depthwise
    assert exp.input_size == (640, 640)
    assert exp.test_size == (640, 640)
    assert yolox.YOLOX_NANO_WEIGHTS_URL.endswith("/yolox_nano.pth")

    model = exp.get_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count < 1_000_000


def test_checkpoint_restores_complete_training_state(tmp_path: Path) -> None:
    """
    Restore raw, EMA, optimizer, scheduler, history, and phase state exactly.
    """
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    ema_model = yolox.ModelEMA(model, 0.9998)
    with torch.no_grad():
        model.weight.fill_(3.0)
    loss = model(torch.ones(1)).sum()
    loss.backward()
    optimizer.step()
    ema_model.update(model)
    checkpoint_path = tmp_path / "weights" / "last_ckpt.pth"
    checkpoint_path.parent.mkdir()
    history = [{"epoch": 1.0, "metrics/mAP50-95(B)": 0.25}]
    pending_metrics = {"train/total_loss": 1.5, "lr": 0.07}

    yolox.save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        2,
        0.25,
        0.25,
        ema_model=ema_model,
        scheduler_position=6,
        history=history,
        mosaic_enabled=False,
        l1_enabled=True,
        validation_pending=True,
        pending_train_metrics=pending_metrics,
        iters_per_epoch=3,
        max_epoch=4,
        batch_size=2,
        image_size=32,
    )

    restored_model = TinyModel()
    restored_optimizer = torch.optim.SGD(
        restored_model.parameters(),
        lr=0.9,
        momentum=0.9,
    )
    restored_ema = yolox.ModelEMA(restored_model, 0.9998)
    state = yolox.restore_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        restored_ema,
        FakeScheduler(),
        torch.device("cpu"),
        iters_per_epoch=3,
        max_epoch=4,
        batch_size=2,
        image_size=32,
    )

    assert torch.equal(restored_model.weight, model.weight)
    assert torch.equal(restored_ema.ema.weight, ema_model.ema.weight)
    assert restored_ema.updates == ema_model.updates
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.07)
    assert state.completed_epoch == 2
    assert state.scheduler_position == 6
    assert state.history == history
    assert state.pending_train_metrics == pending_metrics
    assert not state.mosaic_enabled
    assert state.l1_enabled
    assert state.validation_pending
    assert not any(path.suffix == ".tmp" for path in checkpoint_path.parent.iterdir())


def test_resume_completes_pending_validation_before_next_epoch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Resume pending validation without duplicating or skipping history epochs.
    """
    observations: list[tuple[int, bool, bool, float]] = []
    interrupt_validation = {"enabled": True}

    def fake_write_run_metadata(*args: Any, **kwargs: Any) -> None:
        """
        Avoid experiment-specific metadata requirements in the focused test.
        """
        del args, kwargs

    def fake_build_train_loader(*args: Any, **kwargs: Any) -> FakeLoader:
        """
        Return a fresh phase-aware loader for each trainer invocation.
        """
        del args, kwargs
        return FakeLoader()

    def fake_load_pretrained_weights(
        model: torch.nn.Module,
        checkpoint_path: Path,
    ) -> torch.nn.Module:
        """
        Keep the tiny model's deterministic initial weights.
        """
        del checkpoint_path
        return model

    def fake_make_lr_scheduler(*args: Any, **kwargs: Any) -> FakeScheduler:
        """
        Return the deterministic scheduler.
        """
        del args, kwargs
        return FakeScheduler()

    def fake_train_one_epoch(
        model: TinyModel,
        optimizer: torch.optim.Optimizer,
        train_loader: FakeLoader,
        lr_scheduler: FakeScheduler,
        epoch_index: int,
        iters_per_epoch: int,
        device: torch.device,
        ema_model: yolox.ModelEMA | None,
        progress: bool,
    ) -> dict[str, float]:
        """
        Record restored phase and LR before advancing one synthetic epoch.
        """
        del device, progress
        observations.append(
            (
                epoch_index,
                train_loader.mosaic_closed,
                model.head.use_l1,
                optimizer.param_groups[0]["lr"],
            )
        )
        with torch.no_grad():
            model.weight.add_(1.0)
        if ema_model is not None:
            ema_model.update(model)
        position = (epoch_index + 1) * iters_per_epoch
        learning_rate = lr_scheduler.update_lr(position)
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
        return {"train/total_loss": float(epoch_index + 1), "lr": learning_rate}

    def fake_evaluate_training_epoch(
        eval_model: torch.nn.Module,
        exp: FakeExp,
        settings: yolox.TrainingSettings,
        device: torch.device,
        epoch: int,
        train_metrics: dict[str, float],
    ) -> dict[str, float]:
        """
        Interrupt once, then return a complete deterministic history record.
        """
        del eval_model, exp, settings, device
        if interrupt_validation["enabled"]:
            interrupt_validation["enabled"] = False
            raise RuntimeError("simulated validation interruption")
        return {
            "epoch": float(epoch),
            **train_metrics,
            "val/total_loss": 0.5,
            "metrics/precision(B)": 0.8,
            "metrics/recall(B)": 0.7,
            "metrics/mAP50(B)": 0.6,
            "metrics/mAP50-95(B)": 0.2 if epoch == 1 else 0.1,
            "speed/forward_ms": 1.0,
            "speed/nms_ms": 0.5,
        }

    monkeypatch.setattr(yolox, "write_run_metadata", fake_write_run_metadata)
    monkeypatch.setattr(yolox, "build_train_loader", fake_build_train_loader)
    monkeypatch.setattr(yolox, "load_pretrained_weights", fake_load_pretrained_weights)
    monkeypatch.setattr(yolox, "make_lr_scheduler", fake_make_lr_scheduler)
    monkeypatch.setattr(yolox, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(
        yolox,
        "evaluate_training_epoch",
        fake_evaluate_training_epoch,
    )

    output_dir = tmp_path / "run"
    settings = make_settings()
    fit_args = (
        tmp_path / "pretrained.pth",
        output_dir,
        tmp_path / "dataset",
        settings,
        torch.device("cpu"),
        tmp_path,
    )

    with pytest.raises(RuntimeError, match="simulated validation interruption"):
        yolox.fit_yolox_tiny(FakeExp(), *fit_args)

    pending = torch.load(output_dir / "weights" / "last_ckpt.pth")
    assert pending["completed_epoch"] == 1
    assert pending["validation_pending"]
    assert pending["history"] == []

    history = yolox.fit_yolox_tiny(FakeExp(), *fit_args, resume=True)
    saved_history = pd.read_csv(output_dir / "results.csv")
    final_checkpoint = torch.load(output_dir / "weights" / "last_ckpt.pth")
    best_checkpoint = torch.load(output_dir / "weights" / "best_ckpt.pth")

    assert history["epoch"].tolist() == [1.0, 2.0]
    assert saved_history["epoch"].tolist() == [1.0, 2.0]
    assert {
        "time/train_s",
        "time/val_s",
        "time/epoch_s",
        "time/cumulative_s",
    }.issubset(saved_history.columns)
    assert (saved_history[["time/train_s", "time/val_s"]] >= 0).all().all()
    assert saved_history["time/epoch_s"].tolist() == pytest.approx(
        (saved_history["time/train_s"] + saved_history["time/val_s"]).tolist()
    )
    assert saved_history["time/cumulative_s"].tolist() == pytest.approx(
        saved_history["time/epoch_s"].cumsum().tolist()
    )
    assert observations == [
        (0, False, False, 0.5),
        (1, True, True, pytest.approx(0.02)),
    ]
    assert final_checkpoint["scheduler_position"] == 2
    assert final_checkpoint["ema_updates"] == 2
    assert not final_checkpoint["mosaic_enabled"]
    assert final_checkpoint["l1_enabled"]
    assert not final_checkpoint["validation_pending"]
    assert [record["epoch"] for record in final_checkpoint["history"]] == [1.0, 2.0]
    assert final_checkpoint["best_ap"] == pytest.approx(0.2)
    assert final_checkpoint["curr_ap"] == pytest.approx(0.1)
    assert best_checkpoint["completed_epoch"] == 1
