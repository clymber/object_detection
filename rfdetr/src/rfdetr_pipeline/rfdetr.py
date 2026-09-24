"""
RF-DETR Small training, recovery, and reporting for the Renku basketball experiment.

Detector imports are lazy so data/evaluation helpers work outside its GPU environment.
"""

from __future__ import annotations

import gc
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

from detection_evaluation import (
    benchmark_predict,
    capture_training_hardware,
    create_run_protocol,
    evaluate_predictions,
    finalize_training_attempt,
    file_sha256,
    publish_bundle,
    read_run_protocol,
    read_training_record,
    start_training_attempt,
    write_prediction_artifact,
)
from detection_common import allocate_run_directory
from detection_common.utils.json_io import read_json, write_json
from dataset_builder import (
    canonical_coco_identity,
    capture_dataset_identity,
    validate_rfdetr_layout,
    verify_dataset_identity,
)

from .config import OUTPUT_ROOT, SUBPROJECT_ROOT

RFDETR_VERSION = "1.10.1"
RUN_CONFIG = "run_config.json"
BEST_CHECKPOINT = "checkpoint_best_total.pth"
BEST_ONNX_MODEL = "checkpoint_best_total.onnx"
RESUME_CHECKPOINT = "last.ckpt"
class RunMode(StrEnum):
    """
    Identify whether an RF-DETR run is created or resumed.
    """

    FRESH = "fresh"
    RESUME = "resume"

    @classmethod
    def parse(cls, value: Any) -> RunMode:
        """
        Normalize a string or enum member and report the supported values.
        """
        try:
            return cls(value)
        except (TypeError, ValueError) as error:
            if value == "evaluate":
                raise ValueError(
                    "RFDETR_MODE=evaluate was removed; use python -m "
                    "rfdetr_pipeline.postprocess_cli --run-dir <run>"
                ) from error
            options = ", ".join(option.value for option in cls)
            raise ValueError(f"RFDETR_MODE must be {options}") from error


@dataclass(frozen=True)
class TrainingSettings:
    """
    Explicit, serializable settings for one single-GPU experiment.
    """

    epochs: int = 100
    batch_size: int = 4
    grad_accum_steps: int = 1
    resolution: int = 640
    seed: int = 42
    num_workers: int = 0
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    amp: bool = True
    gradient_checkpointing: bool = False
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = True
    device: str = "cuda:0"
    benchmark: bool = True
    smoke_run: bool = False
    mode: RunMode = RunMode.FRESH
    run_dir: str | None = None

    def __post_init__(self) -> None:
        """
        Reject unsupported resolutions, run modes, and invalid numeric settings.
        """
        for name in (
            "epochs",
            "batch_size",
            "grad_accum_steps",
            "resolution",
            "early_stopping_patience",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.resolution % 32:
            raise ValueError("RF-DETR Small resolution must be a multiple of 32")
        for name in ("num_workers", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("lr", "lr_encoder"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.early_stopping_min_delta)
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError("early_stopping_min_delta must be finite and nonnegative")
        for name in (
            "amp",
            "gradient_checkpointing",
            "early_stopping",
            "early_stopping_use_ema",
            "benchmark",
            "smoke_run",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.device != "cuda:0":
            raise ValueError("RF-DETR requires device cuda:0")
        object.__setattr__(self, "mode", RunMode.parse(self.mode))
        if self.smoke_run and self.epochs != 2:
            raise ValueError("RF-DETR smoke runs require exactly two epochs")
        if self.mode is RunMode.RESUME and not self.run_dir:
            raise ValueError("RFDETR_RUN_DIR is required for resume")
        if self.mode is RunMode.FRESH and self.run_dir:
            raise ValueError("Use resume to open an existing RFDETR_RUN_DIR")

    @property
    def samples_per_optimizer_step(self) -> int:
        """
        Return samples accumulated per optimizer step on one GPU.
        """
        return self.batch_size * self.grad_accum_steps


def _settings_from_protocol(protocol: Mapping[str, Any], run_dir: Path) -> TrainingSettings:
    """
    Reconstruct model settings from immutable provenance for a resumed run.
    """
    return TrainingSettings(
        **dict(protocol["training_settings"]),
        smoke_run=protocol["smoke_run"],
        mode=RunMode.RESUME,
        run_dir=str(run_dir),
    )


def settings_from_env(*, overrides: dict[str, Any] | None = None) -> TrainingSettings:
    """
    Read explicit overrides, inheriting immutable settings for a resumed run.
    """
    overrides = overrides or {}
    mode = RunMode.parse(
        overrides.get("mode", os.environ.get("RFDETR_MODE", RunMode.FRESH))
    )
    run_dir = overrides.get("run_dir", os.environ.get("RFDETR_RUN_DIR")) or None
    values: dict[str, Any] = {}
    if run_dir:
        path = Path(run_dir).expanduser()
        if not path.is_absolute():
            parts = path.parts[1:] if path.parts[:1] == ("outputs",) else path.parts
            path = OUTPUT_ROOT.joinpath(*parts)
        path = path.resolve()
        run_dir = str(path)
        if mode is RunMode.RESUME:
            try:
                protocol = read_run_protocol(path)
            except FileNotFoundError as error:
                raise ValueError(
                    "This RF-DETR run predates the shared protocol; start a fresh "
                    "experiment for comparison."
                ) from error
            values = dict(protocol["training_settings"])
            values["smoke_run"] = protocol["smoke_run"]
    boolean_fields = {
        "amp",
        "gradient_checkpointing",
        "early_stopping",
        "early_stopping_use_ema",
        "benchmark",
        "smoke_run",
    }
    float_fields = {"lr", "lr_encoder", "early_stopping_min_delta"}
    for field in TrainingSettings.__dataclass_fields__:
        if field in {"mode", "run_dir"}:
            continue
        key = "RFDETR_SMOKE" if field == "smoke_run" else f"RFDETR_{field.upper()}"
        if key in os.environ:
            value = os.environ[key]
            if field in boolean_fields:
                if value not in {"0", "1"}:
                    raise ValueError(f"{key} must be 0 or 1")
                values[field] = value == "1"
            else:
                values[field] = float(value) if field in float_fields else int(value)
    values.update(
        {key: value for key, value in overrides.items() if key not in {"mode", "run_dir"}}
    )
    if mode is RunMode.FRESH and values.get("smoke_run"):
        values.setdefault("epochs", 2)
    return TrainingSettings(**{**values, "mode": mode, "run_dir": run_dir})


def configure_caches(project_root: Path) -> dict[str, str]:
    """
    Set project-owned cache directories before RF-DETR or Hugging Face imports.
    """
    settings = {
        "RF_HOME": project_root / "models" / "pretrained" / "rfdetr",
        "HF_HOME": project_root / "models" / "cache" / "rfdetr" / "huggingface",
        "TORCH_HOME": project_root / "models" / "cache" / "rfdetr" / "torch",
    }
    for key, path in settings.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path.resolve())
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return {key: str(path) for key, path in settings.items()}


def runtime_report() -> dict:
    """
    Verify the pinned detector environment and allocated CUDA GPU before execution.
    """
    import torch

    if version("rfdetr") != RFDETR_VERSION:
        raise RuntimeError(
            f"Install rfdetr=={RFDETR_VERSION} in the active environment"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a Renku session with a CUDA GPU")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This notebook is configured for one GPU/process")
    return {
        "python": sys.version,
        "interpreter": sys.executable,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "packages": {
            name: version(name)
            for name in (
                "rfdetr",
                "torch",
                "torchvision",
                "pytorch-lightning",
                "transformers",
                "numpy",
                "pandas",
                "pycocotools",
                "onnx",
                "onnxruntime",
            )
        },
    }


def train_kwargs(
    settings: TrainingSettings,
    dataset_dir: Path,
    run_dir: Path,
    *,
    class_names: list[str] | tuple[str, ...] = ("basketball",),
) -> dict:
    """
    Build the RF-DETR 1.10.1 training configuration with validation-only selection.
    """
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_file": "roboflow",
        "output_dir": str(run_dir.resolve()),
        # RF-DETR 1.10.1 maps an indexed device to ``devices=[0]`, but its
        # trainer helper only accepts an integer or string. ``cuda`` still
        # selects the first visible GPU and keeps the supported scalar form.
        "device": "cuda",
        "epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "eval_batch_size": settings.batch_size,
        "grad_accum_steps": settings.grad_accum_steps,
        "resolution": settings.resolution,
        "seed": settings.seed,
        "num_workers": settings.num_workers,
        "lr": settings.lr,
        "lr_encoder": settings.lr_encoder,
        "optimizer": "adamw",
        "weight_decay": 1e-4,
        "lr_scheduler": "step",
        "lr_drop": settings.epochs,
        "multi_scale": False,
        "expanded_scales": False,
        "scale_jitter": False,
        "do_random_resize_via_padding": False,
        "augmentation_backend": "torchvision",
        "use_ema": True,
        "best_model_metric": "map",
        "early_stopping": settings.early_stopping,
        "early_stopping_patience": settings.early_stopping_patience,
        "early_stopping_min_delta": settings.early_stopping_min_delta,
        "early_stopping_use_ema": settings.early_stopping_use_ema,
        "run_test": False,
        "eval_base_model": True,
        "compute_val_loss": True,
        "eval_interval": 1,
        "eval_max_dets": 100,
        "checkpoint_interval": 10,
        "class_names": list(class_names),
        "tensorboard": False,
        "wandb": False,
        "mlflow": False,
        "progress_bar": "tqdm",
        "devices": 1,
        "num_nodes": 1,
        "amp_dtype": "fp16",
        "num_sanity_val_steps": 0,
    }


LOGICAL_DATASET = "basketball"
MODEL_NAME = "rfdetr_small"
SOURCE_NOTEBOOK = "nb04.02-rfdetr_small_large_basketball.ipynb"


def _dataset_identity(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Capture canonical COCO provenance and the validated RF-DETR loader layout.
    """
    source_dir = Path(manifest["source_dir"])
    dataset_dir = Path(manifest["dataset_dir"])
    return {
        "canonical": canonical_coco_identity(source_dir),
        "loader": validate_rfdetr_layout(source_dir, dataset_dir),
    }


def _protocol_settings(settings: TrainingSettings) -> dict[str, Any]:
    """
    Return immutable controls that define an RF-DETR producer run.
    """
    values = asdict(settings)
    values.pop("mode")
    values.pop("run_dir")
    values.pop("smoke_run")
    return values


def _verify_protocol(
    run_dir: Path,
    identity: Mapping[str, Any],
    settings: TrainingSettings,
    original_utc: datetime | str,
) -> dict[str, Any]:
    """
    Persist or verify immutable shared provenance for one RF-DETR run.
    """
    return create_run_protocol(
        run_dir,
        logical_dataset=LOGICAL_DATASET,
        model=MODEL_NAME,
        source_notebook=SOURCE_NOTEBOOK,
        original_utc=original_utc,
        training_settings=_protocol_settings(settings),
        smoke_run=settings.smoke_run,
        canonical_dataset_identity=identity["canonical"],
        loader_dataset_identity=identity["loader"],
    )


def prepare_run(settings: TrainingSettings, manifest: dict, runtime: dict) -> Path:
    """
    Allocate a fresh run or verify the identity/configuration of an existing run.
    """
    dataset_identity = _dataset_identity(manifest)
    experiment = asdict(settings)
    experiment.pop("mode")
    experiment.pop("run_dir")
    identity = {
        "schema_version": 1,
        "rfdetr_version": RFDETR_VERSION,
        "settings": experiment,
        "dataset_fingerprint": manifest["fingerprint"],
        "model": "rfdetr_small",
        "source_dataset": manifest["source_dir"],
    }
    if settings.mode is not RunMode.FRESH:
        run_dir = Path(settings.run_dir).resolve()
        try:
            protocol = read_run_protocol(run_dir)
        except FileNotFoundError as error:
            raise ValueError(
                "This RF-DETR run predates the shared protocol; start a fresh "
                "experiment for comparison."
            ) from error
        verify_dataset_identity(run_dir / "dataset_identity.json", dataset_identity)
        _verify_protocol(run_dir, dataset_identity, settings, protocol["original_utc"])
        checkpoint = RESUME_CHECKPOINT
        if not (run_dir / checkpoint).is_file():
            raise FileNotFoundError(
                f"Missing {settings.mode} checkpoint: {run_dir / checkpoint}"
            )
        validate_resume_checkpoint(run_dir / checkpoint, settings.epochs)
        return run_dir
    allocation = allocate_run_directory(OUTPUT_ROOT, "basketball", "rfdetr_small")
    run_dir = allocation.path
    identity["created_at"] = allocation.created_at.isoformat()
    write_json(run_dir / RUN_CONFIG, {**identity, "runtime": runtime})
    write_json(run_dir / "dataset_manifest.json", manifest)
    capture_dataset_identity(run_dir / "dataset_identity.json", dataset_identity)
    _verify_protocol(run_dir, dataset_identity, settings, allocation.created_at)
    return run_dir


def validate_resume_checkpoint(checkpoint: Path, epoch_budget: int) -> dict:
    """
    Require optimizer and scheduler state in a project-generated full checkpoint.
    """
    import torch

    # Only called for the explicitly chosen run's own Lightning checkpoint.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "epoch"):
        if key not in state or (key != "epoch" and not state[key]):
            raise ValueError(f"Full resume checkpoint is missing {key}")
    completed = int(state["epoch"]) + 1
    if completed >= epoch_budget:
        raise ValueError(
            "Run already reached its epoch budget; use python -m "
            "rfdetr_pipeline.postprocess_cli --run-dir <run>"
        )
    return {"completed_epochs": completed, "global_step": state.get("global_step")}


def build_model(
    settings: TrainingSettings,
    class_names: list[str] | tuple[str, ...] = ("basketball",),
) -> Any:
    """
    Initialize the Small model with one output per dataset class.
    """
    from pytorch_lightning import seed_everything
    from rfdetr import RFDETRSmall

    # Upstream seeds on_fit_start, after creating the random detection head.
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("Expected nonempty unique class names")
    seed_everything(settings.seed, workers=True)
    return RFDETRSmall(
        num_classes=len(class_names),
        resolution=settings.resolution,
        positional_encoding_size=settings.resolution // 16,
        device="cuda:0",
        amp=settings.amp,
        compile=False,
        gradient_checkpointing=settings.gradient_checkpointing,
    )


def verify_loader(model: Any, settings: TrainingSettings, manifest: dict) -> dict:
    """
    Verify the pinned loader's split IDs, label mapping, and negative-image handling.
    """
    from rfdetr.config import TrainConfig
    from rfdetr.training import RFDETRDataModule

    dataset_dir = Path(manifest["dataset_dir"])
    kwargs = train_kwargs(
        settings,
        dataset_dir,
        dataset_dir,
        class_names=manifest["category_mapping"]["class_names"],
    )
    for key in ("device", "resolution"):
        kwargs.pop(key)
    datamodule = RFDETRDataModule(model.model_config, TrainConfig(**kwargs))
    datamodule.setup("fit")
    datamodule.setup("test")
    results = {}
    for split, field in (
        ("train", "_dataset_train"),
        ("val", "_dataset_val"),
        ("test", "_dataset_test"),
    ):
        dataset = getattr(datamodule, field)
        summary = manifest["splits"][split]
        expected = {item["image_id"] for item in summary["images_identity"]}
        if len(dataset) != summary["images"] or set(dataset.ids) != expected:
            raise RuntimeError(f"RF-DETR loader changed the {split} image membership")
        training_categories = sorted(
            int(value) for value in manifest["category_mapping"]["training_to_source"]
        )
        if dataset.cat2label != {
            value: index for index, value in enumerate(training_categories)
        }:
            raise RuntimeError(
                f"Unexpected class mapping in {split}: {dataset.cat2label}"
            )
        annotation_path = (
            dataset_dir / summary["loader_split"] / "_annotations.coco.json"
        )
        document = read_json(annotation_path)
        annotated = {item["image_id"] for item in document["annotations"]}
        negatives = expected - annotated
        if negatives:
            _, target = dataset[dataset.ids.index(min(negatives))]
            if len(target["labels"]) or len(target["boxes"]):
                raise RuntimeError(f"Negative-image target is not empty in {split}")
        results[split] = {"images": len(dataset), "negatives": len(negatives)}
    return results


def synchronize_cuda(device: str) -> None:
    """
    Synchronize the allocated CUDA device at a native training timing boundary.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _completed_epochs(run_dir: Path) -> int:
    """
    Return the final native epoch number without deriving timing from files.
    """
    try:
        return int(read_training_history(run_dir)["epoch"].max())
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
        return 0


def fit_model(
    model: Any,
    settings: TrainingSettings,
    manifest: dict,
    run_dir: Path,
    *,
    synchronize: Callable[[], None] | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> pd.DataFrame:
    """
    Train or resume, persisting resolved configuration and attempt duration on failure.
    """
    from pytorch_lightning import seed_everything

    kwargs = train_kwargs(
        settings,
        Path(manifest["dataset_dir"]),
        run_dir,
        class_names=manifest["category_mapping"]["class_names"],
    )
    if settings.mode is RunMode.RESUME:
        kwargs["resume"] = str(run_dir / RESUME_CHECKPOINT)
    config_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"device", "resolution"}
    }
    write_json(
        run_dir / "resolved_config.json",
        {
            "model_config": model.model_config.model_dump(mode="json"),
            "train_config": model.get_train_config(**config_kwargs).model_dump(
                mode="json"
            ),
            "samples_per_optimizer_step": settings.samples_per_optimizer_step,
            "preprocessing": "RF-DETR torchvision square resize; fixed scale",
        },
    )
    import torch

    before_epochs = _completed_epochs(run_dir)
    boundary = synchronize or (lambda: synchronize_cuda(settings.device))
    attempt_id = start_training_attempt(
        run_dir,
        completed_epochs_before=before_epochs,
        resumed=settings.mode is RunMode.RESUME,
        training_hardware=capture_training_hardware(
            device=settings.device,
            details={
                "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(settings.device)
                    if settings.device.startswith("cuda") and torch.cuda.is_available()
                    else None
                ),
            },
        ),
        monotonic_clock=monotonic_clock,
        synchronize=boundary,
    )
    outcome = "completed"
    try:
        # train() creates another model; loader verification has consumed RNG.
        seed_everything(settings.seed, workers=True)
        print(
            f"Starting RF-DETR {settings.mode} training for up to "
            f"{settings.epochs} epochs in {run_dir}. "
            "The progress bar refreshes every five batches.",
            flush=True,
        )
        model.train(**kwargs)
    except BaseException:
        outcome = "interrupted"
        raise
    finally:
        finalize_training_attempt(
            run_dir,
            attempt_id,
            outcome=outcome,
            completed_epochs=max(0, _completed_epochs(run_dir) - before_epochs),
            monotonic_clock=monotonic_clock,
            synchronize=boundary,
        )
    if not (run_dir / BEST_CHECKPOINT).is_file():
        raise FileNotFoundError("RF-DETR did not produce the expected best checkpoint")
    return read_training_history(run_dir)


def read_training_history(run_dir: Path) -> pd.DataFrame:
    """
    Combine sparse Lightning rows into one row per epoch and retain native logs.
    """
    raw = pd.read_csv(run_dir / "metrics.csv")
    raw.columns = raw.columns.str.strip()
    if "epoch" not in raw:
        raise ValueError("RF-DETR metrics.csv is missing the epoch column")
    raw = raw.dropna(subset=["epoch"])
    if raw.empty:
        raise ValueError("RF-DETR metrics.csv has no completed epoch records")
    history = raw.groupby("epoch", sort=True).last().reset_index()
    history["epoch"] = history["epoch"].astype(int) + 1
    history.to_csv(run_dir / "results.csv", index=False)
    return history


def plot_history(history: pd.DataFrame) -> tuple[Any, Any]:
    """
    Plot native detection loss components and available regular/EMA validation AP.
    """
    import matplotlib.pyplot as plt

    losses, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, component in zip(
        axes, ("loss", "loss_ce", "loss_bbox", "loss_giou"), strict=True
    ):
        for split in ("train", "val"):
            column = f"{split}/{component}"
            if column in history:
                axis.plot(history["epoch"], history[column], label=split)
        axis.set(title=component, xlabel="epoch")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "Not logged", ha="center", transform=axis.transAxes)
    metrics, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("mAP_50", "mAP_50_95"), strict=True):
        for prefix in ("val/", "val/ema_"):
            column = prefix + metric
            if column in history:
                axis.plot(history["epoch"], history[column], label=column)
        axis.set(title=metric, xlabel="epoch", ylim=(0, 1))
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    losses.tight_layout()
    metrics.tight_layout()
    return losses, metrics


def load_best_model(run_dir: Path) -> tuple[Any, dict]:
    """
    Reload actual selected weights and record their epoch and EMA/regular provenance.
    """
    import torch

    checkpoint = run_dir / BEST_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # The selected total file is stripped in 1.10.1: epoch and model_config
    # are lost. Recover provenance from its unstripped EMA/regular source and
    # explicitly restore our resolution instead of silently using Small's 512.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state.get("model"), dict):
        raise ValueError("Best checkpoint is missing model weights")
    selection = state.get("best_total_source")
    if selection not in {"ema", "regular"}:
        raise ValueError(
            "Best checkpoint is missing its EMA/regular selection metadata"
        )
    source_checkpoint = run_dir / f"checkpoint_best_{selection}.pth"
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(source.get("model"), dict):
        raise ValueError("Selected source checkpoint is missing model weights")
    if not isinstance(source.get("model_config"), dict):
        raise ValueError("Selected source checkpoint is missing model configuration")
    if "epoch" not in source:
        raise ValueError("Selected source checkpoint is missing its epoch")
    weights = state["model"]
    if weights.keys() != source["model"].keys() or any(
        not torch.equal(tensor, source["model"][key]) for key, tensor in weights.items()
    ):
        raise ValueError("Best checkpoint weights differ from its selected source")
    resolution = read_json(run_dir / RUN_CONFIG)["settings"]["resolution"]
    if source["model_config"]["resolution"] != resolution:
        raise ValueError("Best checkpoint resolution differs from the saved run")
    best_epoch = int(source["epoch"]) + 1
    del state, source, weights
    gc.collect()
    from rfdetr import RFDETRSmall

    model = RFDETRSmall.from_checkpoint(
        checkpoint,
        trust_checkpoint=True,
        device="cuda:0",
        amp=False,
        resolution=resolution,
        positional_encoding_size=resolution // 16,
        compile=False,
        gradient_checkpointing=False,
    )
    manifest_path = run_dir / "dataset_manifest.json"
    expected_names = (
        read_json(manifest_path)["category_mapping"]["class_names"]
        if manifest_path.is_file()
        else model.class_names
    )
    if model.class_names != expected_names or model.model_config.num_classes != len(
        expected_names
    ):
        raise ValueError(
            f"Reloaded detector has unexpected classes: {model.class_names}"
        )
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_checkpoint": str(source_checkpoint),
        "best_epoch": best_epoch,
        "selected_weights": selection,
        "selection_metric": "validation AP50:95",
        "parameters": sum(
            parameter.numel() for parameter in model.model.model.parameters()
        ),
    }
    write_json(run_dir / "best_checkpoint.json", metadata)
    return model, metadata


def export_onnx_model(model: Any, run_dir: Path, resolution: int) -> Path:
    """
    Export the verified best RF-DETR model as a static batch-one ONNX graph.
    """
    expected = run_dir / BEST_ONNX_MODEL
    exported = Path(
        model.export(
            format="onnx",
            output_dir=str(run_dir),
            output_name=Path(BEST_ONNX_MODEL).stem,
            shape=(resolution, resolution),
            batch_size=1,
            dynamic_batch=False,
            verbose=False,
        )
    )
    if exported.resolve() != expected.resolve():
        raise RuntimeError(f"RF-DETR exported ONNX to an unexpected path: {exported}")
    if not expected.is_file():
        raise FileNotFoundError(f"RF-DETR did not produce the ONNX model: {expected}")
    return expected


def validate_onnx_model(path: Path) -> Path:
    """
    Load an ONNX artifact and require it to pass the standard graph checker.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    import onnx

    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    return path


def ensure_onnx_model(run_dir: Path) -> Path:
    """
    Reuse a valid export or stage, validate, and atomically install a new one.
    """
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    expected = run_dir / BEST_ONNX_MODEL
    if expected.is_file():
        try:
            return validate_onnx_model(expected)
        except Exception as error:
            print(f"Replacing invalid ONNX model {expected}: {error}", flush=True)

    model, _ = load_best_model(run_dir)
    resolution = model.model_config.resolution
    with tempfile.TemporaryDirectory(
        prefix=".rfdetr-onnx-", dir=run_dir
    ) as temporary_dir:
        staged = export_onnx_model(model, Path(temporary_dir), resolution)
        validate_onnx_model(staged)
        os.replace(staged, expected)
    return expected


def ensure_onnx_model_in_subprocess(project_root: Path, run_dir: Path) -> Path:
    """
    Run RF-DETR ONNX export outside IPython and return its validated artifact.
    """
    project_root = project_root.resolve()
    run_dir = run_dir.resolve()
    worker = project_root / "scripts" / "export_rfdetr_onnx.py"
    if not worker.is_file():
        raise FileNotFoundError(worker)
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(worker),
            "--run-dir",
            str(run_dir),
        ],
        cwd=project_root,
        check=True,
    )
    expected = run_dir / BEST_ONNX_MODEL
    if not expected.is_file():
        raise FileNotFoundError(
            f"RF-DETR export worker did not produce the ONNX model: {expected}"
        )
    return expected


def _postprocess_state(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], TrainingSettings]:
    """
    Read and verify the new-protocol provenance required for recovery output.
    """
    run_dir = run_dir.expanduser().resolve()
    try:
        protocol = read_run_protocol(run_dir)
    except FileNotFoundError as error:
        raise ValueError(
            "This RF-DETR run predates the shared protocol; start a fresh "
            "experiment for comparison."
        ) from error
    manifest_path = run_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "This RF-DETR run has no persisted dataset manifest; start a fresh "
            "experiment for comparison."
        )
    manifest = read_json(manifest_path)
    identity = _dataset_identity(manifest)
    verify_dataset_identity(run_dir / "dataset_identity.json", identity)
    if protocol["dataset_identity"] != identity:
        raise ValueError("Run protocol dataset identity differs from the saved run")
    settings = TrainingSettings(
        **dict(protocol["training_settings"]),
        smoke_run=protocol["smoke_run"],
    )
    if _protocol_settings(settings) != protocol["training_settings"]:
        raise ValueError("Run protocol training settings cannot be reconstructed")
    training = read_training_record(run_dir)
    if (
        not training["attempts"]
        or any(a["status"] == "running" for a in training["attempts"])
        or training["attempts"][-1]["status"] != "completed"
    ):
        raise ValueError("Postprocessing requires a completed RF-DETR training run")
    return protocol, manifest, settings


def predictions_for_image(
    model: Any, image: Image.Image, image_id: int, category_id: int | list[int]
) -> list[dict]:
    """
    Convert RF-DETR's zero-based class labels and original-pixel xyxy boxes to COCO.
    """
    category_ids = [category_id] if isinstance(category_id, int) else category_id
    detections = model.predict(image, threshold=0.001, include_source_image=False)
    rows = []
    for box, score, label in zip(
        detections.xyxy, detections.confidence, detections.class_id, strict=True
    ):
        # 1.10.1 predict() can include its explicit no-object slot at low scores.
        if label == len(category_ids):
            continue
        if label != int(label) or not 0 <= int(label) < len(category_ids):
            raise ValueError("RF-DETR returned an unexpected class label")
        x1, y1, x2, y2 = map(float, box)
        rows.append(
            {
                "image_id": image_id,
                "category_id": category_ids[int(label)],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return rows


def evaluate_split(
    model: Any,
    manifest: dict,
    run_dir: Path,
    split: str,
    best_metadata: dict,
    *,
    settings: TrainingSettings,
    protocol: Mapping[str, Any],
) -> tuple[dict, Path]:
    """
    Evaluate an explicit frozen split and save native metrics and portable predictions.
    """
    if split not in {"val", "test"}:
        raise ValueError("Only val and test can be evaluated")
    summary = manifest["splits"][split]
    derived_dir = Path(manifest["dataset_dir"]) / summary["loader_split"]
    derived_annotations = derived_dir / "_annotations.coco.json"
    if not derived_annotations.is_file():
        raise FileNotFoundError(derived_annotations)
    # Smoke metrics use the explicitly bounded derived annotations. Full exports
    # use byte-identical original ground truth for comparison to the YOLO artifacts.
    annotation_path = (
        derived_annotations
        if manifest["mode"] == "smoke"
        else Path(manifest["source_dir"]) / "annotations" / f"instances_{split}.json"
    )
    ground_truth = read_json(annotation_path)
    mapping = manifest["category_mapping"]
    category_ids = [
        mapping["prediction_to_source"][str(index)]
        for index in range(len(mapping["class_names"]))
    ]
    if manifest["mode"] == "smoke":
        category_ids = [
            mapping["source_to_training"][str(value)] for value in category_ids
        ]
    predictions = []
    for image in ground_truth["images"]:
        with Image.open(derived_dir / image["file_name"]) as source:
            predictions.extend(
                predictions_for_image(
                    model, source.convert("RGB"), image["id"], category_ids
                )
            )
    metrics = evaluate_predictions(annotation_path, predictions)
    native = model.evaluate(
        dataset_dir=manifest["dataset_dir"],
        dataset_file="roboflow",
        split=split,
        # Match the supported scalar form used by train_kwargs for RF-DETR 1.10.1.
        device="cuda",
        devices=1,
        num_workers=0,
        batch_size=1,
        eval_max_dets=100,
        compute_val_loss=True,
        class_names=manifest["category_mapping"]["class_names"],
    )
    write_json(run_dir / f"native_{split}_metrics.json", native)
    write_json(run_dir / f"{split}_metrics.json", metrics)
    metadata = {
        "model": MODEL_NAME,
        "split": split,
        "run_dir": str(run_dir),
        **best_metadata,
        "resolution": model.model_config.resolution,
        "smoke_run": protocol["smoke_run"],
        "rfdetr_version": RFDETR_VERSION,
        "dataset_fingerprint": protocol["dataset_identity"]["canonical"][
            "source_fingerprint"
        ],
        "training_settings": protocol["training_settings"],
        "epochs_completed": int(read_training_history(run_dir)["epoch"].max()),
        "postprocessing": {
            "score_floor": 0.001,
            "num_select": model.model_config.num_select,
            "nms": None,
            "precision": "float32",
            "resize": "square",
        },
    }
    if settings.benchmark:

        def predict_one(image: Image.Image) -> list[dict]:
            """
            Execute the same PIL-to-CPU-predictions path used for detector exports.
            """
            return predictions_for_image(model, image, 0, category_ids)

        metadata["benchmark"] = benchmark_predict(
            predict_one,
            [derived_dir / image["file_name"] for image in ground_truth["images"]],
            device=settings.device,
        )
    artifact = write_prediction_artifact(
        run_dir / f"{split}_predictions.json",
        annotation_path,
        predictions,
        metadata=metadata,
    )
    save_prediction_grids(ground_truth, predictions, derived_dir, run_dir, split)
    return metrics, artifact


def postprocess_run(
    run_dir: Path,
    *,
    bundle_root: Path | None = None,
) -> dict[str, Any]:
    """
    Regenerate completed-run outputs and atomically publish both split artifacts.
    """
    run_dir = run_dir.expanduser().resolve()
    protocol, manifest, settings = _postprocess_state(run_dir)
    history = read_training_history(run_dir)
    loss_figure, metric_figure = plot_history(history)
    loss_figure.savefig(run_dir / "training_losses.png", bbox_inches="tight")
    metric_figure.savefig(run_dir / "validation_metrics.png", bbox_inches="tight")
    onnx_path = ensure_onnx_model_in_subprocess(SUBPROJECT_ROOT, run_dir)
    best_model, best_metadata = load_best_model(run_dir)
    split_metrics = {}
    prediction_artifacts = {}
    for split in ("val", "test"):
        metrics, artifact = evaluate_split(
            best_model,
            manifest,
            run_dir,
            split,
            best_metadata,
            settings=settings,
            protocol=protocol,
        )
        split_metrics[split] = metrics
        prediction_artifacts[split] = artifact
    destination = bundle_root or (
        OUTPUT_ROOT
        / "evaluation"
        / "basketball_large_dataset"
        / MODEL_NAME
        / run_dir.name
    )
    bundle = publish_bundle(
        destination,
        run_dir=run_dir,
        val_predictions=prediction_artifacts["val"],
        test_predictions=prediction_artifacts["test"],
        selected_checkpoint=run_dir / BEST_CHECKPOINT,
        resolution=settings.resolution,
        parameter_count=best_metadata["parameters"],
    )
    return {
        "history": history,
        "onnx_path": onnx_path,
        "best_metadata": best_metadata,
        "split_metrics": split_metrics,
        "prediction_artifacts": prediction_artifacts,
        "bundle": bundle,
    }


def save_prediction_grids(
    ground_truth: dict,
    predictions: list[dict],
    image_dir: Path,
    run_dir: Path,
    split: str,
) -> None:
    """
    Save fixed positive/small-object/negative image IDs with truth and prediction boxes.
    """
    annotations: dict[int, list[dict]] = defaultdict(list)
    detected: dict[int, list[dict]] = defaultdict(list)
    for item in ground_truth["annotations"]:
        annotations[item["image_id"]].append(item)
    for item in predictions:
        if item["score"] >= 0.25:
            detected[item["image_id"]].append(item)
    images = sorted(ground_truth["images"], key=lambda item: item["id"])
    positives = sorted(
        (item for item in images if annotations[item["id"]]),
        key=lambda item: (
            min(a["bbox"][2] * a["bbox"][3] for a in annotations[item["id"]]),
            item["id"],
        ),
    )
    negatives = [item for item in images if not annotations[item["id"]]]
    selected = positives[:4] + negatives[:4]
    selected_ids = {item["id"] for item in selected}
    selected += [item for item in images if item["id"] not in selected_ids][
        : 8 - len(selected)
    ]
    write_json(
        run_dir / f"{split}_sample_ids.json", {"image_ids": [i["id"] for i in selected]}
    )
    for kind, boxes in (("labels", annotations), ("pred", detected)):
        canvas = Image.new("RGB", (4 * 320, 2 * 350), "white")
        for index, entry in enumerate(selected):
            with Image.open(image_dir / entry["file_name"]) as original:
                tile = original.convert("RGB").resize((320, 320))
            draw = ImageDraw.Draw(tile)
            sx, sy = 320 / entry["width"], 320 / entry["height"]
            for box in boxes[entry["id"]]:
                x, y, width, height = box["bbox"]
                draw.rectangle(
                    (x * sx, y * sy, (x + width) * sx, (y + height) * sy),
                    outline="red" if kind == "pred" else "lime",
                    width=2,
                )
            x, y = index % 4 * 320, index // 4 * 350
            canvas.paste(tile, (x, y))
            ImageDraw.Draw(canvas).text(
                (x + 5, y + 325), f"ID {entry['id']}: {kind}", fill="black"
            )
        canvas.save(run_dir / f"{split}_batch0_{kind}.jpg")
