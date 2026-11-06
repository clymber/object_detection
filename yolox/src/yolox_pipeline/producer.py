"""
Run provenance, timing, and bundle publication for YOLOX detection runs.
"""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from dataset_builder import (
    Dataset as VersionedDataset,
)
from dataset_builder import (
    canonical_coco_identity,
    capture_dataset_identity,
    prepare_smoke_layouts,
    smoke_layout_directory,
    validate_coco_layout,
    verify_dataset_identity,
)
from detection_common import Device, allocate_run_directory
from detection_common.utils.digest import file_digest
from detection_common.utils.json_io import read_json
from detection_evaluation import (
    benchmark_predict,
    capture_training_hardware,
    create_run_protocol,
    finalize_training_attempt,
    publish_bundle,
    read_run_protocol,
    start_training_attempt,
)

from . import artifacts, yolox
from .config import OUTPUT_ROOT, WORKSPACE_ROOT

BEST_CHECKPOINT = Path("weights/best_ckpt.pth")


@dataclass(frozen=True)
class ModelVariant:
    """
    Describe the YOLOX model-specific construction required by the producer.
    """

    name: str
    source_notebook: str
    experiment_class: type[yolox.YOLOXTinyExp]
    fit: Callable[..., Any]
    logical_dataset: str = "basketball"


VARIANTS = {
    "tiny": ModelVariant(
        name="yolox_tiny",
        source_notebook="nb03.02-yolox_tiny_large_basketball.ipynb",
        experiment_class=yolox.YOLOXTinyExp,
        fit=yolox.fit_yolox_tiny,
    ),
    "nano": ModelVariant(
        name="yolox_nano",
        source_notebook="nb03.03-yolox_nano_large_basketball.ipynb",
        experiment_class=yolox.YOLOXNanoExp,
        fit=yolox.fit_yolox_nano,
    ),
}


@dataclass(frozen=True)
class DatasetPaths:
    """
    Canonical COCO source consumed directly by YOLOX.
    """

    source_dir: Path


@dataclass(frozen=True)
class ProducerSettings:
    """
    Wrap native training controls with producer-only publication settings.
    """

    training: yolox.TrainingSettings
    benchmark: bool = True

    def protocol_settings(self) -> dict[str, Any]:
        """
        Return immutable settings relevant to a reproducible training attempt.
        """
        return {
            "epochs": self.training.epochs,
            "batch_size": self.training.batch_size,
            "train_batch_limit": self.training.train_batch_limit,
            "image_size": self.training.image_size,
            "seed": self.training.seed,
            "benchmark": self.benchmark,
        }

    def validate(self) -> None:
        """
        Reject smoke settings that would reduce the complete Stage 2 layout.
        """
        if self.training.smoke_run and (
            self.training.epochs != 2 or self.training.train_batch_limit is not None
        ):
            raise ValueError(
                "YOLOX smoke runs require two epochs without train batch limits"
            )


def variant(
    name: str,
    *,
    logical_dataset: str = "basketball",
    source_notebook: str | None = None,
) -> ModelVariant:
    """
    Return the requested Tiny or Nano producer configuration.
    """
    try:
        model = VARIANTS[name]
    except KeyError as error:
        raise ValueError(f"Unsupported YOLOX variant: {name}") from error
    if logical_dataset != "basketball" and not source_notebook:
        raise ValueError("A custom dataset requires its source notebook")
    return replace(
        model,
        logical_dataset=logical_dataset,
        source_notebook=source_notebook or model.source_notebook,
    )


def settings_from_run(run_dir: Path) -> ProducerSettings:
    """
    Restore persisted training controls and smoke identity for resume or recovery.
    """
    protocol = read_run_protocol(run_dir)
    if protocol["model"] not in {item.name for item in VARIANTS.values()}:
        raise ValueError("Expected a YOLOX run protocol")
    values = dict(protocol["training_settings"])
    benchmark = values.pop("benchmark")
    return ProducerSettings(
        yolox.TrainingSettings(
            **values,
            smoke_run=protocol["smoke_run"],
            show_progress=True,
            verbose_output=False,
            resume_run_dir=run_dir.resolve(),
        ),
        benchmark=benchmark,
    )


def resolve_dataset_paths(
    source_dir: Path,
    *,
    smoke_run: bool,
) -> DatasetPaths:
    """
    Return full paths or materialize the complete deterministic smoke layouts.
    """
    source = source_dir.expanduser().resolve()
    if not smoke_run:
        return DatasetPaths(source)
    smoke_root = smoke_layout_directory(source)
    record = prepare_smoke_layouts(source, smoke_root, formats=("coco",))
    if record["smoke_limits"] != {"train": 16, "val": 8, "test": 8}:
        raise ValueError("Smoke dataset does not use the required 16/8/8 layout")
    return DatasetPaths(smoke_root / "coco")


def _dataset_identity(paths: DatasetPaths) -> dict[str, dict[str, Any]]:
    """
    Capture the canonical source and the direct COCO loader identity.
    """
    return {
        "canonical": canonical_coco_identity(paths.source_dir),
        "loader": validate_coco_layout(paths.source_dir),
    }


@dataclass(frozen=True)
class PreparedExperiment:
    """Resolved controls and paths for a notebook's existing producer calls."""

    run_dir: Path
    paths: DatasetPaths
    settings: ProducerSettings
    model: ModelVariant
    device: torch.device
    pretrained_path: Path
    original_utc: datetime | None
    summary: dict[str, Any]
    class_names: tuple[str, ...]


def evaluation_experiment(prepared: PreparedExperiment) -> yolox.YOLOXTinyExp:
    """Construct a fresh experiment with the same class mapping for every phase."""
    settings = prepared.settings.training
    return prepared.model.experiment_class(
        dataset_dir=prepared.paths.source_dir,
        output_dir=prepared.run_dir.parent,
        max_epoch=settings.epochs,
        image_size=settings.image_size,
        project_name=prepared.run_dir.name,
        class_names=prepared.class_names,
        seed=settings.seed,
    )


def prepare_experiment(
    dataset: VersionedDataset,
    *,
    model_name: str,
    class_names: tuple[str, ...],
    logical_dataset: str,
    source_notebook: str,
    env_prefix: str,
    resume_run_dir: str | None = None,
) -> PreparedExperiment:
    """Resolve training controls and paths for a versioned COCO dataset run."""
    summary = dataset.summary()
    key = f"{env_prefix}_RESUME_RUN"
    if resume_run_dir is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = resume_run_dir
    training = yolox.training_settings_from_env(
        default_epochs=100, env_prefix=env_prefix
    )
    settings = (
        settings_from_run(training.resolved_resume_run_dir)
        if training.run_mode is yolox.RunMode.RESUME
        else ProducerSettings(training)
    )
    training = settings.training
    settings.validate()
    paths = resolve_dataset_paths(dataset.coco, smoke_run=training.smoke_run)
    model = variant(
        model_name, logical_dataset=logical_dataset, source_notebook=source_notebook
    )
    original_utc = None
    if training.run_mode is yolox.RunMode.FRESH:
        allocation = allocate_run_directory(OUTPUT_ROOT, logical_dataset, model.name)
        run_dir, original_utc = allocation.path, allocation.created_at
    else:
        run_dir = training.resolved_resume_run_dir
        if not (run_dir / "weights/last_ckpt.pth").is_file():
            raise FileNotFoundError(f"Resume checkpoint not found in {run_dir}")
    prepare_run(run_dir, paths, settings, model, original_utc=original_utc)
    device = torch.device(Device.auto_choose())
    if device.type == "mps":
        yolox.patch_mps_compatibility()
    pretrained = WORKSPACE_ROOT / "models/pretrained/yolox" / f"{model.name}.pth"
    pretrained = yolox.ensure_pretrained_checkpoint(
        pretrained,
        yolox.YOLOX_TINY_WEIGHTS_URL
        if model_name == "tiny"
        else yolox.YOLOX_NANO_WEIGHTS_URL,
    )
    summary.update(
        loader_dir=str(paths.source_dir),
        run_dir=str(run_dir),
        device=str(device),
        training=settings.protocol_settings(),
        smoke_run=training.smoke_run,
    )
    return PreparedExperiment(
        run_dir,
        paths,
        settings,
        model,
        device,
        pretrained,
        original_utc,
        summary,
        class_names,
    )


def prepare_run(
    run_dir: Path,
    paths: DatasetPaths,
    settings: ProducerSettings,
    model: ModelVariant,
    *,
    original_utc: datetime | str | None = None,
) -> dict[str, Any]:
    """
    Persist or verify immutable dataset and run provenance before training.
    """
    settings.validate()
    run_dir = run_dir.expanduser().resolve()
    identity = _dataset_identity(paths)
    identity_path = run_dir / "dataset_identity.json"
    if identity_path.exists():
        saved = read_json(identity_path)
        if saved.get("loader", {}).get("loader", {}).get("format") == "yolo":
            raise ValueError(
                "Legacy YOLOX loader identity uses YOLO format; start a fresh COCO run"
            )
        verify_dataset_identity(identity_path, identity)
    else:
        capture_dataset_identity(identity_path, identity)
    if original_utc is None:
        original_utc = read_run_protocol(run_dir)["original_utc"]
    return create_run_protocol(
        run_dir,
        logical_dataset=model.logical_dataset,
        model=model.name,
        source_notebook=model.source_notebook,
        original_utc=original_utc,
        training_settings=settings.protocol_settings(),
        smoke_run=settings.training.smoke_run,
        canonical_dataset_identity=identity["canonical"],
        loader_dataset_identity=identity["loader"],
    )


def synchronize_cuda(device: torch.device) -> None:
    """
    Synchronize CUDA only when a CUDA device is available for the native fit.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _completed_epochs(run_dir: Path) -> int:
    """
    Count completed native history rows without deriving timing from timestamps.
    """
    history = run_dir / "results.csv"
    if not history.is_file():
        return 0
    with history.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def fit_run(
    run_dir: Path,
    paths: DatasetPaths,
    settings: ProducerSettings,
    model: ModelVariant,
    exp: yolox.YOLOXTinyExp,
    checkpoint_path: Path,
    device: torch.device,
    project_root: Path,
    *,
    original_utc: datetime | str | None = None,
    resumed: bool = False,
    fit: Callable[..., Any] | None = None,
    synchronize: Callable[[], None] | None = None,
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> Any:
    """
    Time exactly one native YOLOX fit call and finalize catchable outcomes.
    """
    run_dir = run_dir.expanduser().resolve()
    protocol = prepare_run(
        run_dir,
        paths,
        settings,
        model,
        original_utc=original_utc,
    )
    if fit is None:
        yolox.verify_coco_loaders(
            exp,
            tuple(
                category["name"]
                for category in protocol["dataset_identity"]["canonical"][
                    "canonical_source"
                ]["category_mapping"]
            ),
            source_dir=paths.source_dir,
        )
    before_epochs = _completed_epochs(run_dir)
    boundary = synchronize or (lambda: synchronize_cuda(device))
    attempt_id = start_training_attempt(
        run_dir,
        completed_epochs_before=before_epochs,
        resumed=resumed,
        training_hardware=capture_training_hardware(
            device=str(device),
            details={
                "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda,
                "gpu": (
                    torch.cuda.get_device_name(str(device))
                    if str(device).startswith("cuda") and torch.cuda.is_available()
                    else None
                ),
            },
        ),
        monotonic_clock=monotonic_clock,
        synchronize=boundary,
    )
    outcome = "completed"
    native_fit = fit or model.fit
    try:
        result = native_fit(
            exp,
            checkpoint_path,
            run_dir,
            paths.source_dir,
            settings.training,
            device,
            project_root,
            class_names=tuple(
                category["name"]
                for category in protocol["dataset_identity"]["canonical"][
                    "canonical_source"
                ]["category_mapping"]
            ),
            resume=resumed,
        )
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
    if protocol["smoke_run"] != settings.training.smoke_run:
        raise RuntimeError("Run protocol smoke state changed during training")
    return result


def _category_ids(protocol: Mapping[str, Any]) -> list[int]:
    """Return source IDs in the canonical model-label order."""
    categories = protocol["dataset_identity"]["canonical"]["canonical_source"][
        "category_mapping"
    ]
    ids = [category["id"] for category in categories]
    if (
        not ids
        or any(type(value) is not int for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Run protocol must describe unique integer category IDs")
    return ids


def _prediction_metadata(
    protocol: Mapping[str, Any],
    checkpoint: Path,
    settings: ProducerSettings,
) -> dict[str, Any]:
    """
    Build shared artifact metadata from immutable run provenance.
    """
    return {
        "model": protocol["model"],
        "run_dir": str(checkpoint.parent.parent),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint),
        "resolution": settings.training.image_size,
        "smoke_run": protocol["smoke_run"],
        "framework_version": version("yolox"),
        "postprocessing": {
            "score_floor": 0.001,
            "precision": "float32",
            "nms_device": "cpu",
            "resize": "YOLOX native ValTransform letterbox",
        },
    }


def _benchmark_metadata(
    loaded_model: torch.nn.Module,
    exp: yolox.YOLOXTinyExp,
    annotation_path: Path,
    image_dir: Path,
    category_ids: list[int],
    device: torch.device,
    settings: ProducerSettings,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Benchmark the same native FP32 prediction adapter used for artifact export.
    """
    if not settings.benchmark:
        return {}

    def predict_one(image: Any) -> list[dict]:
        """
        Run one native YOLOX prediction through the shared adapter.
        """
        return artifacts.predict_image(
            loaded_model,
            exp,
            image,
            device=device,
            category_ids=category_ids,
            score_floor=metadata["postprocessing"]["score_floor"],
        )

    annotations = read_json(annotation_path)
    image_paths = [image_dir / entry["file_name"] for entry in annotations["images"]]
    return {
        "benchmark": benchmark_predict(
            predict_one,
            image_paths,
            device=str(device),
        )
    }


def recover_and_publish(
    run_dir: Path,
    paths: DatasetPaths,
    settings: ProducerSettings,
    model: ModelVariant,
    device: torch.device,
    *,
    load_model: Callable[[Any, Path, torch.device], torch.nn.Module] = (
        yolox.load_trained_model
    ),
    bundle_root: Path | None = None,
) -> dict[str, Any]:
    """
    Reload a completed run's best checkpoint and atomically republish both splits.
    """
    run_dir = run_dir.expanduser().resolve()
    protocol = prepare_run(run_dir, paths, settings, model)
    checkpoint = run_dir / BEST_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    exp = model.experiment_class(
        dataset_dir=paths.source_dir,
        output_dir=run_dir.parent,
        max_epoch=settings.training.epochs,
        image_size=settings.training.image_size,
        project_name=run_dir.name,
        seed=settings.training.seed,
        class_names=tuple(
            category["name"]
            for category in protocol["dataset_identity"]["canonical"][
                "canonical_source"
            ]["category_mapping"]
        ),
    )
    loaded_model = load_model(exp, checkpoint, device)
    if load_model is yolox.load_trained_model:
        yolox.verify_coco_loaders(
            exp,
            tuple(
                category["name"]
                for category in protocol["dataset_identity"]["canonical"][
                    "canonical_source"
                ]["category_mapping"]
            ),
            source_dir=paths.source_dir,
        )
    metadata = _prediction_metadata(protocol, checkpoint, settings)
    category_ids = _category_ids(protocol)
    evaluation_dir = run_dir / "evaluation"
    prediction_paths = {}
    for split in ("val", "test"):
        annotation_path = paths.source_dir / "annotations" / f"instances_{split}.json"
        image_dir = paths.source_dir / "images" / split
        split_metadata = {
            **metadata,
            "split": split,
            **_benchmark_metadata(
                loaded_model,
                exp,
                annotation_path,
                image_dir,
                category_ids,
                device,
                settings,
                metadata,
            ),
        }
        prediction_paths[split] = artifacts.export_model_predictions(
            loaded_model,
            exp,
            evaluation_dir / f"{split}_predictions.json",
            annotation_path,
            image_dir,
            device=device,
            category_ids=category_ids,
            metadata=split_metadata,
        )
    destination = bundle_root or (
        OUTPUT_ROOT
        / "evaluation"
        / (
            "basketball_large_dataset"
            if model.logical_dataset == "basketball"
            else model.logical_dataset
        )
        / model.name
        / run_dir.name
    )
    return publish_bundle(
        destination,
        run_dir=run_dir,
        val_predictions=prediction_paths["val"],
        test_predictions=prediction_paths["test"],
        selected_checkpoint=checkpoint,
        resolution=settings.training.image_size,
        parameter_count=sum(
            parameter.numel() for parameter in loaded_model.parameters()
        ),
    )
