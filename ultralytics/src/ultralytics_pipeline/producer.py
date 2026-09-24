"""
Run protocol, training timing, and bundle publication for YOLO11n basketball runs.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from dataset_builder import (
    canonical_coco_identity,
    capture_dataset_identity,
    prepare_smoke_layouts,
    smoke_layout_directory,
    validate_yolo_layout,
    verify_dataset_identity,
)
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

from .artifacts import export_model_predictions, predict_image
from .config import DATA_ROOT, OUTPUT_ROOT
from .ultralytics import assert_run_directory, reserved_run_callback

MODEL_NAME = "yolo11n"
LOGICAL_DATASET = "basketball"
SOURCE_NOTEBOOK = "nb02.02-ultra_yolo11n_large_basketball.ipynb"
BEST_CHECKPOINT = Path("weights/best.pt")
LAST_CHECKPOINT = Path("weights/last.pt")


class TrainableModel(Protocol):
    """
    Minimal Ultralytics model surface used by the training helper.
    """

    def add_callback(self, event: str, callback: Callable[[Any], None]) -> None:
        """
        Register an Ultralytics lifecycle callback.
        """

    def train(self, **kwargs: Any) -> Any:
        """
        Run the native Ultralytics training operation.
        """


@dataclass(frozen=True)
class DatasetPaths:
    """
    Canonical COCO source and its Ultralytics loader layout.
    """

    source_dir: Path
    yolo_dir: Path


@dataclass(frozen=True)
class TrainingSettings:
    """
    Model-owned settings persisted before the timed native training call.
    """

    epochs: int = 100
    resolution: int = 640
    batch_size: int = 16
    patience: int = 25
    workers: int = 0
    cache: bool = False
    device: str = "cuda:0"
    benchmark: bool = True
    smoke_run: bool = False

    def protocol_settings(self) -> dict[str, Any]:
        """
        Return immutable settings relevant to the reproducible producer run.
        """
        return {
            "epochs": self.epochs,
            "resolution": self.resolution,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "workers": self.workers,
            "cache": self.cache,
            "device": self.device,
            "benchmark": self.benchmark,
        }

    def train_kwargs(
        self, data_yaml: Path, run_dir: Path, *, resume: bool
    ) -> dict[str, Any]:
        """
        Build only Ultralytics-supported training arguments.
        """
        common = {
            "device": self.device,
            "save_dir": str(run_dir),
            "project": str(run_dir.parent),
            "name": run_dir.name,
            "exist_ok": True,
            "batch": self.batch_size,
            "workers": self.workers,
            "cache": self.cache,
        }
        if resume:
            return {"resume": True, **common}
        return {
            "data": data_yaml,
            "epochs": self.epochs,
            "imgsz": self.resolution,
            "patience": self.patience,
            **common,
        }


def smoke_enabled(environment: Mapping[str, str] | None = None) -> bool:
    """
    Return whether the explicit one-value smoke switch is enabled.
    """
    values = environment if environment is not None else __import__("os").environ
    value = values.get("ULTRALYTICS_SMOKE", "0")
    if value not in {"0", "1"}:
        raise ValueError("ULTRALYTICS_SMOKE must be 0 or 1")
    return value == "1"


def settings_from_environment(
    *,
    workers: int,
    environment: Mapping[str, str] | None = None,
) -> TrainingSettings:
    """
    Build full or exact Stage 2 smoke settings from the environment switch.
    """
    smoke_run = smoke_enabled(environment)
    return TrainingSettings(
        epochs=2 if smoke_run else 100,
        workers=workers,
        smoke_run=smoke_run,
    )


def settings_from_run(run_dir: Path) -> TrainingSettings:
    """
    Restore training controls and smoke identity for resume or export recovery.
    """
    protocol = read_run_protocol(run_dir)
    if protocol["model"] != MODEL_NAME:
        raise ValueError("Expected a YOLO11n run protocol")
    return TrainingSettings(
        **protocol["training_settings"], smoke_run=protocol["smoke_run"]
    )


def resolve_dataset_paths(
    source_dir: Path,
    yolo_dir: Path,
    *,
    smoke_run: bool,
) -> DatasetPaths:
    """
    Return the full layouts or materialize the complete deterministic smoke layouts.
    """
    source = source_dir.expanduser().resolve()
    loader = yolo_dir.expanduser().resolve()
    if not smoke_run:
        return DatasetPaths(source, loader)
    smoke_root = smoke_layout_directory(source)
    record = prepare_smoke_layouts(source, smoke_root)
    if record["smoke_limits"] != {"train": 16, "val": 8, "test": 8}:
        raise ValueError("Smoke dataset does not use the required 16/8/8 layout")
    return DatasetPaths(smoke_root / "coco", smoke_root / "yolo")


def _dataset_identity(paths: DatasetPaths) -> dict[str, dict[str, Any]]:
    """
    Capture the canonical source and validated YOLO loader identities.
    """
    return {
        "canonical": canonical_coco_identity(paths.source_dir),
        "loader": validate_yolo_layout(paths.source_dir, paths.yolo_dir),
    }


def _verify_protocol(
    run_dir: Path,
    identity: Mapping[str, Any],
    settings: TrainingSettings,
    original_utc: datetime | str,
) -> dict[str, Any]:
    """
    Persist fresh provenance or verify the immutable metadata of an existing run.
    """
    return create_run_protocol(
        run_dir,
        logical_dataset=LOGICAL_DATASET,
        model=MODEL_NAME,
        source_notebook=SOURCE_NOTEBOOK,
        original_utc=original_utc,
        training_settings=settings.protocol_settings(),
        smoke_run=settings.smoke_run,
        canonical_dataset_identity=identity["canonical"],
        loader_dataset_identity=identity["loader"],
    )


def prepare_run(
    run_dir: Path,
    paths: DatasetPaths,
    settings: TrainingSettings,
    *,
    original_utc: datetime | str | None = None,
) -> dict[str, Any]:
    """
    Persist or verify dataset and run provenance before a timed training attempt.
    """
    run_dir = run_dir.expanduser().resolve()
    identity = _dataset_identity(paths)
    identity_path = run_dir / "dataset_identity.json"
    if identity_path.exists():
        verify_dataset_identity(identity_path, identity)
    else:
        capture_dataset_identity(identity_path, identity)
    if original_utc is None:
        original_utc = read_run_protocol(run_dir)["original_utc"]
    return _verify_protocol(run_dir, identity, settings, original_utc)


def _completed_epochs(run_dir: Path) -> int:
    """
    Count recorded native epochs without deriving timing from file timestamps.
    """
    history = run_dir / "results.csv"
    if not history.is_file():
        return 0
    with history.open(newline="", encoding="utf-8") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def synchronize_cuda(device: str) -> None:
    """
    Synchronize a CUDA device when timing a CUDA-backed native training call.
    """
    if not device.startswith("cuda"):
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def train_pinned_run(
    model: TrainableModel,
    run_dir: Path,
    paths: DatasetPaths,
    settings: TrainingSettings,
    *,
    original_utc: datetime | str | None = None,
    resumed: bool = False,
    synchronize: Callable[[], None] | None = None,
    monotonic_clock: Callable[[], float] | None = None,
) -> Any:
    """
    Train one pinned run while finalizing its attempt on every catchable outcome.
    """
    run_dir = run_dir.expanduser().resolve()
    protocol = prepare_run(run_dir, paths, settings, original_utc=original_utc)
    model.add_callback("on_pretrain_routine_start", reserved_run_callback(run_dir))
    data_yaml = paths.yolo_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    import torch

    before_epochs = _completed_epochs(run_dir)
    clock = (
        monotonic_clock
        if monotonic_clock is not None
        else __import__("time").monotonic
    )
    boundary = synchronize or (lambda: synchronize_cuda(settings.device))
    attempt_id = start_training_attempt(
        run_dir,
        completed_epochs_before=before_epochs,
        resumed=resumed,
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
        monotonic_clock=clock,
        synchronize=boundary,
    )
    outcome = "completed"
    try:
        result = model.train(
            **settings.train_kwargs(data_yaml, run_dir, resume=resumed)
        )
    except BaseException:
        outcome = "interrupted"
        raise
    finally:
        completed_epochs = max(0, _completed_epochs(run_dir) - before_epochs)
        finalize_training_attempt(
            run_dir,
            attempt_id,
            outcome=outcome,
            completed_epochs=completed_epochs,
            monotonic_clock=clock,
            synchronize=boundary,
        )
    assert_run_directory(result.save_dir, run_dir)
    if protocol["smoke_run"] != settings.smoke_run:
        raise RuntimeError("Run protocol smoke state changed during training")
    return result


def _prediction_metadata(
    protocol: Mapping[str, Any],
    checkpoint: Path,
    settings: TrainingSettings,
) -> dict[str, Any]:
    """
    Build shared artifact metadata from immutable run provenance.
    """
    from detection_evaluation import file_sha256

    return {
        "model": MODEL_NAME,
        "run_dir": str(checkpoint.parent.parent),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "resolution": settings.resolution,
        "smoke_run": protocol["smoke_run"],
        "postprocessing": {
            "score_floor": 0.001,
            "nms_iou": 0.7,
            "max_det": 300,
            "agnostic_nms": False,
            "rect": True,
            "precision": "float32",
            "resize": "Ultralytics native letterbox",
        },
    }


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


def _benchmark_metadata(
    model: Any,
    annotation_path: Path,
    image_dir: Path,
    category_ids: list[int],
    settings: TrainingSettings,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Measure native batch-one prediction with the same adapter used for export.
    """
    if not settings.benchmark:
        return {}
    postprocessing = metadata["postprocessing"]

    def predict_one(image: Any) -> list[dict]:
        """
        Run the native adapter with recorded inference postprocessing.
        """
        return predict_image(
            model,
            image,
            category_ids=category_ids,
            resolution=settings.resolution,
            device=settings.device,
            score_floor=postprocessing["score_floor"],
            nms_iou=postprocessing["nms_iou"],
            max_det=postprocessing["max_det"],
            agnostic_nms=postprocessing["agnostic_nms"],
        )

    annotations = read_json(annotation_path)
    image_paths = [image_dir / entry["file_name"] for entry in annotations["images"]]
    return {
        "benchmark": benchmark_predict(
            predict_one,
            image_paths,
            device=settings.device,
        )
    }


def recover_and_publish(
    run_dir: Path,
    paths: DatasetPaths,
    settings: TrainingSettings,
    *,
    load_model: Callable[[Path], Any],
    bundle_root: Path | None = None,
) -> dict[str, Any]:
    """
    Revalidate a completed run, export both splits, and atomically publish a bundle.
    """
    run_dir = run_dir.expanduser().resolve()
    protocol = prepare_run(run_dir, paths, settings)
    checkpoint = run_dir / BEST_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = load_model(checkpoint)
    names = [model.names[index] for index in range(len(model.names))]
    expected_names = [
        category["name"]
        for category in protocol["dataset_identity"]["canonical"]["canonical_source"][
            "category_mapping"
        ]
    ]
    if names != expected_names:
        raise ValueError(f"Unexpected checkpoint classes: {model.names}")
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
                model,
                annotation_path,
                image_dir,
                category_ids,
                settings,
                metadata,
            ),
        }
        prediction_paths[split] = export_model_predictions(
            model,
            evaluation_dir / f"{split}_predictions.json",
            annotation_path,
            image_dir,
            category_ids=category_ids,
            resolution=settings.resolution,
            device=settings.device,
            metadata=split_metadata,
        )
    parameters = sum(parameter.numel() for parameter in model.model.parameters())
    destination = bundle_root or (
        OUTPUT_ROOT
        / "evaluation"
        / "basketball_large_dataset"
        / MODEL_NAME
        / run_dir.name
    )
    return publish_bundle(
        destination,
        run_dir=run_dir,
        val_predictions=prediction_paths["val"],
        test_predictions=prediction_paths["test"],
        selected_checkpoint=checkpoint,
        resolution=settings.resolution,
        parameter_count=parameters,
    )