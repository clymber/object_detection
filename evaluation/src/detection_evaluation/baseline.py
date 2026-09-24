"""
Framework-neutral validation and artifact writing for frozen detector runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from detection_common.utils.json_io import read_json, write_json
from PIL import Image

from .metrics import benchmark_predict, evaluate_predictions, read_prediction_artifact
from .producer import export_predictions


@dataclass(frozen=True)
class BaselineExport:
    """
    Hold a validated, full-dataset run without importing a model framework.
    """

    run_dir: Path
    dataset_dir: Path
    output_dir: Path
    checkpoint: Path
    model_name: str
    resolution: int
    device: str
    splits: tuple[str, ...]
    category_ids: tuple[int, ...]
    class_names: tuple[str, ...]
    training_settings: dict[str, Any]
    epochs_completed: int

    @property
    def category_id(self) -> int:
        """Return the legacy scalar for a single-category export."""
        if len(self.category_ids) != 1:
            raise ValueError("Use category_ids for multiclass exports")
        return self.category_ids[0]


def latest_run_dir(runs_dir: Path, pattern: str, checkpoint: Path) -> Path:
    """
    Find the newest complete training run matching a model-owned pattern.
    """
    required = (Path("args.yaml"), Path("results.csv"), checkpoint)
    candidates = [
        path
        for path in runs_dir.expanduser().resolve().glob(pattern)
        if path.is_dir() and all((path / item).is_file() for item in required)
    ]
    if not candidates:
        raise FileNotFoundError(f"No exportable run matching {pattern!r} in {runs_dir}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def prepare_baseline(
    run_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    *,
    checkpoint: Path,
    model_name: str,
    resolution: int,
    device: str,
    split: str = "both",
) -> BaselineExport:
    """
    Validate frozen inputs and output names before loading a detector.
    """
    run_dir = run_dir.expanduser().resolve()
    dataset_dir = dataset_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if split not in {"val", "test", "both"}:
        raise ValueError("Split must be val, test, or both")
    if resolution <= 0 or resolution % 32:
        raise ValueError("Resolution must be a positive multiple of 32")
    checkpoint_path = run_dir / checkpoint
    for path in (run_dir / "args.yaml", run_dir / "results.csv", checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    with (run_dir / "args.yaml").open() as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("Run args.yaml must contain an object")
    if settings.get("smoke_run"):
        raise ValueError("Smoke runs are not allowed")
    if (
        settings.get("train_batch_limit") is not None
        or float(settings.get("fraction", 1.0)) != 1.0
    ):
        raise ValueError("Truncated training runs are not allowed")
    trained_path = Path(str(settings.get("data", settings.get("dataset", ""))))
    trained_name = (
        trained_path.parent.name
        if trained_path.suffix == ".yaml"
        else trained_path.name
    )
    dataset_name = dataset_dir.name.removeprefix("coco_")
    if trained_name.removeprefix("yolo_").removeprefix("coco_") != dataset_name:
        raise ValueError("Run dataset name does not match comparison dataset")
    train = read_json(dataset_dir / "annotations" / "instances_train.json")
    categories = sorted(train["categories"], key=lambda item: item["id"])
    category_ids = tuple(category["id"] for category in categories)
    class_names = tuple(category["name"] for category in categories)
    if not category_ids or len(set(category_ids)) != len(category_ids):
        raise ValueError("Expected unique COCO categories")
    splits = ("val", "test") if split == "both" else (split,)
    for selected in splits:
        split_output_dir = output_dir / selected
        for kind in ("predictions", "metrics"):
            path = split_output_dir / f"{model_name}_{selected}_{kind}.json"
            if path.exists():
                raise FileExistsError(f"Choose a new output directory: {path}")
        annotations = read_json(
            dataset_dir / "annotations" / f"instances_{selected}.json"
        )
        split_categories = annotations["categories"]
        if (
            sorted((item["id"], item["name"]) for item in split_categories)
            != list(zip(category_ids, class_names, strict=True))
            or not annotations["images"]
        ):
            raise ValueError(f"Invalid category mapping or empty {selected} split")
        for image in annotations["images"]:
            image_path = dataset_dir / "images" / selected / image["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    if history.empty or "epoch" not in history:
        raise ValueError("Run requires completed epochs in results.csv")
    return BaselineExport(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        checkpoint=checkpoint_path,
        model_name=model_name,
        resolution=resolution,
        device=device,
        splits=splits,
        category_ids=category_ids,
        class_names=class_names,
        training_settings=settings,
        epochs_completed=int(history["epoch"].max()),
    )


def export_baseline(
    context: BaselineExport,
    predict_one: Callable[[Image.Image], list[dict]],
    *,
    metadata: dict[str, Any],
    benchmark: bool = False,
) -> list[Path]:
    """
    Write neutral prediction and metric files for every selected split.
    """
    common_metadata = {
        **metadata,
        "model": context.model_name,
        "run_dir": str(context.run_dir),
        "checkpoint": str(context.checkpoint),
        "resolution": context.resolution,
        "device": context.device,
        "smoke_run": False,
        "training_settings": context.training_settings,
        "epochs_completed": context.epochs_completed,
        "dataset_provenance": (
            "Run and dataset names match; annotation hashes are recorded now. "
            "Historical training data has no recorded content hash."
        ),
    }
    paths = []
    for split in context.splits:
        split_output_dir = context.output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        annotation_path = (
            context.dataset_dir / "annotations" / f"instances_{split}.json"
        )
        image_dir = context.dataset_dir / "images" / split
        split_metadata = {**common_metadata, "split": split}
        if benchmark:
            annotations = read_json(annotation_path)
            image_paths = [
                image_dir / row["file_name"] for row in annotations["images"]
            ]
            split_metadata["benchmark"] = benchmark_predict(
                predict_one, image_paths, device=context.device
            )
        artifact_path = (
            split_output_dir / f"{context.model_name}_{split}_predictions.json"
        )
        paths.append(
            export_predictions(
                artifact_path,
                annotation_path,
                image_dir,
                predict_one,
                metadata=split_metadata,
            )
        )
        artifact = read_prediction_artifact(artifact_path, annotation_path)
        write_json(
            split_output_dir / f"{context.model_name}_{split}_metrics.json",
            evaluate_predictions(annotation_path, artifact["predictions"]),
        )
    return paths
