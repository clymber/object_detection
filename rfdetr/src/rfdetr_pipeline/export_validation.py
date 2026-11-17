"""
Explicit preparation, export and validation steps for RF-DETR checkpoints.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from detection_common.utils.json_io import read_json
from detection_evaluation import runtime_stages as stages
from detection_evaluation.runtime_stages import load_export_set, read_results
from detection_evaluation.runtime_workflow import (
    ExportPlan,
    ExportSet,
    select_run,
    validate_dataset,
    validate_export_settings,
    validate_run,
    workspace_path,
)

from . import export_backends
from .config import OUTPUT_ROOT, WORKSPACE_ROOT

__all__ = [
    "ExportPlan",
    "ExportSet",
    "benchmark_host",
    "check_parity",
    "evaluate_accuracy",
    "export_models",
    "load_export_set",
    "prepare_export",
    "read_results",
]


def prepare_export(
    run_dir: Path | None,
    dataset_dir: Path,
    *,
    model: str,
    runs_dir: Path | None = None,
    formats: tuple[str, ...] | None = None,
    device: str = "cpu",
    cpu_threads: int = 4,
    coreml_precision: str = "float16",
    compute_units: str = "all",
) -> ExportPlan:
    """
    Verify the selected run and local data without loading weights or writing files.
    """
    formats = validate_export_settings(
        formats,
        device=device,
        cpu_threads=cpu_threads,
        coreml_precision=coreml_precision,
        compute_units=compute_units,
    )
    dataset_dir = workspace_path(dataset_dir)
    if model not in {"rfdetr_small", "rfdetr_nano"}:
        raise ValueError(f"Expected RF-DETR Small/Nano, got {model}")
    run = select_run(runs_dir or OUTPUT_ROOT / "runs/basketball", model, run_dir)
    protocol = validate_run(run, model)
    resolution, categories, mapping, saved = _saved_model_settings(run, protocol)
    validate_dataset(protocol, dataset_dir)
    configuration = _configuration(
        resolution,
        categories,
        mapping,
        saved,
        device,
        cpu_threads,
        coreml_precision,
        compute_units,
    )
    return ExportPlan(
        run_dir=run,
        dataset_dir=dataset_dir,
        model=model,
        formats=formats,
        protocol=protocol,
        configuration=configuration,
        output_root=OUTPUT_ROOT,
        models_root=WORKSPACE_ROOT / "models",
    )


def _saved_model_settings(run: Path, protocol: dict) -> tuple:
    """
    Verify trained dimensions, class identities and the saved selection count.
    """
    saved = read_json(run / "resolved_config.json")["model_config"]
    resolution = protocol["training_settings"]["resolution"]
    if saved["resolution"] != resolution:
        raise ValueError("Saved model resolution differs from run protocol")
    mapping = read_json(run / "dataset_manifest.json")["category_mapping"]
    categories = [
        mapping["prediction_to_source"][str(i)]
        for i in range(len(mapping["class_names"]))
    ]
    source_categories = protocol["dataset_identity"]["canonical"]["canonical_source"][
        "category_mapping"
    ]
    if categories != [category["id"] for category in source_categories]:
        raise ValueError("Saved RF-DETR category mapping differs from source identity")
    if mapping["class_names"] != [category["name"] for category in source_categories]:
        raise ValueError("Saved RF-DETR class names differ from source identity")
    return resolution, categories, mapping, saved


def _configuration(
    resolution,
    categories,
    mapping,
    saved,
    device,
    cpu_threads,
    coreml_precision,
    compute_units,
) -> dict:
    """
    Describe the saved preprocessing, postprocessing and runtime policy exactly.
    """
    return {
        "resolution": resolution,
        "category_ids": categories,
        "class_mapping": mapping,
        "checkpoint_device": device,
        "cpu_threads": cpu_threads,
        "coreml_precision": coreml_precision,
        "compute_units": compute_units,
        "preprocessing": "RGB float32/255;half-pixel square bilinear;ImageNet normalization",
        "postprocessing": {
            "score_floor": 0.001,
            "num_select": saved["num_select"],
            "nms": None,
            "background": "rank then discard last slot",
            "boxes": "unclipped source xywh",
        },
    }


def export_models(plan: ExportPlan) -> ExportSet:
    """
    Convert requested models without running parity, accuracy or host timing.
    """
    return stages.export_models(plan, partial(export_backends.export_model, plan))


def _saved_plan(export_set: ExportSet) -> ExportPlan:
    """
    Restore and verify model-specific settings from the pinned saved export set.
    """
    manifest = read_results(export_set)
    config = manifest["configuration"]
    plan = prepare_export(
        Path(manifest["run_dir"]),
        Path(manifest["dataset"]["directory"]),
        model=manifest["model"],
        formats=tuple(manifest["exports"]),
        device=config["checkpoint_device"],
        cpu_threads=config["cpu_threads"],
        coreml_precision=config["coreml_precision"],
        compute_units=config["compute_units"],
    )
    if plan.configuration != config:
        raise ValueError("Saved export configuration differs from the source run")
    return plan


def check_parity(export_set: ExportSet) -> dict:
    """
    Compare saved exports to a fresh checkpoint reference, without evaluating splits.
    """
    plan = _saved_plan(export_set)
    return stages.check_parity(export_set, partial(export_backends.load_runtime, plan))


def evaluate_accuracy(export_set: ExportSet) -> dict:
    """
    Evaluate full splits without converting models, checking parity or timing.
    """
    plan = _saved_plan(export_set)
    return stages.evaluate_accuracy(
        export_set, partial(export_backends.load_runtime, plan)
    )


def benchmark_host(export_set: ExportSet) -> dict:
    """
    Time eligible runtimes using their saved successful parity and accuracy results.
    """
    plan = _saved_plan(export_set)
    return stages.benchmark_host(
        export_set, partial(export_backends.load_runtime, plan)
    )
