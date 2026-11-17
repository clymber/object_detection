"""
Explicit preparation, export and validation steps for YOLOX checkpoints.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import yaml
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
    runs_dir: Path | None = None,
    formats: tuple[str, ...] | None = None,
    device: str = "cpu",
    cpu_threads: int = 4,
    coreml_precision: str = "float32",
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
    model = "yolox_nano"
    run = select_run(runs_dir or OUTPUT_ROOT / "runs/basketball", "yolox_nano", run_dir)
    protocol = validate_run(run, "yolox_nano")
    resolution, categories, mapping, nms = _saved_model_settings(run, protocol)
    validate_dataset(protocol, dataset_dir)
    configuration = _configuration(
        resolution,
        categories,
        mapping,
        nms,
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
    Verify the saved Nano dimensions, class names and class-aware NMS policy.
    """
    args = yaml.safe_load((run / "args.yaml").read_text())
    resolution = protocol["training_settings"]["image_size"]
    if args["image_size"] != [resolution, resolution]:
        raise ValueError("Saved YOLOX input dimensions differ from protocol")
    mapping = protocol["dataset_identity"]["canonical"]["canonical_source"][
        "category_mapping"
    ]
    categories = [c["id"] for c in mapping]
    names = tuple(c["name"] for c in mapping)
    if list(names) != args["classes"]:
        raise ValueError("Saved YOLOX class names differ from source categories")
    nms = float(args["nms_threshold"])
    if not 0 < nms <= 1:
        raise ValueError("Invalid saved YOLOX NMS threshold")
    return resolution, categories, mapping, nms


def _configuration(
    resolution,
    categories,
    mapping,
    nms,
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
        "preprocessing": "RGB-to-BGR;top-left bilinear letterbox;padding=114;float32 0..255",
        "postprocessing": {
            "score_floor": 0.001,
            "nms_iou": nms,
            "nms": "class-aware CPU",
            "graph_decoding": False,
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
    if manifest["model"] != "yolox_nano":
        raise ValueError("Expected a YOLOX Nano export set")
    plan = prepare_export(
        Path(manifest["run_dir"]),
        Path(manifest["dataset"]["directory"]),
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
