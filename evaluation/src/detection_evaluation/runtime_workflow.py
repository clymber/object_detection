"""
Framework-neutral contracts, selection, parity and host validation for exports.

Model packages supply fresh predictors and exporters. This module never imports
model frameworks. Training inputs and legacy benchmark records remain unchanged;
managed exports and their validation results are updated in place.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import socket
import shutil
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from detection_common.utils.digest import file_digest
from detection_common.utils.json_io import read_json, write_json
from PIL import Image, ImageDraw

from .baseline import find_latest_path
from .config import WORKSPACE_ROOT
from .run_protocol import read_run_protocol, read_training_record

MODELS = ("rfdetr_small", "rfdetr_nano", "yolox_nano")
FORMATS = ("pytorch", "onnx", "coreml")
TIMING_BOUNDARIES = "decoded-RGB-to-source-COCO;preprocess+predict+postprocess;synchronous"
HOST_TIMING = {"warmups": 5, "samples": 100, "repetitions": 1}


@dataclass(frozen=True)
class ExportPlan:
    """
    Carry verified metadata and paths without retaining a model or runtime.
    """

    run_dir: Path
    dataset_dir: Path
    model: str
    formats: tuple[str, ...]
    protocol: dict
    configuration: dict
    output_root: Path
    models_root: Path


@dataclass(frozen=True)
class ExportSet:
    """
    Pin one saved export generation independently of its changing reports.
    """

    manifest_path: Path
    export_id: str


@dataclass(frozen=True)
class RuntimeAdapter:
    """
    Supply synchronous prediction, provenance and device synchronization.
    """

    predict: Callable
    contract: dict
    synchronize: Callable


def validate_export_settings(
    formats: tuple[str, ...] | None, *, device: str, cpu_threads: int,
    coreml_precision: str, compute_units: str,
) -> tuple[str, ...]:
    """
    Resolve platform defaults and reject unsupported runtime policies early.
    """
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    resolved = tuple(formats if formats is not None else (
        ("onnx", "coreml") if platform.system() == "Darwin" else ("onnx",)))
    validate_format_settings(resolved, coreml_precision, compute_units)
    if device.split(":")[0] not in {"cpu", "mps", "cuda"}:
        raise ValueError("Checkpoint device must be cpu, mps or cuda[:index]")
    return resolved


def utc_id() -> str:
    """
    Allocate a unique UTC attempt name without relying on directory timestamps.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + uuid.uuid4().hex[:8]


def digest_json(value: Any) -> str:
    """
    Hash a deterministic JSON document.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def artifact_identity(path: Path) -> dict:
    """
    Hash a file or a sorted per-file package manifest, including relative names.
    """
    if path.is_file():
        return {"path": str(path.resolve()), "sha256": file_digest(path)}
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = {p.relative_to(path).as_posix(): file_digest(p)
             for p in sorted(path.rglob("*")) if p.is_file() and p.name != ".DS_Store"}
    if not files:
        raise ValueError(f"Empty model package: {path}")
    return {"path": str(path.resolve()), "sha256": digest_json(files), "files": files}


def workspace_path(value: Path | str) -> Path:
    """
    Resolve configured relative paths against the checkout, never the shell cwd.
    """
    path = Path(value).expanduser()
    return (path if path.is_absolute() else WORKSPACE_ROOT / path).resolve()


def checkpoint_path(run: Path, model: str) -> Path:
    """
    Resolve the selected inference checkpoint for a supported model.
    """
    if model not in MODELS:
        raise ValueError(f"Unsupported model: {model}")
    return run / ("weights/best_ckpt.pth" if model == "yolox_nano"
                  else "checkpoint_best_total.pth")


def export_artifact_path(run: Path, model: str, format_name: str) -> Path:
    """
    Return the single canonical artifact location for one run and format.
    """
    checkpoint = checkpoint_path(run, model)
    suffix = ".onnx" if format_name == "onnx" else ".mlpackage"
    return run / "weights" / f"{checkpoint.stem}{suffix}"


def install_export(staged: Path, destination: Path, *,
                   previous: Path | None = None) -> Path:
    """
    Replace a canonical export only after a fresh conversion has succeeded.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir():
        shutil.rmtree(destination)
    os.replace(staged, destination)
    if previous is not None and previous != destination:
        if previous.is_dir():
            shutil.rmtree(previous)
        else:
            previous.unlink(missing_ok=True)
    return destination


def _replace_link(alias: Path, target: Path) -> None:
    """
    Keep a relative compatibility link without duplicating an exported artifact.
    """
    if alias.is_symlink() and alias.resolve() == target.resolve():
        return
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.is_symlink() or alias.is_file():
        alias.unlink()
    elif alias.is_dir():
        shutil.rmtree(alias)
    alias.symlink_to(os.path.relpath(target, alias.parent), target_is_directory=target.is_dir())


def publish_export(staged: Path, directory: Path, run: Path,
                   model: str, format_name: str) -> Path:
    """
    Publish the single model artifact and link its uniform run/weights path.
    """
    alias = export_artifact_path(run, model, format_name)
    canonical = directory / alias.name
    install_export(staged, canonical)
    _replace_link(alias, canonical)
    if model.startswith("rfdetr"):
        legacy = run / alias.name
        if legacy.is_symlink() or legacy.is_file():
            legacy.unlink()
        elif legacy.is_dir():
            shutil.rmtree(legacy)
    return canonical


def training_source_file(path: Path, run: Path, model: str) -> bool:
    """
    Protect training inputs while allowing managed exports and validation updates.
    """
    if path.name == ".DS_Store":
        return False
    managed = [run / "export_validation", *[
        export_artifact_path(run, model, name) for name in ("onnx", "coreml")]]
    if model.startswith("rfdetr"):
        managed += [run / export_artifact_path(run, model, name).name
                    for name in ("onnx", "coreml")]
    return not any(path == root or root in path.parents for root in managed)


def validate_run(run: Path, model: str) -> dict:
    """
    Require full completed training and saved model-specific reload configuration.
    """
    protocol = read_run_protocol(run)
    training = read_training_record(run)
    attempts = training["attempts"]
    if (protocol["model"] != model or protocol["logical_dataset"] != "basketball"
            or protocol["smoke_run"]
            or protocol["training_settings"].get("train_batch_limit") is not None
            or not attempts or attempts[-1]["status"] != "completed"
            or any(a["status"] == "running" for a in attempts)):
        raise ValueError(f"Expected a completed full basketball {model} run: {run}")
    required = [checkpoint_path(run, model), run / "dataset_identity.json"]
    if model == "yolox_nano":
        required.append(run / "args.yaml")
    if model.startswith("rfdetr"):
        required += [run / name for name in (
            "run_config.json", "resolved_config.json", "dataset_manifest.json")]
        metadata_path = run / "best_checkpoint.json"
        selection = (read_json(metadata_path).get("selected_weights")
                     if metadata_path.is_file() else None)
        sources = (selection,) if selection in {"ema", "regular"} else ("ema", "regular")
        required += [run / f"checkpoint_best_{source}.pth" for source in sources]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if read_json(run / "dataset_identity.json") != protocol["dataset_identity"]:
        raise ValueError("Saved dataset identity differs from run protocol")
    return protocol


def eligible_run(run: Path, model: str) -> bool:
    """
    Filter discovery candidates while letting explicit selections fail loudly.
    """
    try:
        validate_run(run, model)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def select_run(runs_dir: Path, model: str, run_dir: Path | None = None) -> Path:
    """
    Honor an exact override or use the shared mtime/path discovery order.
    """
    if run_dir is not None:
        selected = workspace_path(run_dir)
        validate_run(selected, model)
        return selected
    selected = find_latest_path(
        workspace_path(runs_dir), f"{model}_*", predicate=lambda p: eligible_run(p, model)
    )
    if selected is None:
        raise FileNotFoundError(f"No completed {model} run under {runs_dir}")
    return selected


def validate_dataset(protocol: Mapping, dataset_dir: Path) -> dict:
    """
    Verify local annotations and pixels against archived path-independent identity.
    """
    canonical = protocol["dataset_identity"]["canonical"]
    source = canonical["canonical_source"]
    annotations = {}
    for split, saved in source["splits"].items():
        path = dataset_dir / "annotations" / f"instances_{split}.json"
        current = read_json(path)
        if (sorted(current["annotations"], key=lambda a: a["id"])
                != sorted(saved["annotations"], key=lambda a: a["id"])):
            raise ValueError(f"Dataset annotations differ: {split}")
        categories = sorted(current["categories"], key=lambda c: c["id"])
        if categories != source["category_mapping"]:
            raise ValueError(f"Dataset categories differ: {split}")
        expected = saved["images"]
        actual = sorted(current["images"], key=lambda i: i["id"])
        if len(actual) != len(expected):
            raise ValueError(f"Dataset image count differs: {split}")
        for image, record in zip(actual, expected, strict=True):
            if any(image[key] != record[saved_key] for key, saved_key in (
                ("id", "image_id"), ("file_name", "file_name"),
                ("width", "width"), ("height", "height"))):
                raise ValueError(f"Dataset image identity differs: {split}")
            pixel_path = dataset_dir / "images" / split / image["file_name"]
            if file_digest(pixel_path) != record["sha256"]:
                raise ValueError(f"Dataset pixel hash differs: {pixel_path}")
        if split in {"val", "test"}:
            annotations[split] = {"path": str(path), "sha256": file_digest(path),
                                  "image_ids": [i["id"] for i in actual]}
    return {"fingerprint": canonical["source_fingerprint"],
            "directory": str(dataset_dir), "annotations": annotations}


def development_selection(ground_truth: dict, annotation_sha256: str) -> dict:
    """
    Select fixed validation positives/negatives and describe coverage independently.
    """
    grouped = defaultdict(list)
    for row in ground_truth["annotations"]:
        grouped[row["image_id"]].append(row)
    images = sorted(ground_truth["images"], key=lambda i: i["id"])
    positives = [i["id"] for i in images if grouped[i["id"]]][:20]
    negatives = [i["id"] for i in images if not grouped[i["id"]]][:5]
    ids = positives + negatives
    small = any(a.get("area", a["bbox"][2] * a["bbox"][3]) < 32 ** 2
                for i in ids for a in grouped[i])
    overlaps = any(_iou(a["bbox"], b["bbox"]) > 0
                   for i in ids for index, a in enumerate(grouped[i])
                   for b in grouped[i][index + 1:])
    return {"kind": "development;not-frozen-mobile-parity",
            "policy": {"confidence": .25, "inclusive": True, "top_k": 100,
                       "score_absolute_tolerance": .02,
                       "coordinate_tolerance": "max(2 pixels, 0.002 * source axis dimension)",
                       "matching": "reference score order;unmatched same-category maximum IoU"},
            "image_ids": ids, "positive_ids": positives, "negative_ids": negatives,
            "annotation_sha256": annotation_sha256,
            "covers_small_objects": small, "covers_overlaps": overlaps}


def _iou(a: list, b: list) -> float:
    """
    Compute source-coordinate COCO box IoU.
    """
    intersection = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
        0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return intersection / max(a[2]*a[3]+b[2]*b[3]-intersection, 1e-12)


def retained(rows: list[dict], threshold: float = 0.25) -> list[dict]:
    """
    Apply inclusive confidence and stable final top-100 selection.
    """
    return sorted((r for r in rows if r["score"] >= threshold),
                  key=lambda r: -r["score"])[:100]


def compare_detections(reference: list, runtime: list, width: int, height: int) -> dict:
    """
    Follow protocol reference-score order and unmatched same-category maximum IoU.
    """
    for row in reference + runtime:
        if (not np.isfinite(row["score"]) or not 0 <= row["score"] <= 1
                or not np.isfinite(row["bbox"]).all()
                or row["bbox"][2] <= 0 or row["bbox"][3] <= 0):
            raise ValueError("Parity requires finite valid scores and positive boxes")
    expected, actual = retained(reference), retained(runtime)
    tolerance = [max(2.0, .002 * width), max(2.0, .002 * height)] * 2

    def corners(row: dict) -> list:
        """
        Convert xywh to xyxy for axis-specific comparison.
        """
        x, y, w, h = row["bbox"]
        return [x, y, x+w, y+h]

    def unmatched_failure(kind: str, row: dict, pool: list) -> dict:
        """
        Distinguish unmatched detections caused by confidence threshold crossings.
        """
        crossing = any(r["category_id"] == row["category_id"]
                       and r["score"] < .25 and _iou(r["bbox"], row["bbox"]) > .5
                       for r in pool)
        return {"kind": kind, "reason": "threshold_crossing" if crossing else kind,
                "detection": row}

    available = set(range(len(actual)))
    failures, matches = [], []
    for row in expected:
        candidates = [i for i in available if actual[i]["category_id"] == row["category_id"]]
        if not candidates:
            failures.append(unmatched_failure("missing", row, runtime))
            continue
        selected = max(candidates, key=lambda i: (_iou(row["bbox"], actual[i]["bbox"]), -i))
        available.remove(selected)
        other = actual[selected]
        errors = [abs(a-b) for a, b in zip(corners(row), corners(other), strict=True)]
        score_error = abs(row["score"]-other["score"])
        match = {"reference": row, "runtime": other, "corner_errors": errors,
                 "score_error": score_error, "iou": _iou(row["bbox"], other["bbox"])}
        matches.append(match)
        if any(error > bound + 1e-8 for error, bound in zip(errors, tolerance, strict=True)):
            failures.append({"kind": "shifted_box", **match})
        if score_error > .02 + 1e-8:
            failures.append({"kind": "score_difference", **match})
    for index in sorted(available):
        failures.append(unmatched_failure("extra", actual[index], reference))
    return {"passed": not failures, "reference_count": len(expected),
            "runtime_count": len(actual), "matched": len(matches),
            "coordinate_tolerances": tolerance, "matches": matches, "failures": failures}


def host_timing(predict: Callable, images: dict, sequence: list[int], *,
                backend: dict, synchronize: Callable = lambda: None,
                warmups: int = 5, repetitions: int = 1) -> dict:
    """
    Measure synchronous batch-one host inference on predecoded, fixed pixels.
    """
    for index in range(warmups):
        predict(images[sequence[index % len(sequence)]])
        synchronize()
    measurements = []
    for _ in range(repetitions):
        samples = []
        for image_id in sequence:
            synchronize()
            start = time.perf_counter_ns()
            predict(images[image_id])
            synchronize()
            samples.append((time.perf_counter_ns()-start) / 1e6)
        measurements.append(samples)
    values = np.asarray(measurements).reshape(-1)
    return {"kind": "host", "host": {"name": socket.gethostname(),
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor()}, "backend": backend,
            "sequence": sequence, "sequence_sha256": digest_json(sequence),
            "warmups": warmups, "samples": len(sequence), "repetitions": repetitions,
            "batch_size": 1, "boundaries": TIMING_BOUNDARIES,
            "mean_ms": float(values.mean()), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "throughput_images_second": float(1000 / values.mean()),
            "latencies_ms": measurements}


def resolved_versions() -> dict:
    """
    Capture installed producer/converter/runtime versions without importing them.
    """
    result = {}
    for name in ("torch", "torchvision", "rfdetr", "yolox", "onnx", "onnxruntime",
                 "coremltools", "numpy", "pillow", "pycocotools"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def validate_format_settings(formats: tuple, precision: str, compute_units: str) -> None:
    """
    Reject unknown formats/settings; host support is recorded per requested stage.
    """
    if not formats or len(set(formats)) != len(formats) or set(formats)-{"onnx", "coreml"}:
        raise ValueError("Request a nonempty unique sequence of onnx/coreml formats")
    if precision not in {"float16", "float32"}:
        raise ValueError("Core ML precision must be float16 or float32")
    if compute_units not in {"all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"}:
        raise ValueError("Unknown Core ML compute units")


def qualitative_grid(destination: Path, ground_truth: dict, predictions: list,
                     dataset: Path, split: str, ids: list) -> None:
    """
    Save fixed source-image overlays for every runtime, including negatives.
    """
    by_id = {i["id"]: i for i in ground_truth["images"]}
    canvas = Image.new("RGB", (4*256, 2*280), "white")
    grouped = defaultdict(list)
    for row in predictions:
        if row["score"] >= .25:
            grouped[row["image_id"]].append(row)
    for index, image_id in enumerate(ids[:8]):
        entry = by_id[image_id]
        with Image.open(dataset / "images" / split / entry["file_name"]) as source:
            tile = source.convert("RGB").resize((256, 256))
        draw = ImageDraw.Draw(tile)
        for rows, color in (([a for a in ground_truth["annotations"]
                              if a["image_id"] == image_id], "lime"),
                             (grouped[image_id], "red")):
            for row in rows:
                x, y, w, h = row["bbox"]
                draw.rectangle((x*256/entry["width"], y*256/entry["height"],
                                (x+w)*256/entry["width"], (y+h)*256/entry["height"]),
                               outline=color, width=2)
        x, y = index % 4*256, index // 4*280
        canvas.paste(tile, (x, y))
        ImageDraw.Draw(canvas).text((x+4, y+258), f"ID {image_id}", fill="black")
    canvas.save(destination)


def _write_manifest(path: Path, manifest: dict) -> None:
    """
    Publish each progress update atomically so concurrent readers see whole JSON.
    """
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_json(temporary, manifest)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_in_progress(run: Path) -> bool:
    """
    Identify a live canonical-export writer without trusting a stale lock file.
    """
    lock = run / "export_validation/.export.lock"
    if not lock.exists():
        return False
    with lock.open("rb") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return False


def require_success(result: dict) -> None:
    """
    Reject failed, unfinished or unaccepted validation after displaying evidence.
    """
    failed = [name for name, outcome in result["formats"].items()
              if outcome["status"] != "not_requested"
              and (outcome["status"] != "completed"
                   or not outcome.get("accepted", False))]
    if (failed or not result["source_run_unchanged"]
            or result.get("attempt_status", "finished") != "finished"):
        raise RuntimeError(
            f"Export validation requires review: {failed}; "
            f"details: {result['manifest_path']}")


def raw_predictor(forward: Callable, *, family: str, resolution: int,
                  category_ids: list, num_select: int = 300,
                  nms_iou: float | None = None) -> Callable:
    """
    Adapt fixed-shape raw outputs to native, unclipped source-coordinate boxes.

    RF-DETR ranks all sigmoid slots before dropping the explicit background slot,
    just as its checkpoint postprocessor does. YOLOX uses external grid decoding
    and class-aware NMS, with no graph-side decoding in either exported format.
    """
    from .onnx_inference import preprocess_rfdetr, preprocess_yolox, xyxy_nms

    def predict(image: Image.Image) -> list[dict]:
        """
        Preprocess decoded RGB, synchronously execute and decode one image.
        """
        transform = (preprocess_rfdetr if family == "rfdetr" else preprocess_yolox)(
            image, resolution, resolution)
        outputs = forward(transform.tensor)
        if not all(np.isfinite(np.asarray(output)).all() for output in outputs):
            raise ValueError("Runtime returned nonfinite output tensors")
        if family == "rfdetr":
            boxes, logits = map(np.asarray, outputs)
            if (boxes.shape[0] != 1 or boxes.shape[-1] != 4
                    or logits.shape[:2] != boxes.shape[:2]
                    or logits.shape[-1] != len(category_ids)+1):
                raise ValueError("Unexpected RF-DETR foreground/background output contract")
            scores = 1 / (1 + np.exp(-np.clip(logits[0].astype(np.float32), -88, 88)))
            classes = scores.shape[-1]
            order = np.argsort(-scores.ravel(), kind="stable")[:num_select]
            order = order[(order % classes < len(category_ids))
                          & (scores.ravel()[order] >= .001)]
            labels = order % classes
            scores = scores.ravel()[order]
            boxes = boxes[0, order // classes].astype(np.float32).copy()
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, [0, 2]] *= image.width
            boxes[:, [1, 3]] *= image.height
        else:
            raw = np.asarray(outputs[0], dtype=np.float32)
            if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[-1] != 5+len(category_ids):
                raise ValueError("Unexpected YOLOX output contract")
            boxes = raw[0, :, :4].copy()
            grid_rows, stride_rows = [], []
            for stride in (8, 16, 32):
                x, y = np.meshgrid(np.arange(resolution // stride),
                                   np.arange(resolution // stride))
                grid = np.stack((x, y), axis=-1).reshape(-1, 2)
                grid_rows.append(grid)
                stride_rows.append(np.full((len(grid), 1), stride))
            grids, strides = np.concatenate(grid_rows), np.concatenate(stride_rows)
            if len(boxes) != len(grids):
                raise ValueError("YOLOX raw grid differs from trained resolution")
            boxes[:, :2] = (boxes[:, :2]+grids)*strides
            boxes[:, 2:] = np.exp(boxes[:, 2:])*strides
            boxes[:, :2] -= boxes[:, 2:]/2
            boxes[:, 2:] += boxes[:, :2]
            class_scores = raw[0, :, 4:5]*raw[0, :, 5:]
            labels = class_scores.argmax(axis=-1)
            scores = class_scores[np.arange(len(labels)), labels]
            kept = []
            for label in range(len(category_ids)):
                indices = np.flatnonzero((labels == label) & (scores >= .001))
                kept.extend(indices[xyxy_nms(boxes[indices], scores[indices], nms_iou)])
            order = sorted(kept, key=lambda i: -scores[i])
            boxes, labels, scores = boxes[order], labels[order], scores[order]
            boxes /= transform.scale
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise ValueError("Runtime decoding produced nonfinite detections")
        result = []
        for box, label, score in zip(boxes, labels, scores, strict=True):
            x1, y1, x2, y2 = map(float, box)
            if x2 > x1 and y2 > y1:
                result.append({"category_id": int(category_ids[int(label)]),
                               "bbox": [x1, y1, x2-x1, y2-y1], "score": float(score)})
        return result

    return predict
