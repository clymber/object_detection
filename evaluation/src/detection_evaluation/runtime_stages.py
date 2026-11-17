"""
Persist independently runnable export, parity, accuracy and host timing stages.

Model packages provide conversion and runtime callbacks. No inference framework
is imported here, and validation never calls an exporter.
"""

from __future__ import annotations

import copy
import fcntl
import gc
import platform
import re
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from detection_common.utils.digest import file_digest
from detection_common.utils.json_io import read_json, write_json
from PIL import Image

from . import runtime_workflow as workflow
from .metrics import evaluate_predictions, write_prediction_artifact
from .runtime_workflow import ExportPlan, ExportSet, RuntimeAdapter

STAGES = ("parity", "accuracy", "timing")
TERMINAL = {"completed", "failed", "skipped"}


def load_export_set(manifest_path: Path | str) -> ExportSet:
    """
    Select the current saved export generation without loading any runtimes.
    """
    path = workflow.workspace_path(manifest_path)
    manifest = read_json(path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("model") not in workflow.MODELS
    ):
        raise ValueError("Expected a saved schema-1 export set")
    _legacy_exports(manifest)
    return ExportSet(path, manifest["export_id"])


def read_results(export_set: ExportSet) -> dict:
    """
    Read fresh saved results while rejecting a replaced export generation.
    """
    manifest = read_json(export_set.manifest_path)
    if manifest["export_id"] != export_set.export_id:
        raise ValueError("Selected export set has been replaced")
    _legacy_exports(manifest)
    return {
        **manifest,
        "manifest_path": str(export_set.manifest_path),
        "report_dir": manifest["storage_layout"]["validation"],
        "export_dir": manifest["storage_layout"]["exports"],
    }


def _legacy_exports(manifest: dict) -> None:
    """
    Recover export identities from older combined workflow runtime contracts.
    """
    if "exports" in manifest:
        return
    exports = {}
    for name in ("onnx", "coreml"):
        outcome = manifest["formats"][name]
        if outcome["status"] == "not_requested":
            continue
        artifact = outcome.get("artifact") or outcome.get("contract", {}).get(
            "artifact"
        )
        if artifact is not None:
            _check_record(artifact)
            exports[name] = {
                "status": "exported",
                "artifact": artifact,
                "accepted": False,
            }
        elif outcome["status"] == "failed":
            exports[name] = {
                "status": "failed",
                "accepted": False,
                "error": outcome["error"],
                "failed_stage": outcome["failed_stage"],
            }
        else:
            raise ValueError(f"Missing saved export identity for {name}")
    manifest["exports"] = exports


def _snapshot(run: Path, model: str) -> dict:
    """
    Hash immutable training inputs using paths relative to the selected run.
    """
    return {
        p.relative_to(run).as_posix(): file_digest(p)
        for p in sorted(run.rglob("*"))
        if p.is_file() and workflow.training_source_file(p, run, model)
    }


def _check_record(record: dict) -> Path:
    """
    Verify saved file or package evidence before using it in another stage.
    """
    path = Path(record["path"])
    if workflow.artifact_identity(path) != record:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return path


@contextmanager
def _lock(run: Path, reports: Path) -> Iterator[None]:
    """
    Serialize every writer of a run's canonical export and report set.
    """
    if workflow.export_in_progress(run):
        raise RuntimeError(f"An export is already updating {run}")
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / ".export.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"An export is already updating {run}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _verify_sources(manifest: dict) -> None:
    """
    Reject changed training inputs, configuration and local data before writing.
    """
    run = Path(manifest["run_dir"])
    protocol = workflow.validate_run(run, manifest["model"])
    if (
        workflow.artifact_identity(workflow.checkpoint_path(run, manifest["model"]))
        != manifest["checkpoint"]
        or file_digest(run / "run_protocol.json") != manifest["run_protocol_sha256"]
    ):
        raise ValueError("Export source changed before validation")
    if (
        workflow.validate_dataset(protocol, Path(manifest["dataset"]["directory"]))
        != manifest["dataset"]
    ):
        raise ValueError("Export dataset changed before validation")
    if (
        manifest.get("configuration_sha256") is not None
        and workflow.digest_json(manifest["configuration"])
        != manifest["configuration_sha256"]
    ):
        raise ValueError("Export configuration changed before validation")
    snapshot = read_json(_check_record(manifest["source_snapshot"]))["files"]
    if _snapshot(run, manifest["model"]) != snapshot:
        raise ValueError("Training source files changed since export")
    if not manifest["source_run_unchanged"]:
        raise ValueError("Export modified training source files")
    _check_record(manifest["parity_manifest"])
    if "timing_sequence" in manifest:
        _check_record(manifest["timing_sequence"])


def _record_source_state(manifest: dict) -> None:
    """
    Preserve evidence of source mutations even when a callback raises.
    """
    before = read_json(Path(manifest["source_snapshot"]["path"]))["files"]
    after = _snapshot(Path(manifest["run_dir"]), manifest["model"])
    changed = sorted(
        k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
    )
    manifest["source_run_unchanged"] = not changed
    if changed:
        manifest["source_mutations"] = changed


def _pending_stages() -> dict:
    """
    Allocate independent status records for one runtime's validation stages.
    """
    return {stage: {"status": "pending"} for stage in STAGES}


def _hydrate_stages(manifest: dict) -> None:
    """
    Resume older two-cell manifests using only verified completed evidence.
    """
    for name, outcome in manifest["formats"].items():
        if name == "pytorch" and outcome["status"] == "not_requested":
            # Older export-only manifests defer the reference until validation.
            outcome["status"] = "pending"
        if outcome["status"] == "not_requested" or "stages" in outcome:
            continue
        stages = outcome["stages"] = _pending_stages()
        if name != "pytorch" and manifest["exports"][name]["status"] != "exported":
            for stage in stages.values():
                stage.update(status="skipped", reason="Model export failed")
            continue
        if "parity" in outcome:
            parity = read_json(_check_record(outcome["parity"]))
            if parity["status"] != outcome["parity_status"]:
                raise ValueError("Parity status differs from evidence")
            stages["parity"]["status"] = "completed"
        splits = set(manifest.get("accuracy_splits", ("val", "test")))
        if set(outcome.get("predictions", {})) == splits:
            for record in outcome["predictions"].values():
                _check_record(record)
            if set(outcome.get("metrics", {})) == splits:
                stages["accuracy"]["status"] = "completed"
        if "timing" in outcome:
            _check_record(outcome["timing"])
            stages["timing"]["status"] = "completed"
        elif stages["parity"]["status"] == "completed" and outcome[
            "parity_status"
        ] not in {"passed", "reference"}:
            stages["timing"].update(status="skipped", reason="Parity did not pass")
        failed = {
            "full_split_evaluation": "accuracy",
            "host_timing": "timing",
            "parity": "parity",
        }.get(outcome.get("failed_stage"))
        if failed:
            stages[failed].update(status="failed", error=outcome.get("error"))


def _refresh_outcomes(manifest: dict) -> None:
    """
    Project stage records into the existing comparison-compatible outcome fields.
    """
    complete = True
    for name, outcome in manifest["formats"].items():
        if outcome["status"] == "not_requested":
            continue
        stages = outcome["stages"]
        done = all(value["status"] in TERMINAL for value in stages.values())
        complete &= done
        outcome["accepted"] = False
        exported = manifest["exports"].get(name)
        outcome.pop("error", None)
        outcome.pop("failed_stage", None)
        if exported is not None and exported["status"] != "exported":
            outcome.update(
                status="failed",
                error=exported["error"],
                failed_stage=exported["failed_stage"],
            )
            continue
        failed = next((s for s in STAGES if stages[s]["status"] == "failed"), None)
        if failed:
            outcome.update(
                status="failed",
                error=stages[failed]["error"],
                failed_stage=stages[failed].get("failed_stage", failed),
            )
        else:
            outcome["status"] = "completed" if done else "pending"
            outcome["accepted"] = (
                done
                and outcome.get("parity_status") in {"reference", "passed"}
                and all(stages[s]["status"] == "completed" for s in STAGES)
            )
    manifest["attempt_status"] = "finished" if complete else "validating"


@contextmanager
def _validation(export_set: ExportSet) -> Iterator[dict]:
    """
    Verify and finalize a single stage without discarding other stage results.
    """
    selected = read_results(export_set)
    with _lock(Path(selected["run_dir"]), export_set.manifest_path.parent):
        manifest = read_results(export_set)
        _verify_sources(manifest)
        _hydrate_stages(manifest)
        try:
            yield manifest
        finally:
            _record_source_state(manifest)
            _refresh_outcomes(manifest)
            workflow._write_manifest(export_set.manifest_path, _stored(manifest))


def _stored(result: dict) -> dict:
    """
    Exclude convenience result paths from the persisted manifest document.
    """
    return {
        k: v
        for k, v in result.items()
        if k not in {"manifest_path", "report_dir", "export_dir"}
    }


def _initialize_manifest(plan: ExportPlan, reports: Path, exports: Path) -> dict:
    """
    Capture the immutable source and fixed validation selections for new exports.
    """
    protocol = workflow.validate_run(plan.run_dir, plan.model)
    if protocol != plan.protocol:
        raise ValueError("Training protocol changed after preparation")
    identity = workflow.validate_dataset(protocol, plan.dataset_dir)
    documents = {s: read_json(v["path"]) for s, v in identity["annotations"].items()}
    ids = sorted(i["id"] for i in documents["test"]["images"])
    if not ids:
        raise ValueError("Cannot time an empty test split")
    snapshot = reports / "source_run_checksums.json"
    write_json(snapshot, {"files": _snapshot(plan.run_dir, plan.model)})
    parity = reports / "development_parity_manifest.json"
    write_json(
        parity,
        workflow.development_selection(
            documents["val"], identity["annotations"]["val"]["sha256"]
        ),
    )
    sequence = reports / "host_timing_sequence.json"
    timing = plan.configuration.get("host_timing", workflow.HOST_TIMING)
    write_json(sequence, {
        "image_ids": [ids[i % len(ids)] for i in range(timing["samples"])]
    })
    configuration = copy.deepcopy(plan.configuration)
    return {
        "schema_version": 1,
        "attempt_status": "running",
        "export_id": workflow.utc_id(),
        "model": plan.model,
        "accuracy_splits": plan.configuration.get("accuracy_splits", ["test"]),
        "host_timing": timing,
        "storage_policy": "one export set per training run; no attempt directory",
        "storage_layout": {"exports": str(exports), "validation": str(reports)},
        "source_immutability_scope": "training files; managed exports and export_validation excluded",
        "run_dir": str(plan.run_dir),
        "dataset": identity,
        "run_protocol_sha256": file_digest(plan.run_dir / "run_protocol.json"),
        "checkpoint": workflow.artifact_identity(
            workflow.checkpoint_path(plan.run_dir, plan.model)
        ),
        "source_snapshot": workflow.artifact_identity(snapshot),
        "parity_manifest": workflow.artifact_identity(parity),
        "timing_sequence": workflow.artifact_identity(sequence),
        "configuration": configuration,
        "configuration_sha256": workflow.digest_json(configuration),
        "tool_versions": workflow.resolved_versions(),
        "source_run_unchanged": True,
        "exports": {},
        "formats": {
            name: (
                {"status": "pending", "accepted": False, "stages": _pending_stages()}
                if name == "pytorch" or name in plan.formats
                else {"status": "not_requested"}
            )
            for name in workflow.FORMATS
        },
    }


def _cleanup_previous(plan: ExportPlan, reports: Path, exports: Path) -> None:
    """
    Remove obsolete reports and legacy attempt directories for an explicit export.
    """
    for name in workflow.FORMATS:
        directory = reports / name
        if directory.is_dir():
            shutil.rmtree(directory)
    for base in (reports, exports):
        for previous in base.iterdir():
            if (
                previous.is_dir()
                and not previous.is_symlink()
                and re.fullmatch(r"\d{8}T\d{6,12}Z_[0-9a-f]{8,32}", previous.name)
            ):
                shutil.rmtree(previous)
    for name in ("onnx", "coreml"):
        alias = workflow.export_artifact_path(plan.run_dir, plan.model, name)
        if alias.is_symlink() and not alias.exists():
            alias.unlink()


def _require_host(name: str) -> None:
    """
    Report unsupported Core ML operations as independent format failures.
    """
    if name == "coreml" and platform.system() != "Darwin":
        raise RuntimeError(
            "Core ML export/runtime requires macOS; request ONNX on Renku"
        )


def export_models(plan: ExportPlan, exporter: Callable) -> ExportSet:
    """
    Publish requested artifacts without reference inference, evaluation or timing.
    """
    hierarchy = Path("basketball") / plan.model / plan.run_dir.name
    reports = plan.output_root / "exports" / hierarchy
    exports = plan.models_root / "exports" / hierarchy
    with _lock(plan.run_dir, reports):
        exports.mkdir(parents=True, exist_ok=True)
        manifest = _initialize_manifest(plan, reports, exports)
        _cleanup_previous(plan, reports, exports)
        workflow._replace_link(plan.run_dir / "export_validation", reports)
        path = reports / "model_manifest.json"
        workflow._write_manifest(path, manifest)
        try:
            for name in plan.formats:
                try:
                    _require_host(name)
                    artifact = exporter(name, exports / name)
                    _check_record(artifact)
                    outcome = {
                        "status": "exported",
                        "artifact": artifact,
                        "accepted": False,
                    }
                except Exception as error:  # noqa: BLE001
                    # Persist converter failures and continue with the other formats.
                    outcome = {
                        "status": "failed",
                        "accepted": False,
                        "failed_stage": "conversion_and_loading",
                        "error": f"{type(error).__name__}: {error}",
                    }
                manifest["exports"][name] = outcome
                manifest["formats"][name].update(outcome)
                workflow._write_manifest(path, manifest)
        finally:
            _record_source_state(manifest)
            manifest["attempt_status"] = "exported"
            workflow._write_manifest(path, manifest)
    return ExportSet(path, manifest["export_id"])


def _reset_stage(outcome: dict, directory: Path, stage: str) -> None:
    """
    Clear only the selected stage's evidence and any dependent timing results.
    """
    keys = {
        "parity": ("parity", "parity_status"),
        "accuracy": ("predictions", "metrics"),
        "timing": ("timing",),
    }
    patterns = {
        "parity": ("parity.json",),
        "accuracy": ("*_predictions.json", "*_metrics.json", "*_comparison.jpg"),
        "timing": ("host_timing.json",),
    }
    for key in keys[stage]:
        outcome.pop(key, None)
    for pattern in patterns[stage]:
        for path in directory.glob(pattern):
            path.unlink()
    outcome["stages"][stage] = {"status": "pending"}
    if stage != "timing":
        _reset_stage(outcome, directory, "timing")
    outcome.pop("timing_unavailable_reason", None)


def _timing_eligible(outcome: dict) -> bool:
    """
    Require existing successful parity and accuracy without running either stage.
    """
    prerequisites = [outcome["stages"][s]["status"] for s in ("parity", "accuracy")]
    if any(status not in TERMINAL for status in prerequisites):
        reason = "Run parity and accuracy before host timing"
        outcome["stages"]["timing"].update(status="pending", reason=reason)
    elif prerequisites != ["completed", "completed"] or outcome.get(
        "parity_status"
    ) not in {"passed", "reference"}:
        reason = "Parity or accuracy failed; accuracy remains diagnostic"
        outcome["stages"]["timing"].update(status="skipped", reason=reason)
    else:
        for key in ("parity", "timing"):
            if key in outcome:
                _check_record(outcome[key])
        for record in outcome.get("predictions", {}).values():
            _check_record(record)
        return True
    outcome["timing_unavailable_reason"] = reason
    return False


def _runtime(manifest: dict, name: str, loader: Callable) -> RuntimeAdapter:
    """
    Verify a saved graph before constructing a fresh runtime with a stable contract.
    """
    _require_host(name)
    if name != "pytorch":
        _check_record(manifest["exports"][name]["artifact"])
    adapter = loader(name, Path(manifest["storage_layout"]["exports"]) / name)
    expected = (
        manifest["checkpoint"]
        if name == "pytorch"
        else manifest["exports"][name]["artifact"]
    )
    if adapter.contract["artifact"] != expected:
        raise ValueError("Runtime loaded a different artifact")
    previous = manifest["formats"][name].get("contract")
    if previous is not None and previous != adapter.contract:
        raise ValueError("Runtime contract changed between validation stages")
    manifest["formats"][name]["contract"] = adapter.contract
    return adapter


def _run_stage(
    export_set: ExportSet, loader: Callable, stage: str, action: Callable
) -> dict:
    """
    Execute exactly one validation stage while persisting independent outcomes.
    """
    with _validation(export_set) as manifest:
        reference = {}
        for name in workflow.FORMATS:
            outcome = manifest["formats"][name]
            if outcome["status"] == "not_requested":
                continue
            exported = manifest["exports"].get(name)
            if exported is not None and exported["status"] != "exported":
                outcome["stages"][stage] = {
                    "status": "skipped",
                    "reason": "Model export failed",
                }
                continue
            directory = Path(manifest["report_dir"]) / name
            directory.mkdir(exist_ok=True)
            _reset_stage(outcome, directory, stage)
            record = outcome["stages"][stage]
            adapter = None
            try:
                if stage == "timing" and not _timing_eligible(outcome):
                    continue
                record.update(status="running", failed_stage="artifact_verification")
                manifest["attempt_status"] = "running"
                workflow._write_manifest(export_set.manifest_path, _stored(manifest))
                adapter = _runtime(manifest, name, loader)
                record["failed_stage"] = stage
                print(f"{manifest['model']}/{name}: {stage}", flush=True)
                action(manifest, name, adapter, directory, reference)
                record.update(status="completed")
                record.pop("failed_stage", None)
            except Exception as error:  # noqa: BLE001
                # Save runtime failures without discarding independent evidence.
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
            finally:
                outcome.pop("progress", None)
                del adapter
                gc.collect()
                _refresh_outcomes(manifest)
                workflow._write_manifest(export_set.manifest_path, _stored(manifest))
    return read_results(export_set)


def _parity(
    manifest: dict, name: str, adapter: RuntimeAdapter, directory: Path, reference: dict
) -> None:
    """
    Save development predictions and compare them to this stage's fresh reference.
    """
    selection = read_json(manifest["parity_manifest"]["path"])
    truth = read_json(manifest["dataset"]["annotations"]["val"]["path"])
    images = {i["id"]: i for i in truth["images"]}
    rows = {}
    for image_id in selection["image_ids"]:
        path = (
            Path(manifest["dataset"]["directory"])
            / "images/val"
            / images[image_id]["file_name"]
        )
        with Image.open(path) as source:
            rows[image_id] = adapter.predict(source.convert("RGB"))
    if name == "pytorch":
        reference["rows"] = rows
        parity = {"status": "reference", "images": []}
    elif "rows" not in reference:
        parity = {"status": "unavailable", "reason": "checkpoint reference failed"}
    else:
        checks = [
            {
                "image_id": i,
                **workflow.compare_detections(
                    reference["rows"][i],
                    rows[i],
                    images[i]["width"],
                    images[i]["height"],
                ),
            }
            for i in selection["image_ids"]
        ]
        parity = {
            "status": "passed" if all(c["passed"] for c in checks) else "failed",
            "images": checks,
        }
    parity.update(detections=rows, reference_detections=reference.get("rows"))
    write_json(directory / "parity.json", parity)
    manifest["formats"][name].update(
        parity_status=parity["status"],
        parity=workflow.artifact_identity(directory / "parity.json"),
    )


def _evaluate_split(
    manifest: dict,
    name: str,
    adapter: RuntimeAdapter,
    directory: Path,
    split: str,
    annotation: dict,
) -> None:
    """
    Save complete source-coordinate predictions, metrics and fixed comparison tiles.
    """
    truth = read_json(annotation["path"])
    outcome = manifest["formats"][name]
    predictions = []
    images = sorted(truth["images"], key=lambda i: i["id"])
    for index, entry in enumerate(images):
        path = (
            Path(manifest["dataset"]["directory"])
            / "images"
            / split
            / entry["file_name"]
        )
        with Image.open(path) as source:
            rows = adapter.predict(source.convert("RGB"))
        predictions.extend({**row, "image_id": entry["id"]} for row in rows)
        if (index + 1) % 100 == 0:
            outcome["progress"] = {
                "split": split,
                "processed": index + 1,
                "total": len(images),
            }
            workflow._write_manifest(Path(manifest["manifest_path"]), _stored(manifest))
    config = manifest["configuration"]
    metadata = {
        "model": manifest["model"],
        "split": split,
        "run_dir": manifest["run_dir"],
        "checkpoint": manifest["checkpoint"]["path"],
        "smoke_run": False,
        "resolution": config["resolution"],
        "dataset_fingerprint": manifest["dataset"]["fingerprint"],
        "runtime": name,
        "runtime_contract": adapter.contract,
        "postprocessing": config["postprocessing"],
    }
    path = write_prediction_artifact(
        directory / f"{split}_predictions.json",
        annotation["path"],
        predictions,
        metadata=metadata,
    )
    outcome["predictions"][split] = workflow.artifact_identity(path)
    metrics = evaluate_predictions(annotation["path"], predictions)
    write_json(directory / f"{split}_metrics.json", metrics)
    outcome["metrics"][split] = metrics
    positive = {a["image_id"] for a in truth["annotations"]}
    ids = [i["id"] for i in images]
    samples = [i for i in ids if i in positive][:4] + [
        i for i in ids if i not in positive
    ][:4]
    samples += [i for i in ids if i not in samples][: 8 - len(samples)]
    workflow.qualitative_grid(
        directory / f"{split}_comparison.jpg",
        truth,
        predictions,
        Path(manifest["dataset"]["directory"]),
        split,
        samples,
    )
    workflow._write_manifest(Path(manifest["manifest_path"]), _stored(manifest))


def _accuracy(
    manifest: dict, name: str, adapter: RuntimeAdapter, directory: Path, reference: dict
) -> None:
    """
    Evaluate full splits independently of parity outcomes or host timing.
    """
    manifest["formats"][name].update(predictions={}, metrics={})
    for split, annotation in manifest["dataset"]["annotations"].items():
        if split in manifest.get("accuracy_splits", ("val", "test")):
            _evaluate_split(manifest, name, adapter, directory, split, annotation)


def _timing(
    manifest: dict, name: str, adapter: RuntimeAdapter, directory: Path, reference: dict
) -> None:
    """
    Measure the unchanged host protocol on the saved predecoded test sequence.
    """
    sequence = read_json(Path(manifest["report_dir"]) / "host_timing_sequence.json")[
        "image_ids"
    ]
    timing = manifest.get("host_timing", {"warmups": 30, "repetitions": 3})
    truth = read_json(manifest["dataset"]["annotations"]["test"]["path"])
    entries = {i["id"]: i for i in truth["images"]}
    pixels = {}
    for image_id in set(sequence):
        path = (
            Path(manifest["dataset"]["directory"])
            / "images/test"
            / entries[image_id]["file_name"]
        )
        with Image.open(path) as source:
            pixels[image_id] = source.convert("RGB")
    timing = workflow.host_timing(
        adapter.predict,
        pixels,
        sequence,
        backend={
            k: adapter.contract[k]
            for k in ("backend", "device", "precision", "compute_units", "cpu_threads")
            if k in adapter.contract
        },
        synchronize=adapter.synchronize,
        warmups=timing["warmups"],
        repetitions=timing["repetitions"],
    )
    write_json(directory / "host_timing.json", timing)
    manifest["formats"][name]["timing"] = workflow.artifact_identity(
        directory / "host_timing.json"
    )


def check_parity(export_set: ExportSet, loader: Callable) -> dict:
    """
    Run development parity only, preserving accuracy and invalidating host timing.
    """
    return _run_stage(export_set, loader, "parity", _parity)


def evaluate_accuracy(export_set: ExportSet, loader: Callable) -> dict:
    """
    Run full-split accuracy only, preserving parity and invalidating host timing.
    """
    return _run_stage(export_set, loader, "accuracy", _accuracy)


def benchmark_host(export_set: ExportSet, loader: Callable) -> dict:
    """
    Run host timing only for runtimes with successful saved parity and accuracy.
    """
    return _run_stage(export_set, loader, "timing", _timing)
