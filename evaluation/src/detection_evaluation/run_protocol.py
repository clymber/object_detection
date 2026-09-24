"""
Persist framework-neutral training runs and atomically published evaluation bundles.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from detection_common.utils.json_io import json_ready, read_json

from .metrics import file_sha256, read_prediction_artifact

RUN_PROTOCOL_FILE_NAME = "run_protocol.json"
TRAINING_FILE_NAME = "training.json"
BUNDLE_MANIFEST_FILE_NAME = "manifest.json"
RUN_PROTOCOL_SCHEMA_VERSION = 1
BUNDLE_SCHEMA_VERSION = 1
TIMING_PROTOCOL = "monotonic-synchronized-v1"
_ATTEMPT_LOCKS: dict[str, int] = {}


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    """
    Encode a JSON-compatible document in the project's stable representation.
    """
    return (
        json.dumps(json_ready(document), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _clone(document: Mapping[str, Any]) -> dict[str, Any]:
    """
    Make an independent JSON-compatible copy of a mapping.
    """
    return json.loads(_json_bytes(document))


def _valid_sha256(value: Any) -> bool:
    """
    Return whether a value is a lowercase SHA256 digest.
    """
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(path: Path) -> None:
    """
    Flush a directory entry update to the filesystem.
    """
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_fsynced(path: Path, document: Mapping[str, Any] | bytes) -> None:
    """
    Write one file and synchronously flush its contents.
    """
    payload = document if isinstance(document, bytes) else _json_bytes(document)
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _replace_fsynced(path: Path, document: Mapping[str, Any]) -> None:
    """
    Atomically replace one JSON file and flush its parent directory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_fsynced(temporary, document)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _capture_immutable(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    """
    Create a document once or verify that an existing copy is identical.
    """
    expected = _clone(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_fsynced(temporary, expected)
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            recorded = read_json(path)
            if recorded != expected:
                raise ValueError(f"Immutable protocol differs at {path}")
            return recorded
    finally:
        if temporary.exists():
            temporary.unlink()
    return expected


def _utc_timestamp(value: str | datetime) -> str:
    """
    Normalize an aware UTC datetime into a portable protocol timestamp.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("original_utc must be an ISO UTC timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("original_utc must include the UTC offset")
    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _nonempty_string(value: Any, name: str) -> str:
    """
    Require a nonempty protocol string field.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validate_dataset_identity(identity: Any) -> dict[str, Any]:
    """
    Require captured canonical and loader identities with matching source hashes.
    """
    if not isinstance(identity, Mapping):
        raise ValueError("dataset_identity must be an object")
    canonical = identity.get("canonical")
    loader = identity.get("loader")
    if not isinstance(canonical, Mapping) or not isinstance(loader, Mapping):
        raise ValueError("dataset_identity requires canonical and loader objects")
    source_fingerprint = canonical.get("source_fingerprint")
    if not _valid_sha256(source_fingerprint):
        raise ValueError("Canonical dataset identity requires source_fingerprint")
    if loader.get("source_fingerprint") != source_fingerprint:
        raise ValueError("Canonical and loader source fingerprints differ")
    if not _valid_sha256(loader.get("loader_fingerprint")):
        raise ValueError("Loader dataset identity requires loader_fingerprint")
    return _clone({"canonical": canonical, "loader": loader})


def _validate_run_protocol(protocol: Any) -> dict[str, Any]:
    """
    Validate the immutable run provenance captured before training begins.
    """
    if not isinstance(protocol, Mapping):
        raise ValueError("Run protocol must be an object")
    required = {
        "schema_version",
        "logical_dataset",
        "model",
        "source_notebook",
        "original_utc",
        "training_settings",
        "smoke_run",
        "dataset_identity",
    }
    if set(protocol) != required:
        raise ValueError("Run protocol has missing or unsupported fields")
    if protocol["schema_version"] != RUN_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("Unsupported run protocol schema")
    if not isinstance(protocol["training_settings"], Mapping):
        raise ValueError("training_settings must be an object")
    normalized = {
        "schema_version": RUN_PROTOCOL_SCHEMA_VERSION,
        "logical_dataset": _nonempty_string(
            protocol["logical_dataset"], "logical_dataset"
        ),
        "model": _nonempty_string(protocol["model"], "model"),
        "source_notebook": _nonempty_string(
            protocol["source_notebook"], "source_notebook"
        ),
        "original_utc": _utc_timestamp(protocol["original_utc"]),
        "training_settings": _clone(protocol["training_settings"]),
        "smoke_run": protocol["smoke_run"],
        "dataset_identity": _validate_dataset_identity(protocol["dataset_identity"]),
    }
    if type(normalized["smoke_run"]) is not bool:
        raise ValueError("smoke_run must be an explicit boolean")
    return normalized


def create_run_protocol(
    run_dir: Path | str,
    *,
    logical_dataset: str,
    model: str,
    source_notebook: str,
    original_utc: str | datetime,
    training_settings: Mapping[str, Any],
    smoke_run: bool,
    canonical_dataset_identity: Mapping[str, Any],
    loader_dataset_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Capture immutable run provenance before the first timed training invocation.

    Calling this for an existing run verifies every field. That preserves a
    smoke run's identity across resume and recovery instead of permitting it to
    be relabeled as a full run.
    """
    document = _validate_run_protocol(
        {
            "schema_version": RUN_PROTOCOL_SCHEMA_VERSION,
            "logical_dataset": logical_dataset,
            "model": model,
            "source_notebook": source_notebook,
            "original_utc": original_utc,
            "training_settings": training_settings,
            "smoke_run": smoke_run,
            "dataset_identity": {
                "canonical": canonical_dataset_identity,
                "loader": loader_dataset_identity,
            },
        }
    )
    return _capture_immutable(Path(run_dir) / RUN_PROTOCOL_FILE_NAME, document)


def read_run_protocol(run_dir: Path | str) -> dict[str, Any]:
    """
    Read and validate the immutable metadata for one protocol-bearing run.
    """
    return _validate_run_protocol(read_json(Path(run_dir) / RUN_PROTOCOL_FILE_NAME))


def capture_training_hardware(
    *, device: str, details: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """
    Capture training hardware independently of any inference benchmark.
    """
    return {
        "device": _nonempty_string(device, "device"),
        "host": platform.node() or "unknown",
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "processor": platform.processor() or "unknown",
        "details": _clone(details or {}),
    }


def _protocol_digest(protocol: Mapping[str, Any]) -> str:
    """
    Return the stable SHA256 identifier for immutable run provenance.
    """
    return hashlib.sha256(_json_bytes(protocol)).hexdigest()


def _summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate finalized attempts without reconstructing unfinished durations.
    """
    finalized = [attempt for attempt in attempts if attempt["status"] != "running"]
    completed_epochs = sum(attempt["completed_epochs"] for attempt in finalized)
    if any(attempt["status"] in {"running", "incomplete"} for attempt in attempts):
        return {
            "status": "incomplete",
            "timing_available": False,
            "total_seconds": None,
            "total_hours": None,
            "completed_epochs": completed_epochs,
            "amortized_seconds_per_completed_epoch": None,
        }
    total_seconds = sum(attempt["duration_seconds"] for attempt in finalized)
    return {
        "status": finalized[-1]["status"] if finalized else "completed",
        "timing_available": True,
        "total_seconds": total_seconds,
        "total_hours": total_seconds / 3600,
        "completed_epochs": completed_epochs,
        "amortized_seconds_per_completed_epoch": (
            total_seconds / completed_epochs if completed_epochs else None
        ),
    }


def _validate_attempt(attempt: Any) -> dict[str, Any]:
    """
    Validate a started or finalized training attempt.
    """
    if not isinstance(attempt, Mapping):
        raise ValueError("Training attempt must be an object")
    required = {
        "id",
        "status",
        "kind",
        "started_monotonic_seconds",
        "timing_protocol",
        "training_hardware",
    }
    if attempt.get("status") in {"completed", "interrupted", "incomplete"}:
        required |= {"duration_seconds", "completed_epochs"}
    if set(attempt) != required:
        raise ValueError("Training attempt has missing or unsupported fields")
    if attempt["status"] not in {"running", "completed", "interrupted", "incomplete"}:
        raise ValueError("Training attempt has an invalid status")
    if attempt["kind"] not in {"fresh", "resume"}:
        raise ValueError("Training attempt kind must be fresh or resume")
    if attempt["timing_protocol"] != TIMING_PROTOCOL:
        raise ValueError("Training attempt uses an unsupported timing protocol")
    if not isinstance(attempt["id"], str) or not attempt["id"]:
        raise ValueError("Training attempt requires an ID")
    started = attempt["started_monotonic_seconds"]
    if (
        isinstance(started, bool)
        or not isinstance(started, (int, float))
        or not math.isfinite(started)
    ):
        raise ValueError("Training attempt requires a finite monotonic start time")
    hardware = attempt["training_hardware"]
    if not isinstance(hardware, Mapping) or not isinstance(hardware.get("device"), str):
        raise ValueError("Training attempt requires captured training hardware")
    normalized = _clone(attempt)
    if attempt["status"] != "running":
        duration = attempt["duration_seconds"]
        epochs = attempt["completed_epochs"]
        if attempt["status"] == "incomplete":
            if duration is not None:
                raise ValueError("Incomplete attempt cannot have a guessed duration")
        elif (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("Finalized attempt requires a finite nonnegative duration")
        if type(epochs) is not int or epochs < 0:
            raise ValueError("Finalized attempt requires nonnegative completed_epochs")
    return normalized


def _validate_training_document(
    document: Any,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Validate attempt state and its derived summary against run provenance.
    """
    if not isinstance(document, Mapping):
        raise ValueError("Training metadata must be an object")
    if set(document) != {"schema_version", "run_protocol_sha256", "attempts", "summary"}:
        raise ValueError("Training metadata has missing or unsupported fields")
    if document["schema_version"] != RUN_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("Unsupported training metadata schema")
    if document["run_protocol_sha256"] != _protocol_digest(protocol):
        raise ValueError("Training metadata belongs to different run provenance")
    if not isinstance(document["attempts"], list):
        raise ValueError("Training metadata requires an attempts list")
    attempts = [_validate_attempt(attempt) for attempt in document["attempts"]]
    if len({attempt["id"] for attempt in attempts}) != len(attempts):
        raise ValueError("Training attempt IDs must be unique")
    summary = _summary(attempts)
    if document["summary"] != summary:
        raise ValueError("Training metadata summary does not match its attempts")
    return {
        "schema_version": RUN_PROTOCOL_SCHEMA_VERSION,
        "run_protocol_sha256": _protocol_digest(protocol),
        "attempts": attempts,
        "summary": summary,
    }


def _initial_training_document(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return an empty, valid timing record for a new run.
    """
    attempts: list[dict[str, Any]] = []
    return {
        "schema_version": RUN_PROTOCOL_SCHEMA_VERSION,
        "run_protocol_sha256": _protocol_digest(protocol),
        "attempts": attempts,
        "summary": _summary(attempts),
    }


def read_training_record(run_dir: Path | str) -> dict[str, Any]:
    """
    Read training attempts and their non-reconstructed timing summary.
    """
    protocol = read_run_protocol(run_dir)
    return _validate_training_document(
        read_json(Path(run_dir) / TRAINING_FILE_NAME), protocol
    )


def start_training_attempt(
    run_dir: Path | str,
    *,
    resumed: bool,
    training_hardware: Mapping[str, Any],
    completed_epochs_before: int = 0,
    monotonic_clock: Callable[[], float] = time.monotonic,
    synchronize: Callable[[], None] | None = None,
) -> str:
    """
    Hold an exclusive run lock until finalization, recovering only dead attempts.

    The OS releases the lock after a hard termination. A subsequent explicit
    resume keeps that attempt's duration unknown and starts a new timed attempt.
    Epoch counts may come from history; elapsed time is never reconstructed.
    """
    if type(completed_epochs_before) is not int or completed_epochs_before < 0:
        raise ValueError("completed_epochs_before must be a nonnegative integer")
    descriptor = os.open(Path(run_dir) / ".training.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Cannot resume while an unfinished trainer is active") from error
        attempt_id = _start_training_attempt(
            run_dir,
            resumed=resumed,
            training_hardware=training_hardware,
            completed_epochs_before=completed_epochs_before,
            monotonic_clock=monotonic_clock,
            synchronize=synchronize,
        )
        _ATTEMPT_LOCKS[attempt_id] = descriptor
        return attempt_id
    except BaseException:
        os.close(descriptor)
        raise


def _start_training_attempt(
    run_dir: Path | str,
    *,
    resumed: bool,
    completed_epochs_before: int,
    training_hardware: Mapping[str, Any],
    monotonic_clock: Callable[[], float] = time.monotonic,
    synchronize: Callable[[], None] | None = None,
) -> str:
    """
    Persist an attempt immediately before the synchronized training boundary.
    """
    root = Path(run_dir)
    protocol = read_run_protocol(root)
    state_path = root / TRAINING_FILE_NAME
    state = (
        _validate_training_document(read_json(state_path), protocol)
        if state_path.exists()
        else _initial_training_document(protocol)
    )
    unfinished = [a for a in state["attempts"] if a["status"] == "running"]
    if unfinished:
        if not resumed or len(unfinished) != 1:
            raise ValueError("An unfinished attempt requires explicit resume")
        recorded_epochs = sum(
            a["completed_epochs"] for a in state["attempts"]
            if a["status"] != "running"
        )
        unfinished[0].update(
            status="incomplete",
            duration_seconds=None,
            completed_epochs=max(0, completed_epochs_before - recorded_epochs),
        )
    if synchronize is not None:
        synchronize()
    started = monotonic_clock()
    if (
        isinstance(started, bool)
        or not isinstance(started, (int, float))
        or not math.isfinite(started)
    ):
        raise ValueError("monotonic_clock must return a finite number")
    attempt_id = uuid.uuid4().hex
    state["attempts"].append(
        {
            "id": attempt_id,
            "status": "running",
            "kind": "resume" if resumed else "fresh",
            "started_monotonic_seconds": float(started),
            "timing_protocol": TIMING_PROTOCOL,
            "training_hardware": _clone(training_hardware),
        }
    )
    state["summary"] = _summary(state["attempts"])
    _replace_fsynced(state_path, state)
    return attempt_id


def finalize_training_attempt(
    run_dir: Path | str,
    attempt_id: str,
    *,
    outcome: str,
    completed_epochs: int,
    monotonic_clock: Callable[[], float] = time.monotonic,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """
    Finalize the owned attempt and release its training lock, including on error.
    """
    try:
        return _finalize_training_attempt(
            run_dir, attempt_id, outcome=outcome,
            completed_epochs=completed_epochs, monotonic_clock=monotonic_clock,
            synchronize=synchronize,
        )
    finally:
        descriptor = _ATTEMPT_LOCKS.pop(attempt_id, None)
        if descriptor is not None:
            os.close(descriptor)


def _finalize_training_attempt(
    run_dir: Path | str,
    attempt_id: str,
    *,
    outcome: str,
    completed_epochs: int,
    monotonic_clock: Callable[[], float] = time.monotonic,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """
    Finalize a completed or catchably interrupted attempt exactly once.
    """
    if outcome not in {"completed", "interrupted"}:
        raise ValueError("outcome must be completed or interrupted")
    if type(completed_epochs) is not int or completed_epochs < 0:
        raise ValueError("completed_epochs must be a nonnegative integer")
    root = Path(run_dir)
    protocol = read_run_protocol(root)
    state_path = root / TRAINING_FILE_NAME
    state = _validate_training_document(read_json(state_path), protocol)
    matches = [attempt for attempt in state["attempts"] if attempt["id"] == attempt_id]
    if len(matches) != 1:
        raise ValueError("Unknown training attempt")
    attempt = matches[0]
    if attempt["status"] != "running":
        raise ValueError("Training attempt has already been finalized")
    if synchronize is not None:
        synchronize()
    finished = monotonic_clock()
    if (
        isinstance(finished, bool)
        or not isinstance(finished, (int, float))
        or not math.isfinite(finished)
    ):
        raise ValueError("monotonic_clock must return a finite number")
    duration = float(finished) - attempt["started_monotonic_seconds"]
    if duration < 0:
        raise ValueError("Monotonic clock moved backwards")
    attempt.update(
        {
            "status": outcome,
            "duration_seconds": duration,
            "completed_epochs": completed_epochs,
        }
    )
    state["summary"] = _summary(state["attempts"])
    _replace_fsynced(state_path, state)
    return state["summary"]


def _invoke_failpoint(
    failpoint: Callable[[str], None] | None, phase: str
) -> None:
    """
    Invoke an optional test hook at a publication durability boundary.
    """
    if failpoint is not None:
        failpoint(phase)


def _copy_fsynced(source: Path, destination: Path) -> None:
    """
    Copy an immutable artifact into a staging generation and flush it.
    """
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def _require_completed_training(training: Mapping[str, Any]) -> None:
    """
    Accept completed training even when an older crashed attempt lacks timing.
    """
    attempts = training["attempts"]
    if (
        not attempts
        or any(a["status"] == "running" for a in attempts)
        or attempts[-1]["status"] != "completed"
    ):
        raise ValueError("Bundle requires completed training; unfinished training attempt")


def _validate_bundle_inputs(
    protocol: Mapping[str, Any],
    training: Mapping[str, Any],
    val_artifact: Mapping[str, Any],
    test_artifact: Mapping[str, Any],
    checkpoint: Path,
    resolution: int,
    parameter_count: int,
) -> str:
    """
    Verify one coherent model run before it is published as a generation.
    """
    _require_completed_training(training)
    if type(resolution) is not int or resolution <= 0:
        raise ValueError("resolution must be a positive integer")
    if type(parameter_count) is not int or parameter_count <= 0:
        raise ValueError("parameter_count must be a positive integer")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_digest = file_sha256(checkpoint)
    for split, artifact in (("val", val_artifact), ("test", test_artifact)):
        metadata = artifact["metadata"]
        if metadata["split"] != split:
            raise ValueError(f"{split} artifact has the wrong split")
        if metadata["model"] != protocol["model"]:
            raise ValueError(f"{split} artifact model differs from run provenance")
        if metadata["smoke_run"] != protocol["smoke_run"]:
            raise ValueError(f"{split} artifact smoke flag differs from run provenance")
        if metadata["resolution"] != resolution:
            raise ValueError(f"{split} artifact resolution differs from bundle")
        if metadata["checkpoint_sha256"] != checkpoint_digest:
            raise ValueError(f"{split} artifact checkpoint differs from bundle")
    return checkpoint_digest


def _validate_bundle_manifest(manifest: Any) -> dict[str, Any]:
    """
    Validate the immutable pointer to one bundle generation.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("Bundle manifest must be an object")
    required = {
        "schema_version",
        "generation",
        "provenance",
        "selected_checkpoint",
        "resolution",
        "parameter_count",
        "artifacts",
    }
    if set(manifest) != required or manifest["schema_version"] != BUNDLE_SCHEMA_VERSION:
        raise ValueError("Unsupported bundle manifest schema")
    generation = manifest["generation"]
    if (
        not isinstance(generation, str)
        or len(generation) != 32
        or any(char not in "0123456789abcdef" for char in generation)
    ):
        raise ValueError("Bundle manifest requires a hexadecimal generation ID")
    provenance = _validate_run_protocol(manifest["provenance"])
    checkpoint = manifest["selected_checkpoint"]
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != {"path", "sha256"}
        or not isinstance(checkpoint["path"], str)
        or not _valid_sha256(checkpoint["sha256"])
    ):
        raise ValueError("Bundle manifest requires selected checkpoint provenance")
    if type(manifest["resolution"]) is not int or manifest["resolution"] <= 0:
        raise ValueError("Bundle manifest requires a positive resolution")
    if type(manifest["parameter_count"]) is not int or manifest["parameter_count"] <= 0:
        raise ValueError("Bundle manifest requires a positive parameter count")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "val_predictions",
        "test_predictions",
        "training",
    }:
        raise ValueError("Bundle manifest requires both splits and training metadata")
    for name, entry in artifacts.items():
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path", "sha256"}
            or not isinstance(entry["path"], str)
            or not _valid_sha256(entry["sha256"])
        ):
            raise ValueError(f"Bundle manifest has an invalid {name} artifact")
        relative = Path(entry["path"])
        expected = Path("generations") / generation / f"{name}.json"
        if relative.is_absolute() or relative != expected:
            raise ValueError("Bundle manifest artifact path escapes its generation")
    return _clone(manifest)


def publish_bundle(
    bundle_dir: Path | str,
    *,
    run_dir: Path | str,
    val_predictions: Path | str,
    test_predictions: Path | str,
    selected_checkpoint: Path | str,
    resolution: int,
    parameter_count: int,
    failpoint: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Publish a complete immutable generation by atomically replacing its manifest.
    """
    protocol = read_run_protocol(run_dir)
    training = read_training_record(run_dir)
    val_path = Path(val_predictions)
    test_path = Path(test_predictions)
    checkpoint = Path(selected_checkpoint)
    val_artifact = read_prediction_artifact(val_path, allow_smoke=True)
    test_artifact = read_prediction_artifact(test_path, allow_smoke=True)
    checkpoint_digest = _validate_bundle_inputs(
        protocol,
        training,
        val_artifact,
        test_artifact,
        checkpoint,
        resolution,
        parameter_count,
    )
    root = Path(bundle_dir)
    root.mkdir(parents=True, exist_ok=True)
    generations = root / "generations"
    generations.mkdir(exist_ok=True)
    _fsync_directory(root)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    try:
        staged_artifacts = {
            "val_predictions": val_path,
            "test_predictions": test_path,
        }
        for name, source in staged_artifacts.items():
            _copy_fsynced(source, staging / f"{name}.json")
        _write_fsynced(staging / "training.json", training)
        copied_digest = _validate_bundle_inputs(
            protocol,
            _validate_training_document(read_json(staging / "training.json"), protocol),
            read_prediction_artifact(staging / "val_predictions.json", allow_smoke=True),
            read_prediction_artifact(staging / "test_predictions.json", allow_smoke=True),
            checkpoint,
            resolution,
            parameter_count,
        )
        if copied_digest != checkpoint_digest:
            raise ValueError("Selected checkpoint changed during bundle publication")
        _fsync_directory(staging)
        _invoke_failpoint(failpoint, "after_staging")
        generation = uuid.uuid4().hex
        destination = generations / generation
        if destination.exists():
            raise FileExistsError(destination)
        os.rename(staging, destination)
        _fsync_directory(generations)
        _invoke_failpoint(failpoint, "after_generation")
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "generation": generation,
            "provenance": protocol,
            "selected_checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_digest,
            },
            "resolution": resolution,
            "parameter_count": parameter_count,
            "artifacts": {
                name: {
                    "path": str(Path("generations") / generation / f"{name}.json"),
                    "sha256": file_sha256(destination / f"{name}.json"),
                }
                for name in ("val_predictions", "test_predictions", "training")
            },
        }
        _validate_bundle_manifest(manifest)
        _invoke_failpoint(failpoint, "before_manifest_replace")
        _replace_fsynced(root / BUNDLE_MANIFEST_FILE_NAME, manifest)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def read_bundle(
    bundle_dir: Path | str, *, allow_smoke: bool = False
) -> dict[str, Any]:
    """
    Snapshot one manifest and validate its immutable generation before returning it.
    """
    root = Path(bundle_dir)
    manifest = _validate_bundle_manifest(read_json(root / BUNDLE_MANIFEST_FILE_NAME))
    documents: dict[str, dict[str, Any]] = {}
    for name, entry in manifest["artifacts"].items():
        path = root / entry["path"]
        if not path.is_file() or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Bundle artifact hash mismatch: {name}")
        documents[name] = read_json(path)
    provenance = manifest["provenance"]
    val = read_prediction_artifact(
        root / manifest["artifacts"]["val_predictions"]["path"],
        allow_smoke=allow_smoke,
    )
    test = read_prediction_artifact(
        root / manifest["artifacts"]["test_predictions"]["path"],
        allow_smoke=allow_smoke,
    )
    training = _validate_training_document(
        documents["training"], provenance
    )
    _require_completed_training(training)
    for split, artifact in (("val", val), ("test", test)):
        metadata = artifact["metadata"]
        if (
            metadata["split"] != split
            or metadata["model"] != provenance["model"]
            or metadata["smoke_run"] != provenance["smoke_run"]
            or metadata["resolution"] != manifest["resolution"]
            or metadata["checkpoint_sha256"]
            != manifest["selected_checkpoint"]["sha256"]
        ):
            raise ValueError("Bundle artifact provenance differs from its manifest")
    if provenance["smoke_run"] and not allow_smoke:
        raise ValueError("Smoke bundles cannot enter a full comparison")
    return {
        "manifest": manifest,
        "val_predictions": val,
        "test_predictions": test,
        "training": training,
    }