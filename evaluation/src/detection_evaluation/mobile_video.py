"""
Read native schema-v1 video exports without importing a detector framework.

Integrity and record consistency do not establish source coverage or model parity.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .mobile_video_coverage import validate_recalculated
from .mobile_camera import validate_camera


TIMING_NAMES = (
    "ready", "due", "delivered", "decided", "model_submitted", "model_completed",
    "done", "recorded", "publication_checked", "published", "model_latency",
    "frame_latency", "completed_age", "published_age",
)
CSV_COLUMNS = (
    "run_id", "source_id", "loop_index", "source_frame_index", "pts_value",
    "pts_timescale", "pts_flags", "pts_epoch", "status", "publication_status",
) + tuple(f"{name}_ms" for name in TIMING_NAMES)
STATUSES = {
    "completed", "failed", "unfinished", "cancelled", "skipped_superseded",
    "skipped_stale",
    "capture_dropped", "capture_failed",
}
PUBLICATION_STATUSES = {
    "not_applicable", "not_requested", "not_recorded", "pending", "published",
    "late", "superseded", "cancelled", "failed", "not_received",
}


def _reject_constant(value):
    """
    Reject nonstandard JSON constants before they can enter numerical comparisons.
    """
    raise ValueError(f"Non-finite JSON constant: {value}")


def _unique_object(pairs):
    """
    Reject ambiguous duplicate keys at every level of the handoff.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text):
    """
    Read a strict JSON object while preserving arbitrary-size PTS integers.
    """
    value = json.loads(text, parse_constant=_reject_constant,
                       object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _number(value):
    """
    Accept finite numbers while excluding boolean values.
    """
    return type(value) in (int, float) and math.isfinite(value)


def _validate_frame(row, run):
    """
    Validate frame identity, exact PTS, detections and nullable timing endpoints.
    """
    if row.get("schema_version") != 1 or row.get("run_id") != run["run_id"]:
        raise ValueError("Frame schema/run identity mismatch")
    if row.get("source_id") != run["input"]["source_id"]:
        raise ValueError("Frame source identity mismatch")
    for name in ("loop_index", "source_frame_index"):
        if type(row.get(name)) is not int or row[name] < 0:
            raise ValueError(f"Invalid {name}")
    if row.get("status") not in STATUSES:
        raise ValueError("Invalid frame status")
    if row["status"].startswith("capture_") and run["mode"] != "camera":
        raise ValueError("Capture-only status in a video run")
    if row.get("publication_status") not in PUBLICATION_STATUSES:
        raise ValueError("Invalid publication status")
    pts = row["pts"]
    if pts is not None:
        if set(pts) != {"value", "timescale", "flags", "epoch"} or any(
            type(value) is not int for value in pts.values()
        ):
            raise ValueError("PTS fields must be exact integers")
        if pts["timescale"] <= 0 or not pts["flags"] & 1:
            raise ValueError("Decoded frame PTS must be valid")
    elif not row.get("pts_unavailable_reason"):
        raise ValueError("Missing PTS needs an unavailable reason")
    detections = row["detections"]
    if detections is None:
        if row["status"] == "completed" or not row.get("detections_unavailable_reason"):
            raise ValueError("Successful frames need a detection array")
    elif not isinstance(detections, list) or len(detections) > 100:
        raise ValueError("Invalid detections array")
    else:
        previous = 1.0
        for detection in detections:
            score, box = detection["score"], detection["bbox"]
            if (type(detection["category_id"]) is not int
                    or detection["category_id"] < 0 or not _number(score)
                    or not 0.25 <= score <= previous or len(box) != 4
                    or not all(_number(value) for value in box)
                    or box[2] <= 0 or box[3] <= 0):
                raise ValueError("Invalid retained source-pixel detection")
            previous = score
    times = row["timing_ms"]
    if set(times) != set(TIMING_NAMES):
        raise ValueError("Missing or unknown timing columns")
    for name, value in times.items():
        if value is not None and not _number(value):
            raise ValueError(f"Invalid timing endpoint: {name}")
    ordered = [times[key] for key in (
        "ready", "model_submitted", "model_completed", "done", "published"
    ) if times[key] is not None]
    if ordered != sorted(ordered) or any(value < 0 for value in ordered):
        raise ValueError("Timing endpoints are out of order")
    if row["publication_status"] == "published" and times["published"] is None:
        raise ValueError("Published result needs an actual receipt timestamp")


def _validate_csv(rows, frames):
    """
    Compare CSV identity and raw numbers with JSON, leaving absent timings blank.
    """
    if len(rows) != len(frames):
        raise ValueError("JSONL/CSV record counts differ")
    for csv_row, frame in zip(rows, frames):
        expected = {name: frame[name] for name in CSV_COLUMNS[:4]}
        expected.update({name: frame[name] for name in ("status", "publication_status")})
        expected.update({f"pts_{name}": (frame["pts"] or {}).get(name)
                         for name in ("value", "timescale", "flags", "epoch")})
        expected.update({f"{name}_ms": frame["timing_ms"][name] for name in TIMING_NAMES})
        for name, value in expected.items():
            actual = csv_row[name]
            if value is None:
                valid = actual == ""
            elif name.endswith("_ms"):
                valid = math.isclose(float(actual), value, rel_tol=1e-12, abs_tol=1e-9)
            elif type(value) is int:
                valid = int(actual) == value
            elif type(value) is float:
                valid = float(actual) == value
            else:
                valid = actual == value
            if not valid:
                raise ValueError(f"JSONL/CSV mismatch in {name}")


def validate_mobile_video_export(directory: Path) -> dict:
    """
    Verify three returned files without upgrading their declared benchmark validity.
    """
    directory = Path(directory)
    run = _json((directory / "mobile_run.json").read_text())
    if (run.get("schema_version") != 1 or run.get("scope") not in {"video_diagnostic", "camera_diagnostic"}
            or run.get("mode") not in {"offline", "paced", "camera"}
            or run.get("status") not in {"completed", "failed", "cancelled"}):
        raise ValueError("Unsupported video run schema, scope, mode or status")
    for name in ("runtime", "settings", "selection", "counts", "metrics", "validity",
                 "input", "model_provenance", "timing", "files"):
        if not isinstance(run.get(name), dict):
            raise ValueError(f"Missing run metadata: {name}")
    files = {}
    for name in ("frames.jsonl", "timings.csv"):
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != run["files"][name]["sha256"]:
            raise ValueError(f"File hash mismatch: {name}")
        files[name] = data.decode("utf-8")
    frames = [_json(line) for line in files["frames.jsonl"].splitlines()]
    for row in frames:
        _validate_frame(row, run)
    with (directory / "timings.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_COLUMNS):
            raise ValueError("CSV columns differ from schema v1")
        _validate_csv(list(reader), frames)
    if any(run["files"][name]["records"] != len(frames) for name in files):
        raise ValueError("File record count mismatch")
    if run["frame_records"] != len(frames):
        raise ValueError("Run record count mismatch")
    coverage, metrics = (validate_camera(run, frames) if run["mode"] == "camera"
                         else validate_recalculated(run, frames))
    return {"run_id": run["run_id"], "mode": run["mode"], "records": len(frames),
            "integrity": "passed", "validity": run["validity"],
            "coverage_status": coverage["status"],
            "accounting_balanced": coverage["accounting_balanced"],
            "metrics_recalculated": bool(metrics)}


def main():
    """
    Validate one or more downloaded run folders in the evaluation environment.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--original", type=Path,
                        help="Compare a saved copy with an original three-file export")
    args = parser.parse_args()
    if args.original is not None and len(args.directories) != 1:
        parser.error("--original requires exactly one saved directory")
    for directory in args.directories:
        report = (validate_transfer(args.original, directory) if args.original is not None
                  else validate_mobile_video_export(directory))
        print(json.dumps(report, sort_keys=True))


def validate_transfer(original: Path, saved: Path) -> dict:
    """
    Validate both packages and require exact transfer of all metadata and raw results.
    """
    validate_mobile_video_export(original)
    result = validate_mobile_video_export(saved)
    for name in ("frames.jsonl", "timings.csv", "mobile_run.json"):
        if (Path(original) / name).read_bytes() != (Path(saved) / name).read_bytes():
            raise ValueError(f"Saved file differs from original: {name}")
    return result | {"transfer": "passed"}


if __name__ == "__main__":
    main()
