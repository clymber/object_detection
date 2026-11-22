"""
Check framework-neutral reading and corruption rejection for mobile video exports.
"""

import csv
import hashlib
import json

import pytest

from detection_evaluation.mobile_video import (
    CSV_COLUMNS,
    TIMING_NAMES,
    validate_mobile_video_export,
)


def write_export(folder, *, empty=False):
    """
    Create a small returned package with CSV quoting and exact large PTS integers.
    """
    frame = {
        "schema_version": 1, "run_id": "run", "source_id": 'clip,"quoted"\nline',
        "loop_index": 2, "source_frame_index": 3,
        "pts": {"value": 9007199254740993, "timescale": 30000, "epoch": 2, "flags": 3},
        "status": "completed", "publication_status": "late", "detections": [],
        "timing_ms": dict.fromkeys(TIMING_NAMES),
    }
    frame["timing_ms"].update(ready=1000, model_submitted=1010,
                             model_completed=1040, done=1060,
                             model_latency=30, frame_latency=60)
    run = {
        "schema_version": 1, "scope": "video_diagnostic", "run_id": "run",
        "mode": "paced", "status": "cancelled", "frame_records": int(not empty),
        "runtime": {}, "settings": {}, "selection": {}, "counts": {},
        "metrics": {}, "input": {"source_id": frame["source_id"]},
        "model_provenance": {}, "timing": {}, "files": {},
        "validity": {"coverage_status": "unverified", "benchmark_status": "not_validated"},
    }
    (folder / "frames.jsonl").write_text("" if empty else json.dumps(frame) + "\n")
    with (folder / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        if not empty:
            row = {key: frame[key] for key in CSV_COLUMNS[:4]}
            row.update(status="completed", publication_status="late")
            row.update({f"pts_{key}": value for key, value in frame["pts"].items()})
            row.update({f"{key}_ms": value for key, value in frame["timing_ms"].items()})
            row["ready_ms"] = "1000.0"
            writer.writerow(row)
    save_run(folder, run)
    return frame, run


def save_run(folder, run):
    """
    Pin file digests independently of each test's semantic mutations.
    """
    for name in ("frames.jsonl", "timings.csv"):
        run["files"][name] = {
            "sha256": hashlib.sha256((folder / name).read_bytes()).hexdigest(),
            "records": run["frame_records"],
        }
    (folder / "mobile_run.json").write_text(json.dumps(run))


def test_partial_run_and_exact_identity(tmp_path):
    """
    Read escaped identity and integer PTS without promoting diagnostic validity.
    """
    _, run = write_export(tmp_path)
    result = validate_mobile_video_export(tmp_path)
    assert result["records"] == 1
    assert result["integrity"] == "passed"
    assert result["validity"] == run["validity"]


def test_zero_frame_failure_is_readable(tmp_path):
    """
    Retain attempts that failed before the first source frame was admitted.
    """
    write_export(tmp_path, empty=True)
    assert validate_mobile_video_export(tmp_path)["records"] == 0


def test_changed_file_hash_is_rejected(tmp_path):
    """
    Detect corruption even when both transport files would otherwise still parse.
    """
    write_export(tmp_path)
    with (tmp_path / "frames.jsonl").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_mobile_video_export(tmp_path)


@pytest.mark.parametrize("mutation", ["pts", "missing_time", "zero_time", "status"])
def test_csv_mismatch_is_rejected(tmp_path, mutation):
    """
    Reject subtle identity, missing-value and status differences despite valid hashes.
    """
    _, run = write_export(tmp_path)
    with (tmp_path / "timings.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if mutation == "pts":
        rows[0]["pts_value"] = "9007199254740992"
    elif mutation == "missing_time":
        rows[0]["ready_ms"] = ""
    elif mutation == "zero_time":
        rows[0]["published_ms"] = "0"
    else:
        rows[0]["status"] = "failed"
    with (tmp_path / "timings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    save_run(tmp_path, run)
    with pytest.raises(ValueError):
        validate_mobile_video_export(tmp_path)


@pytest.mark.parametrize("mutation", ["nan", "null_success", "bad_clock", "bad_score"])
def test_invalid_frame_is_rejected(tmp_path, mutation):
    """
    Reject nonfinite outputs, absent successes, invalid boxes and reversed endpoints.
    """
    frame, run = write_export(tmp_path)
    if mutation == "nan":
        frame["timing_ms"]["done"] = float("nan")
    elif mutation == "null_success":
        frame["detections"] = None
    elif mutation == "bad_clock":
        frame["timing_ms"]["done"] = 500
    else:
        frame["detections"] = [{"category_id": 1, "score": 0.2, "bbox": [0, 0, 2, 2]}]
    (tmp_path / "frames.jsonl").write_text(json.dumps(frame) + "\n")
    save_run(tmp_path, run)
    with pytest.raises(ValueError):
        validate_mobile_video_export(tmp_path)
