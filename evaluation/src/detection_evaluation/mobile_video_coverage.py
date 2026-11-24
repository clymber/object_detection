"""
Reconcile saved video source coverage and recalculate numerical timing summaries.
"""

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math


def source_reference(run):
    """
    Read matching annotation-derived or catalog source timestamps.
    """
    source = run["input"]
    if "annotation_reference" in source:
        reference = source["annotation_reference"]
        accuracy = run.get("accuracy") or {}
        if not isinstance(reference, dict) or not isinstance(accuracy, dict):
            return None
        count = reference.get("frame_count")
        annotation_hash = reference.get("annotation_sha256")
        if (reference.get("schema_version") != 1
                or not source.get("sha256")
                or source["sha256"] != reference.get("video_sha256")
                or source["sha256"] != accuracy.get("video_sha256")
                or not isinstance(annotation_hash, str) or len(annotation_hash) != 64
                or any(value not in "0123456789abcdef" for value in annotation_hash)
                or annotation_hash != accuracy.get("annotation_sha256")
                or reference.get("annotation_file") != accuracy.get("annotation_file")
                or type(count) is not int or count <= 0
                or not isinstance(reference.get("time_base"), str)):
            return None
        try:
            time_base = Fraction(reference["time_base"])
        except (ValueError, ZeroDivisionError):
            return None
        return ([index * time_base for index in range(count)]
                if time_base > 0 else None)
    manifest = source.get("manifest") or {}
    entry = manifest.get("entry") or {}
    sequence = entry.get("frame_reference") or {}
    pts, base = sequence.get("pts"), sequence.get("time_base")
    video = entry.get("video") or {}
    if (not source.get("sha256") or source["sha256"] != entry.get("sha256")
            or source.get("source_id") != entry.get("id")
            or not isinstance(pts, list) or not pts
            or any(type(value) is not int for value in pts)
            or any(a >= b for a, b in zip(pts, pts[1:]))
            or len(pts) != video.get("frame_count") or not isinstance(base, str)):
        return None
    try:
        time_base = Fraction(base)
    except (ValueError, ZeroDivisionError):
        return None
    digest = hashlib.sha256(json.dumps(
        {"time_base": base, "pts": pts}, separators=(",", ":")
    ).encode()).hexdigest()
    if digest != video.get("pts_sha256") or time_base <= 0:
        return None
    return [value * time_base for value in pts]


def reconcile_coverage(run, frames):
    """
    Rebuild identity sets and input balances independently of the native summary.
    """
    counts, selection = run["counts"], run["selection"]
    statuses = Counter(frame["status"] for frame in frames)
    completed, failed = statuses["completed"], statuses["failed"]
    skipped = statuses["skipped_superseded"] + statuses["skipped_stale"]
    unfinished = statuses["unfinished"] + statuses["cancelled"]
    source = run.get("source") or {}
    decode_events = 0
    if run["mode"] == "offline":
        decoded = counts.get("decoded", 0)
        indices = selection.get("frame_indices")
        input_count = (sum(index < decoded for index in indices)
                       if indices is not None else decoded)
        input_count += sum(frame["pts"] is None
                           and frame.get("failure_stage") == "decoding" for frame in frames)
        counters_match = all(counts.get(key) == value for key, value in (
            ("completed", completed), ("failed", failed), ("unfinished", unfinished)))
        indices_in_order = [frame["source_frame_index"] for frame in frames]
        counters_match &= indices_in_order == sorted(indices_in_order) and all(
            frame["pts"] is None or frame["source_frame_index"] < decoded for frame in frames)
        decode_events = int(bool(run.get("decode_loss")) or (
            (run.get("issue") or {}).get("stage") == "decoding" and run["status"] == "failed"))
        decoded_not_delivered = 0
    else:
        input_count = counts.get("input", 0)
        counters_match = all(counts.get(key) == value for key, value in (
            ("completed", completed), ("failed", failed),
            ("cancelled", statuses["cancelled"]),
            ("skipped_superseded", statuses["skipped_superseded"]),
            ("skipped_stale", statuses["skipped_stale"])))
        unfinished += counts.get("active", 0) + counts.get("pending", 0)
        counters_match &= (counts.get("unfinished", 0) == unfinished
                           and source.get("delivered_frames", 0) == input_count
                           and source.get("decoded_frames", 0) >= input_count)
        decoded_not_delivered = max(0, source.get("decoded_frames", 0)
                                   - source.get("delivered_frames", 0))
        decode_events = int(bool((source.get("issue") or {}).get("decode_failure")))
    balanced = (counters_match and input_count == len(frames)
                and input_count == completed + failed + skipped + unfinished)
    sequence = source_reference(run)
    if selection.get("requested_track_id") is not None:
        sequence = None
    loops = selection.get("loop_count")
    first_loop = selection.get("first_loop_index", 0)
    expected = None
    if sequence is not None and type(loops) is int and 0 < loops <= 10000:
        indices = selection.get("frame_indices")
        if indices is None:
            indices = range(len(sequence))
        expected = {f"{loop}:{index}" for loop in range(first_loop, first_loop + loops)
                    for index in indices}
    identities = Counter(f'{frame["loop_index"]}:{frame["source_frame_index"]}'
                         for frame in frames)
    observed = set(identities)
    mismatches = []
    for frame in frames:
        pts, index = frame["pts"], frame["source_frame_index"]
        if pts is None or (sequence is not None and (
            index >= len(sequence) or pts["epoch"] != 0
            or Fraction(pts["value"], pts["timescale"]) != sequence[index]
        )):
            mismatches.append(f'{frame["loop_index"]}:{index}')
    missing = sorted(expected - observed) if expected is not None else []
    unexpected = sorted(observed - expected) if expected is not None else []
    duplicates = sorted(key for key, value in identities.items() if value > 1)
    complete = (expected is not None and balanced and not missing and not unexpected
                and not duplicates and not mismatches and not decode_events
                and not decoded_not_delivered and not unfinished)
    return {
        "status": "complete" if complete else "incomplete",
        "input_frames": input_count, "completed": completed, "failed": failed,
        "skipped": skipped, "unfinished": unfinished, "observed_outcomes": len(frames),
        "expected_frames": len(expected) if expected is not None else None,
        "missing_identities": missing, "duplicate_identities": duplicates,
        "unexpected_identities": unexpected, "timestamp_mismatches": sorted(set(mismatches)),
        "decoded_not_delivered": decoded_not_delivered, "decode_loss_events": decode_events,
        "accounting_balanced": balanced,
    }


def _statistics(samples):
    """
    Apply the same nearest-rank protocol to independently recovered raw samples.
    """
    ordered = sorted(samples)
    count = len(ordered)
    total = sum(ordered)
    return {"sample_count": count, "total": total,
            "mean": total / count if count else None,
            "p50": ordered[math.ceil(0.5 * count) - 1] if count else None,
            "p95": ordered[math.ceil(0.95 * count) - 1] if count else None}


def _rate(count, duration):
    """
    Leave zero-width windows unavailable rather than inventing a throughput.
    """
    return count * 1000 / duration if duration > 0 else None


def recalculate_metrics(run, frames):
    """
    Recover latency, event rates and result ages directly from retained endpoints.
    """
    timings = [row["timing_ms"] for row in frames]
    successful = [row["timing_ms"] for row in frames if row["status"] == "completed"]
    saved = run["metrics"]
    if run["mode"] in {"paced", "camera"}:
        if saved.get("window_start_ms") is None:
            return {}
        start, end = saved["window_start_ms"], saved["window_end_ms"]
        duration = end - start
        complete = [t for t in successful if start <= t["done"] < end]
        published = [row["timing_ms"] for row in frames
                     if row["status"] == "completed" and row["publication_status"] == "published"]
        received = [t for t in published if start <= t["published"] < end]
        fresh = sum(0 <= t["published"] - t["due"] <= 200
                    and t["published"] >= t["done"] for t in received)
        inputs = sum(t["delivered"] is not None and start <= t["delivered"] < end
                     for t in timings)
        value = {"input_frames": inputs, "completed_frames": len(complete),
                 "fresh_results": fresh, "input_fps": _rate(inputs, duration),
                 "completed_fps": _rate(len(complete), duration),
                 "fresh_fps": _rate(fresh, duration) if saved.get("fresh_fps") is not None else None,
                 "post_window_completions": sum(t["done"] >= end for t in successful),
                 "post_window_publications": sum(t["published"] >= end for t in published),
                 "invalid_publications": len(received) - fresh,
                 "completed_age_ms": _statistics([t["done"] - t["due"] for t in complete]),
                 "published_age_ms": _statistics([t["published"] - t["due"] for t in received])}
        successful = complete
    else:
        window = run.get("host_window_ms")
        if not window or window.get("ended") is None:
            return {}
        duration = window["ended"] - window["started"]
        ready = [t["ready"] for t in timings if t["ready"] is not None]
        done = [t["done"] for t in timings if t["done"] is not None]
        models = [t["model_completed"] - t["model_submitted"] for t in successful
                  if t["model_completed"] is not None]
        value = {"input_fps": _rate(run["counts"]["decoded"], duration),
                 "completed_fps": _rate(len(successful), duration),
                 "model_fps": _rate(len(models), sum(models)),
                 "end_to_end_fps": _rate(len(successful), max(done) - min(ready))
                 if ready and done else None}
    value["model_latency_ms"] = _statistics([
        t["model_completed"] - t["model_submitted"] for t in successful
        if t["model_completed"] is not None])
    value["frame_latency_ms"] = _statistics([t["done"] - t["ready"] for t in successful])
    return value


def validate_recalculated(run, frames):
    """
    Reject a false completeness claim or summaries that disagree with the raw files.
    """
    coverage = reconcile_coverage(run, frames)
    declared = run.get("coverage")
    if declared is not None:
        for key, value in coverage.items():
            if key == "status":
                if declared.get(key) == "complete" and value != "complete":
                    raise ValueError("False complete source coverage claim")
            elif declared.get(key) != value:
                raise ValueError(f"Coverage summary mismatch: {key}")
        if run["validity"].get("coverage_status") != declared.get("status"):
            raise ValueError("Coverage validity differs from reconciliation")
    metrics = recalculate_metrics(run, frames)
    for key, value in metrics.items():
        expected = run["metrics"].get(key)
        pairs = value.items() if isinstance(value, dict) else [(key, value)]
        for field, actual in pairs:
            saved = expected.get(field) if isinstance(value, dict) and isinstance(expected, dict) else expected
            valid = saved is None if actual is None else (
                type(saved) in (int, float) and math.isclose(saved, actual, rel_tol=1e-9, abs_tol=1e-9))
            if not valid:
                raise ValueError(f"Metric summary mismatch: {key}.{field}")
    return coverage, metrics
