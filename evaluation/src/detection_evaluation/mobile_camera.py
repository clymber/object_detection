"""
Validate live-camera callback accounting and clock mapping without AVFoundation.
"""

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math

from .mobile_video_coverage import recalculate_metrics


def _time(value):
    """
    Read an exact numeric Core Media timestamp, retaining its epoch separately.
    """
    if (not isinstance(value, dict) or value.get("timescale", 0) <= 0
            or not value.get("flags", 0) & 1):
        raise ValueError("Invalid camera host timestamp")
    return Fraction(value["value"], value["timescale"]), value["epoch"]


def validate_camera(run, frames):
    """
    Reconcile observed callbacks without pretending to know every sensor frame.
    """
    capture, declared = run["capture"], run["coverage"]
    source = run["input"]
    text = source["manifest_json"]
    if (hashlib.sha256(text.encode()).hexdigest() != source["manifest_sha256"]
            or json.loads(text) != source["manifest"]
            or source["manifest"]["source_id"] != source["source_id"]):
        raise ValueError("Camera input manifest mismatch")
    by_index = {}
    inference = []
    duplicate = False
    unknown = False
    for frame in frames:
        index = frame["source_frame_index"]
        duplicate |= index in by_index
        by_index[index] = frame
        if frame["loop_index"] != 0 or frame["capture_kind"] not in {"delivered", "dropped"}:
            raise ValueError("Invalid camera callback identity/kind")
        if frame["record_type"] == "inference":
            if not frame["capture_admitted"] or frame["capture_kind"] != "delivered":
                raise ValueError("Inference has no admitted capture callback")
            inference.append(frame)
        elif frame["record_type"] != "capture":
            raise ValueError("Invalid camera record type")
        if frame["capture_kind"] == "dropped":
            if frame["status"] != "capture_dropped" or frame["capture_admitted"]:
                raise ValueError("Camera drop was counted as inference")
            unknown |= frame.get("capture_lost_frames") is None
        unknown |= (frame["timing_ms"]["due"] is None
                    or (frame["record_type"] == "capture" and frame.get("failure") is not None
                        and not frame.get("capture_cancelled", False)))
        mapping, host = capture["timestamp_mapping"], frame["capture_host_pts"]
        if host is not None and mapping is not None:
            anchor, epoch = _time(mapping["host_anchor"])
            actual, host_epoch = _time(host)
            expected = mapping["inference_anchor_ms"] + float(actual - anchor) * 1000
            if (host_epoch != epoch or frame["timing_ms"]["due"] is None
                    or not math.isclose(frame["timing_ms"]["due"], expected, abs_tol=1e-6)):
                raise ValueError("Capture timestamp mapping differs from recorded calibration")
    events = list(by_index.values())
    admitted = {row["source_frame_index"] for row in events if row["capture_admitted"]}
    processed = {row["source_frame_index"] for row in inference}
    statuses = Counter(row["status"] for row in inference)
    c = run["counts"]
    unfinished = statuses["cancelled"] + statuses["unfinished"]
    balanced = (not duplicate and admitted == processed
                and sorted(by_index) == list(range(len(events)))
                and c["input"] == len(admitted)
                and c["input"] == statuses["completed"] + statuses["failed"]
                + statuses["skipped_stale"] + statuses["skipped_superseded"] + unfinished)
    delivered = sum(row["capture_kind"] == "delivered" for row in events)
    drops = sum(row["capture_kind"] == "dropped" for row in events)
    for key, actual in {"input": len(admitted), "completed": statuses["completed"],
                        "failed": statuses["failed"], "unfinished": unfinished,
                        "cancelled": statuses["cancelled"],
                        "skipped_stale": statuses["skipped_stale"],
                        "skipped_superseded": statuses["skipped_superseded"],
                        "capture_callbacks": len(events), "capture_delivered": delivered,
                        "capture_drop_events": drops}.items():
        if c[key] != actual:
            raise ValueError(f"Camera count mismatch: {key}")
    if (capture["observed_callbacks"] != len(events) or capture["drop_events"] != drops
            or capture["delivered_callbacks"] != delivered
            or declared["accounting_balanced"] != balanced
            or declared["scope"] != "observed_capture_callbacks"
            or declared["sensor_coverage_status"] != "unverified"
            or declared["expected_sensor_frames"] is not None):
        raise ValueError("Camera coverage summary mismatch")
    source_unfinished = any(row["capture_kind"] == "delivered" and not row["capture_admitted"]
                            for row in events)
    complete = (balanced and not unknown and unfinished == 0 and not source_unfinished
                and bool(events) and capture["timestamp_mapping"] is not None)
    if declared["status"] == "complete" and not complete:
        raise ValueError("False complete camera coverage claim")
    if run["validity"]["coverage_status"] != declared["status"]:
        raise ValueError("Camera validity differs from coverage")
    metrics = recalculate_metrics(run, frames)
    for key, expected in metrics.items():
        saved = run["metrics"].get(key)
        pairs = expected.items() if isinstance(expected, dict) else [(key, expected)]
        for field, value in pairs:
            actual = saved.get(field) if isinstance(expected, dict) and isinstance(saved, dict) else saved
            if value is None:
                valid = actual is None
            else:
                valid = type(actual) in (int, float) and math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-9)
            if not valid:
                raise ValueError(f"Camera metric mismatch: {key}.{field}")
    start, end = capture["started_ms"], capture["ended_ms"]
    fps = None
    if start is not None and end is not None and end > start:
        fps = sum(row["capture_kind"] == "delivered" and start <= row["capture_received_ms"] < end
                  for row in events) * 1000 / (end - start)
    saved_fps = capture["actual_delivered_fps"]
    if (fps is None and saved_fps is not None) or (fps is not None and (
            saved_fps is None or not math.isclose(fps, saved_fps, rel_tol=1e-9, abs_tol=1e-9))):
        raise ValueError("Actual capture cadence differs from callback times")
    return {"status": declared["status"], "accounting_balanced": balanced}, metrics
