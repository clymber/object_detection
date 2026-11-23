"""
Check capture-drop accounting, actual timestamp mapping and camera summary integrity.
"""

from copy import deepcopy
import hashlib
import json

import pytest

from detection_evaluation.mobile_camera import validate_camera


def timestamp(value, scale=1000, epoch=0):
    """
    Supply exact Core Media integer fields without normalizing the source PTS.
    """
    return {"value": value, "timescale": scale, "flags": 1, "epoch": epoch}


def statistics(value):
    """
    Specify a one-sample summary independently of the production accumulator.
    """
    return dict.fromkeys(("total", "mean", "p50", "p95"), value) | {"sample_count": 1}


def fixture():
    """
    Describe successful inference, one camera-system drop and one stale queue skip.
    """
    manifest = {"source_id": "camera:test", "source_type": "camera"}
    text = json.dumps(manifest)
    source = {"source_id": "camera:test", "manifest": manifest, "manifest_json": text,
              "manifest_sha256": hashlib.sha256(text.encode()).hexdigest()}
    frames = []
    for index, due in enumerate((1010, 1015, 1020)):
        frames.append({"source_frame_index": index, "loop_index": 0,
                       "capture_kind": "dropped" if index == 1 else "delivered",
                       "capture_admitted": index != 1, "capture_lost_frames": 1 if index == 1 else None,
                       "record_type": "capture" if index == 1 else "inference",
                       "status": ("completed", "capture_dropped", "skipped_stale")[index],
                       "publication_status": "published" if index == 0 else "not_applicable",
                       "capture_host_pts": timestamp(10000 + due - 1000),
                       "capture_received_ms": 1300 if index == 2 else due + 5,
                       "timing_ms": {"due": due, "ready": due + 5, "model_submitted": None,
                                     "model_completed": None, "done": None, "published": None,
                                     "delivered": None}})
    frames[0]["timing_ms"].update(delivered=1016, model_submitted=1017,
                                 model_completed=1037, done=1040, published=1050)
    frames[2]["timing_ms"].update(ready=1300, delivered=1301)
    run = {"mode": "camera", "input": source,
           "capture": {"timestamp_mapping": {"host_anchor": timestamp(10, scale=1),
                                             "inference_anchor_ms": 1000},
                       "observed_callbacks": 3, "drop_events": 1, "delivered_callbacks": 2,
                       "started_ms": 1000, "ended_ms": 2000, "actual_delivered_fps": 2},
           "counts": {"input": 2, "completed": 1, "failed": 0, "unfinished": 0,
                      "cancelled": 0, "skipped_stale": 1, "skipped_superseded": 0,
                      "capture_callbacks": 3, "capture_delivered": 2, "capture_drop_events": 1},
           "coverage": {"scope": "observed_capture_callbacks", "status": "complete",
                        "sensor_coverage_status": "unverified", "expected_sensor_frames": None,
                        "accounting_balanced": True},
           "validity": {"coverage_status": "complete"},
           "metrics": {"window_start_ms": 1000, "window_end_ms": 2000,
                       "input_frames": 2, "completed_frames": 1, "fresh_results": 1,
                       "input_fps": 2, "completed_fps": 1, "fresh_fps": 1,
                       "post_window_completions": 0, "post_window_publications": 0,
                       "invalid_publications": 0, "completed_age_ms": statistics(30),
                       "published_age_ms": statistics(40), "model_latency_ms": statistics(20),
                       "frame_latency_ms": statistics(25)}}
    return run, frames


def test_camera_drops_and_queue_skips_are_distinct():
    """
    Fully account for observed callbacks while leaving unknown sensor coverage explicit.
    """
    run, frames = fixture()
    coverage, metrics = validate_camera(run, frames)
    assert coverage["status"] == "complete"
    assert metrics["input_frames"] == 2
    assert metrics["completed_age_ms"]["p95"] == 30


def test_unknown_capture_discontinuity_is_incomplete():
    """
    A reported discontinuity does not identify how many physical frames were lost.
    """
    run, frames = fixture()
    frames[1]["capture_lost_frames"] = None
    with pytest.raises(ValueError, match="False complete"):
        validate_camera(run, frames)
    run["coverage"]["status"] = run["validity"]["coverage_status"] = "incomplete"
    assert validate_camera(run, frames)[0]["status"] == "incomplete"


@pytest.mark.parametrize("mutation", ["clock", "drop", "count", "sensor", "metric", "manifest", "fps"])
def test_invalid_camera_summaries_are_rejected(mutation):
    """
    Reject callback-time substitution, misclassified drops and altered returned settings.
    """
    run, frames = deepcopy(fixture())
    if mutation == "clock":
        frames[0]["timing_ms"]["due"] = frames[0]["capture_received_ms"]
    elif mutation == "drop":
        frames[1]["capture_admitted"] = True
    elif mutation == "count":
        run["counts"]["input"] = 3
    elif mutation == "sensor":
        run["coverage"]["sensor_coverage_status"] = "complete"
    elif mutation == "metric":
        run["metrics"]["frame_latency_ms"]["mean"] = 1
    elif mutation == "manifest":
        run["input"]["manifest_json"] += "changed"
    else:
        run["capture"]["actual_delivered_fps"] = 30
    with pytest.raises(ValueError):
        validate_camera(run, frames)
