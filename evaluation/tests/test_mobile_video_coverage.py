"""
Exercise source-coverage false passes and independent latency/rate reconstruction.
"""

from copy import deepcopy
import hashlib
import json

import pytest

from detection_evaluation.mobile_video_coverage import (
    recalculate_metrics,
    reconcile_coverage,
    source_reference,
    validate_recalculated,
)


def fixture():
    """
    Pin three irregular source instants independently of the observed decoder prefix.
    """
    sequence = {"time_base": "1/1000", "pts": [0, 33, 67]}
    digest = hashlib.sha256(json.dumps(sequence, separators=(",", ":")).encode()).hexdigest()
    entry = {"id": "source", "sha256": "video-hash", "frame_reference": sequence,
             "video": {"frame_count": 3, "pts_sha256": digest}}
    run = {"mode": "offline", "status": "completed", "input": {
        "source_id": "source", "sha256": "video-hash", "manifest": {"entry": entry}},
        "selection": {"first_loop_index": 0, "loop_count": 1, "frame_indices": None},
        "counts": {"decoded": 3, "completed": 3, "failed": 0, "unfinished": 0},
        "metrics": {}, "validity": {"coverage_status": "complete"},
        "host_window_ms": {"started": 0, "ended": 1000}}
    frames = []
    for index, pts in enumerate(sequence["pts"]):
        ready = index * 100
        frames.append({"loop_index": 0, "source_frame_index": index,
                       "status": "completed", "publication_status": "not_recorded",
                       "pts": {"value": pts * 30, "timescale": 30000, "flags": 1, "epoch": 0},
                       "timing_ms": {"ready": ready, "model_submitted": ready + 10,
                                     "model_completed": ready + 40, "done": ready + 60}})
    return run, frames


def test_complete_and_sparse_source_selection():
    """
    Accept equivalent rational PTS and a completely observed explicit selection.
    """
    run, frames = fixture()
    assert reconcile_coverage(run, frames)["status"] == "complete"
    run["selection"]["frame_indices"] = [0, 2]
    run["counts"]["completed"] = 2
    coverage = reconcile_coverage(run, frames[::2])
    assert coverage["status"] == "complete" and coverage["expected_frames"] == 2


def test_balanced_silent_eof_is_incomplete():
    """
    A clean, balanced decoded prefix does not prove that all source frames arrived.
    """
    run, frames = fixture()
    run["counts"].update(decoded=2, completed=2)
    coverage = reconcile_coverage(run, frames[:2])
    assert coverage["accounting_balanced"]
    assert coverage["status"] == "incomplete"
    assert coverage["missing_identities"] == ["0:2"]


@pytest.mark.parametrize("mutation", ["hash", "pts", "reference_digest", "track"])
def test_untrusted_reference_is_not_complete(mutation):
    """
    Reject stale video bytes, altered sequence records, drift and a different track.
    """
    run, frames = fixture()
    if mutation == "hash":
        run["input"]["sha256"] = "changed"
    elif mutation == "pts":
        frames[1]["pts"]["value"] += 1
    elif mutation == "reference_digest":
        run["input"]["manifest"]["entry"]["frame_reference"]["pts"][1] += 1
    else:
        run["selection"]["requested_track_id"] = 2
    assert reconcile_coverage(run, frames)["status"] == "incomplete"


def test_duplicate_and_failed_or_unfinished_outcomes():
    """
    Keep source accounting separate from numerical success and unfinished inference.
    """
    run, frames = fixture()
    frames[1]["status"] = "failed"
    run["counts"].update(completed=2, failed=1)
    assert reconcile_coverage(run, frames)["status"] == "complete"
    frames[1]["status"] = "unfinished"
    run["counts"].update(failed=0, unfinished=1)
    assert reconcile_coverage(run, frames)["status"] == "incomplete"
    repeated = reconcile_coverage(run, frames + [frames[0]])
    assert repeated["duplicate_identities"] == ["0:0"]
    assert not repeated["accounting_balanced"]


def test_unknown_decoder_loss_does_not_pass():
    """
    An observed decoder error leaves coverage incomplete even if counts balance.
    """
    run, frames = fixture()
    run["decode_loss"] = {"attempted_frame_index": 3, "pts": None, "lost_frame_count": None}
    coverage = reconcile_coverage(run, frames)
    assert coverage["decode_loss_events"] == 1
    assert coverage["status"] == "incomplete"


def test_recalculate_offline_latency_and_rates():
    """
    Recover model/frame statistics and two different rate denominators from raw data.
    """
    run, frames = fixture()
    metrics = recalculate_metrics(run, frames)
    assert metrics["model_latency_ms"]["mean"] == 30
    assert metrics["frame_latency_ms"]["p95"] == 60
    assert metrics["input_fps"] == 3
    assert metrics["completed_fps"] == 3
    assert metrics["model_fps"] == pytest.approx(100 / 3)
    assert metrics["end_to_end_fps"] == pytest.approx(3000 / 260)
    run["metrics"] = deepcopy(metrics)
    run["coverage"] = reconcile_coverage(run, frames)
    validate_recalculated(run, frames)
    run["metrics"]["frame_latency_ms"]["p95"] = 1
    with pytest.raises(ValueError, match="Metric summary"):
        validate_recalculated(run, frames)


def test_false_coverage_claim_is_rejected():
    """
    Prevent a returned metadata file from claiming complete coverage without proof.
    """
    run, frames = fixture()
    run["coverage"] = reconcile_coverage(run, frames)
    run["input"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="False complete"):
        validate_recalculated(run, frames)
