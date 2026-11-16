"""
Verify run selection, export isolation, parity, complete coverage and comparison.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from detection_common.utils.digest import file_digest
from detection_common.utils.json_io import read_json, write_json
from PIL import Image

from detection_evaluation import (
    create_run_protocol,
    finalize_training_attempt,
    start_training_attempt,
)
from detection_evaluation import runtime_comparison as comparison
from detection_evaluation import runtime_workflow as workflow


@pytest.fixture
def fixture_run(tmp_path):
    """
    Build a completed archived Nano run with real pixels and negative images.
    """
    dataset = tmp_path / "dataset"
    canonical_splits = {}
    categories = [{"id": 7, "name": "basketball"}]
    for split in ("train", "val", "test"):
        images = [{"id": i, "file_name": f"{i}.png", "width": 64, "height": 64}
                  for i in (1, 2, 3)]
        annotations = [{"id": 1, "image_id": 1, "category_id": 7,
                        "bbox": [2, 3, 10, 12], "area": 120, "iscrowd": 0}]
        directory = dataset / "images" / split
        directory.mkdir(parents=True)
        for image in images:
            Image.new("RGB", (64, 64), (image["id"], 0, 0)).save(directory / image["file_name"])
        (dataset / "annotations").mkdir(exist_ok=True)
        write_json(dataset / "annotations" / f"instances_{split}.json",
                   {"images": images, "annotations": annotations, "categories": categories})
        canonical_splits[split] = {"annotations": annotations, "images": [
            {"image_id": i["id"], "file_name": i["file_name"], "width": 64, "height": 64,
             "sha256": file_digest(directory / i["file_name"])} for i in images]}
    source = {"category_mapping": categories, "splits": canonical_splits}
    canonical = {"source_fingerprint": workflow.digest_json(source), "canonical_source": source}
    run = tmp_path / "runs/yolox_nano_archived"
    protocol = create_run_protocol(
        run, logical_dataset="basketball", model="yolox_nano",
        source_notebook="nb03.03-yolox_nano_large_basketball.ipynb",
        original_utc=datetime(2026, 11, 5, tzinfo=UTC),
        training_settings={"image_size": 64}, smoke_run=False,
        canonical_dataset_identity=canonical,
        loader_dataset_identity={"source_fingerprint": canonical["source_fingerprint"],
                                 "loader_fingerprint": "f" * 64})
    write_json(run / "dataset_identity.json", protocol["dataset_identity"])
    (run / "args.yaml").write_text("dataset: /original/linux/dataset\n")
    (run / "weights").mkdir()
    (run / "weights/best_ckpt.pth").write_bytes(b"selected-ema-weights")
    attempt = start_training_attempt(run, resumed=False,
                                     training_hardware={"device": "cpu"},
                                     monotonic_clock=lambda: 10.)
    finalize_training_attempt(run, attempt, outcome="completed", completed_epochs=1,
                              monotonic_clock=lambda: 20.)
    return run, dataset


def fake_factory(run, *, fail_coreml=False, fail_image=False, shift=False):
    """
    Supply deterministic model-free predictions and package records.
    """
    def factory(name, directory):
        """
        Create isolated format artifacts without touching source run state.
        """
        if name == "coreml" and fail_coreml:
            raise RuntimeError("converter unavailable")
        if name == "pytorch":
            path = run / "weights/best_ckpt.pth"
        else:
            directory.mkdir(parents=True, exist_ok=True)
            staged = directory / "temporary-export"
            staged.write_bytes(name.encode())
            path = workflow.publish_export(staged, directory, run, "yolox_nano", name)

        prediction_calls = 0

        def predict(image):
            """
            Return empty negatives or a known positive, optionally failing.
            """
            nonlocal prediction_calls
            prediction_calls += 1
            image_id = image.getpixel((0, 0))[0]
            if (fail_image and image_id == 3 and name == "onnx"
                    and prediction_calls > 3):
                raise OSError("bad image inference")
            if image_id != 1:
                return []
            offset = 5 if shift and name == "onnx" else 0
            return [{"category_id": 7, "bbox": [2+offset, 3, 10, 12], "score": .9}]

        return predict, {"backend": name, "device": "cpu", "precision": "float32",
                         "artifact": workflow.artifact_identity(path)}, lambda: None
    return factory


def execute(fixture_run, tmp_path, **kwargs):
    """
    Execute a compact real protocol/evaluator test without costly model inference.
    """
    run, dataset = fixture_run
    return workflow.execute_validation(
        run_dir=run, dataset_dir=dataset, model="yolox_nano", formats=("onnx", "coreml"),
        output_root=tmp_path / "outputs", models_root=tmp_path / "models",
        configuration={"resolution": 64, "postprocessing": {"score_floor": .001}},
        factory=fake_factory(run, **kwargs))


def test_discovery_uses_mtime_tiebreak_and_exact_override(fixture_run, tmp_path):
    """
    Keep the shared discovery order and reject invalid exact overrides.
    """
    import shutil

    run, _ = fixture_run
    other = run.with_name("yolox_nano_z")
    shutil.copytree(run, other)
    os.utime(run, ns=(100, 100))
    os.utime(other, ns=(100, 100))
    assert workflow.select_run(run.parent, "yolox_nano") == other
    assert workflow.select_run(run.parent, "yolox_nano", run) == run
    invalid = run.with_name("yolox_nano_incomplete")
    invalid.mkdir()
    os.utime(invalid, ns=(200, 200))
    assert workflow.select_run(run.parent, "yolox_nano") == other
    with pytest.raises(FileNotFoundError):
        workflow.select_run(run.parent, "yolox_nano", invalid)
    assert not workflow.eligible_run(run, "rfdetr_nano")


def test_latest_run_without_export_does_not_substitute_older(fixture_run, tmp_path):
    """
    Report missing latest-run results and validate exact export provenance.
    """
    run, _ = fixture_run
    selected = comparison.select_export("yolox_nano", run.parent)
    assert selected["run_dir"] == str(run)
    assert selected["manifest_path"] is None
    path = tmp_path / "pin.json"
    write_json(path, {"schema_version": 1, "model": "rfdetr_small", "run_dir": str(run)})
    with pytest.raises(ValueError, match="explicit export"):
        comparison.select_export("yolox_nano", run.parent,
                                 export_selection=path)


def test_independent_failure_and_complete_negative_coverage(fixture_run, tmp_path, monkeypatch):
    """
    Retain ONNX accuracy/timing after Core ML failure, with no source mutations.
    """
    monkeypatch.setattr(workflow.platform, "system", lambda: "Darwin")
    run, _ = fixture_run
    before = {p: file_digest(p) for p in run.rglob("*") if p.is_file()}
    result = execute(fixture_run, tmp_path, fail_coreml=True)
    assert result["source_run_unchanged"]
    assert all(file_digest(p) == digest for p, digest in before.items())
    assert result["formats"]["onnx"]["status"] == "completed"
    assert result["formats"]["onnx"]["parity_status"] == "passed"
    assert result["formats"]["coreml"]["failed_stage"] == "conversion_and_loading"
    artifact = read_json(result["formats"]["onnx"]["predictions"]["test"]["path"])
    assert artifact["image_ids"] == [1, 2, 3]
    assert [r["image_id"] for r in artifact["predictions"]] == [1]
    timing = read_json(result["formats"]["onnx"]["timing"]["path"])
    assert timing["sequence"] == [1, 2, 3] * 100
    assert len(timing["latencies_ms"]) == 3
    with pytest.raises(RuntimeError, match="requires review"):
        workflow.require_success(result)


@pytest.mark.parametrize("fail_image,shift,expected", [(True, False, "failed"),
                                                       (False, True, "completed")])
def test_inference_errors_and_parity_failures_remain_diagnostic(
        fixture_run, tmp_path, fail_image, shift, expected):
    """
    Never convert an inference error to empty coverage or accept failed parity.
    """
    result = execute(fixture_run, tmp_path, fail_image=fail_image, shift=shift)
    onnx = result["formats"]["onnx"]
    assert onnx["status"] == expected
    assert not onnx["accepted"]
    if shift:
        assert onnx["parity_status"] == "failed"
        assert set(onnx["predictions"]) == {"val", "test"}
    else:
        assert onnx["failed_stage"] == "full_split_evaluation"
        assert onnx["parity_status"] == "passed"
        assert not onnx.get("predictions")


def test_comparison_recomputes_accuracy_and_gates_timing(fixture_run, tmp_path):
    """
    Keep valid accuracy when host timing changes, and reject artifact tampering.
    """
    run, dataset = fixture_run
    result = execute(fixture_run, tmp_path)
    manifest_path = Path(result["manifest_path"])
    manifest = read_json(manifest_path)
    timing_record = manifest["formats"]["onnx"]["timing"]
    timing = read_json(timing_record["path"])
    timing["host"]["name"] = "different-host"
    write_json(timing_record["path"], timing)
    manifest["formats"]["onnx"]["timing"] = workflow.artifact_identity(Path(timing_record["path"]))
    write_json(manifest_path, manifest)
    kwargs = dict(runs_dir=run.parent,
                  dataset_dir=dataset, output_dir=tmp_path / "comparison",
                  models=("yolox_nano",), export_selections={"yolox_nano": manifest_path})
    compared = comparison.compare_runtime_results(**kwargs)
    table = compared["table"]
    assert set(table["timing_group"].dropna()) == {1, 2}
    assert table.loc[table["runtime"] == "onnx", "runtime_precision"].eq("float32").all()
    assert table.loc[table["runtime"] == "onnx", "precision"].eq(1.).all()
    assert table.loc[table["runtime"] == "onnx", "ap50_95"].notna().all()
    assert table.loc[table["runtime"] == "onnx", "delta_ap50_95"].eq(0).all()
    assert "torch" not in sys.modules
    record = manifest["formats"]["onnx"]["predictions"]["val"]
    Path(record["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        comparison.compare_runtime_results(**kwargs)


def test_annotation_and_checkpoint_mutation_rejected(fixture_run, tmp_path):
    """
    Reject changed source identities before joining otherwise valid runtime rows.
    """
    run, dataset = fixture_run
    result = execute(fixture_run, tmp_path)
    selection = {"model": "yolox_nano", "run_dir": str(run)}
    manifest = read_json(result["manifest_path"])
    checkpoint = run / "weights/best_ckpt.pth"
    checkpoint.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="checkpoint differs"):
        comparison.validate_manifest(manifest, selection, dataset)


def test_parity_threshold_crossings_axis_tolerances_and_categories():
    """
    Enforce inclusive thresholds, one-to-one categories and per-axis corner bounds.
    """
    row = {"category_id": 7, "bbox": [1., 1., 10., 10.], "score": .25}
    assert workflow.compare_detections([row], [row], 640, 640)["passed"]
    crossing = {**row, "score": .249}
    result = workflow.compare_detections([row], [crossing], 640, 640)
    assert result["failures"][0]["reason"] == "threshold_crossing"
    assert not workflow.compare_detections([row], [row, row], 640, 640)["passed"]
    assert not workflow.compare_detections([row], [{**row, "category_id": 1}], 640, 640)["passed"]
    assert workflow.compare_detections([row], [{**row, "bbox": [4.9, 2.9, 10., 10.]}], 2000, 640)["passed"]
    assert not workflow.compare_detections([row], [{**row, "bbox": [5.1, 1., 10., 10.]}], 2000, 640)["passed"]


def test_package_hash_and_source_mutation_protection(fixture_run, tmp_path):
    """
    Detect sorted package file changes and source mutation during a failed stage.
    """
    package = tmp_path / "package.mlpackage"
    package.mkdir()
    (package / "b").write_bytes(b"b")
    (package / "a").write_bytes(b"a")
    identity = workflow.artifact_identity(package)
    assert list(identity["files"]) == ["a", "b"]
    (package / "a").write_bytes(b"new")
    with pytest.raises(ValueError, match="hash mismatch"):
        comparison.checked_record(identity)
    run, dataset = fixture_run
    factory = fake_factory(run)

    def mutating_factory(name, directory):
        """
        Simulate an exporter accidentally changing selected checkpoint contents.
        """
        result = factory(name, directory)
        if name == "onnx":
            (run / "weights/best_ckpt.pth").write_bytes(b"mutated")
        return result

    result = workflow.execute_validation(
        run_dir=run, dataset_dir=dataset, model="yolox_nano", formats=("onnx",),
        output_root=tmp_path / "outputs", models_root=tmp_path / "models",
        configuration={"resolution": 64, "postprocessing": {"score_floor": .001}},
        factory=mutating_factory)
    assert not result["source_run_unchanged"]
    with pytest.raises(RuntimeError):
        workflow.require_success(result)


def test_comparison_imports_no_model_frameworks():
    """
    Prove a clean consumer process does not import inference frameworks.
    """
    subprocess.run([sys.executable, "-c", "import sys; "
                    "import detection_evaluation.runtime_comparison; "
                    "assert not {'torch','rfdetr','yolox','coremltools','onnxruntime'} "
                    "& set(sys.modules)"], check=True)


def test_raw_rfdetr_preserves_background_ranking_and_source_coordinates():
    """
    Discard background after query/class selection without clipping valid boxes.
    """
    import numpy as np

    boxes = np.array([[[.05, .05, .2, .2]]], dtype=np.float32)
    logits = np.array([[[0., 10.]]], dtype=np.float32)
    forward = lambda tensor: (boxes, logits)
    predictor = workflow.raw_predictor(forward, family="rfdetr", resolution=64,
                                       category_ids=[7], num_select=1)
    assert predictor(Image.new("RGB", (64, 64))) == []
    predictor = workflow.raw_predictor(forward, family="rfdetr", resolution=64,
                                       category_ids=[7], num_select=2)
    rows = predictor(Image.new("RGB", (64, 64)))
    assert len(rows) == 1 and rows[0]["category_id"] == 7
    assert rows[0]["bbox"][0] < 0


def test_explicit_run_export_conflict_and_latest_failed_attempt(fixture_run, tmp_path):
    """
    Keep failed latest attempts visible and reject conflicting explicit pins.
    """
    run, _ = fixture_run
    result = execute(fixture_run, tmp_path)
    path = Path(result["manifest_path"])
    assert path == (run / "export_validation/model_manifest.json").resolve()
    manifest = read_json(path)
    manifest["formats"]["onnx"] = {"status": "failed", "error": "converter failed"}
    write_json(path, manifest)
    selected = comparison.select_export("yolox_nano", run.parent)
    assert Path(selected["manifest_path"]) == path
    assert read_json(selected["manifest_path"])["formats"]["onnx"]["status"] == "failed"
    with pytest.raises(ValueError, match="different runs"):
        comparison.select_export("yolox_nano", run.parent,
                                  run_selection=tmp_path / "other", export_selection=path)


def test_dataset_pixel_and_annotation_identity_rejected(fixture_run):
    """
    Verify supplied local pixels and annotations, not archived Linux path strings.
    """
    run, dataset = fixture_run
    protocol = workflow.validate_run(run, "yolox_nano")
    identity = workflow.validate_dataset(protocol, dataset)
    assert identity["directory"] == str(dataset)
    Image.new("RGB", (64, 64), "red").save(dataset / "images/val/1.png")
    with pytest.raises(ValueError, match="pixel hash"):
        workflow.validate_dataset(protocol, dataset)


def test_relative_selection_is_independent_of_shell_cwd(fixture_run, monkeypatch, tmp_path):
    """
    Interpret configured relative run paths against the checkout root.
    """
    run, _ = fixture_run
    monkeypatch.setattr(workflow, "WORKSPACE_ROOT", tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert workflow.select_run(Path("runs"), "yolox_nano") == run
    assert workflow.select_run(Path("runs"), "yolox_nano", Path("runs") / run.name) == run


def test_unsupported_coreml_host_is_explicit_without_losing_onnx(
        fixture_run, tmp_path, monkeypatch):
    """
    Keep successful ONNX results while reporting an actionable unsupported host.
    """
    monkeypatch.setattr(workflow.platform, "system", lambda: "Linux")
    result = execute(fixture_run, tmp_path)
    assert result["formats"]["onnx"]["status"] == "completed"
    assert result["formats"]["coreml"]["status"] == "failed"
    assert "requires macOS" in result["formats"]["coreml"]["error"]


def test_parity_follows_reference_score_order_and_highest_iou():
    """
    Reject a lower-IoU reassignment even when it could satisfy numeric tolerances.
    """
    reference = [{"category_id": 7, "bbox": [0, 0, 10, 10], "score": .9},
                 {"category_id": 7, "bbox": [1, 0, 10, 10], "score": .8}]
    runtime = [{"category_id": 7, "bbox": [0, 0, 10, 10], "score": .8},
               {"category_id": 7, "bbox": [1, 0, 10, 10], "score": .9}]
    result = workflow.compare_detections(reference, runtime, 640, 640)
    assert not result["passed"]
    assert any(failure["kind"] == "score_difference" for failure in result["failures"])
    with pytest.raises(ValueError, match="finite"):
        workflow.compare_detections([{**reference[0], "score": float('nan')}], [], 640, 640)


def test_comparison_rejects_different_runtime_checkpoint_provenance(fixture_run, tmp_path):
    """
    Reject internally rehashed predictions attributed to a different checkpoint.
    """
    run, dataset = fixture_run
    result = execute(fixture_run, tmp_path)
    path = Path(result["manifest_path"])
    manifest = read_json(path)
    record = manifest["formats"]["onnx"]["predictions"]["val"]
    artifact = read_json(record["path"])
    artifact["metadata"]["checkpoint_sha256"] = "a" * 64
    write_json(record["path"], artifact)
    manifest["formats"]["onnx"]["predictions"]["val"] = workflow.artifact_identity(Path(record["path"]))
    write_json(path, manifest)
    with pytest.raises(ValueError, match="provenance differs"):
        comparison.compare_runtime_results(
            runs_dir=run.parent,
            dataset_dir=dataset, output_dir=tmp_path / "comparison",
            models=("yolox_nano",), export_selections={"yolox_nano": path})


def test_comparison_rejects_changed_annotation_identity(fixture_run, tmp_path):
    """
    Reject changed annotations even if the same split image IDs are present.
    """
    run, dataset = fixture_run
    result = execute(fixture_run, tmp_path)
    manifest = read_json(result["manifest_path"])
    path = dataset / "annotations/instances_test.json"
    annotations = read_json(path)
    annotations["annotations"][0]["bbox"][0] += 1
    write_json(path, annotations)
    with pytest.raises(ValueError, match="annotations differ"):
        comparison.validate_manifest(manifest, {"run_dir": str(run), "model": "yolox_nano"}, dataset)


def test_nonfinite_raw_inference_is_not_an_empty_success():
    """
    Fail invalid raw tensors even when threshold filtering would remove them.
    """
    import numpy as np

    predictor = workflow.raw_predictor(
        lambda tensor: (np.zeros((1, 1, 4)), np.full((1, 1, 2), np.nan)),
        family="rfdetr", resolution=64, category_ids=[7])
    with pytest.raises(ValueError, match="nonfinite output"):
        predictor(Image.new("RGB", (64, 64)))


def test_rerun_updates_one_canonical_export_set(fixture_run, tmp_path):
    """
    Replace the saved result in place without accumulating packages or reports.
    """
    run, _ = fixture_run
    first = execute(fixture_run, tmp_path)
    first_id = read_json(first["manifest_path"])["export_id"]
    export_root = tmp_path / "models/exports/basketball/yolox_nano" / run.name
    obsolete = export_root / "20261116T203432209355Z_deadbeef"
    obsolete.mkdir()
    (obsolete / "old-export").write_bytes(b"obsolete")
    second = execute(fixture_run, tmp_path)
    assert not obsolete.exists()
    assert second["manifest_path"] == first["manifest_path"]
    assert Path(second["report_dir"]) == (run / "export_validation").resolve()
    assert read_json(second["manifest_path"])["export_id"] != first_id
    assert len(list((run / "export_validation").glob("model_manifest.json"))) == 1
    assert len(list(run.rglob("*.onnx"))) == 1
    export_root = tmp_path / "models/exports/basketball/yolox_nano" / run.name
    assert Path(second["export_dir"]) == export_root
    assert {p.name for p in export_root.iterdir()} <= {"onnx", "coreml"}
    assert (run / "weights/best_ckpt.onnx").is_symlink()
    assert (run / "weights/best_ckpt.onnx").resolve() == (
        export_root / "onnx/best_ckpt.onnx")
    assert second["source_run_unchanged"]


def test_canonical_export_lock_rejects_concurrent_writers(fixture_run, tmp_path):
    """
    Prevent two validations from overwriting the same canonical export set.
    """
    import fcntl

    run, _ = fixture_run
    report = run / "export_validation"
    report.mkdir()
    with (report / ".export.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert workflow.export_in_progress(run)
        with pytest.raises(RuntimeError, match="already updating"):
            execute(fixture_run, tmp_path)
    assert not workflow.export_in_progress(run)


def test_finder_metadata_does_not_change_model_or_training_identity(tmp_path):
    """
    Ignore macOS folder metadata while retaining checksums for model payloads.
    """
    package = tmp_path / "model.mlpackage"
    package.mkdir()
    (package / "weights.bin").write_bytes(b"weights")
    original = workflow.artifact_identity(package)
    (package / ".DS_Store").write_bytes(b"Finder view state")
    assert workflow.artifact_identity(package) == original
    assert not workflow.training_source_file(tmp_path / ".DS_Store", tmp_path,
                                              "rfdetr_small")
