"""
Exercise false-pass protections and correspondence in the iOS parity comparator.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest


TOOLS = runpy.run_path(str(Path(__file__).resolve().parents[1]
                           / "compare_device_results.py"))
MANIFEST_PATH = TOOLS["DEFAULT_MANIFEST"]


def prediction_export(manifest: dict, digest: str) -> dict:
    """
    Build synthetic device results from one model's saved checkpoint predictions.
    """
    results = {
        key: manifest[key] for key in (
            "model", "run_id", "model_sha256", "checkpoint_sha256", "split",
            "confidence_threshold", "max_detections", "box_convention")
    }
    results.update({
        "schema_version": 1, "scope": manifest["scope"], "status": "completed",
        "sample_manifest_sha256": digest,
        "runtime": {"platform": "ios", "hardware": "synthetic_test_fixture",
                    "os_version": "test", "compute_units": "all"},
        "images": [{
            "image_id": sample["image_id"], "file_name": sample["file_name"],
            "image_sha256": sample["sha256"], "width": sample["width"],
            "height": sample["height"], "status": "completed", "latency_ms": 1.0,
            "detections": deepcopy(sample["expected_checkpoint_detections"]),
        } for sample in manifest["samples"]],
    })
    return results


class CompareDeviceResultsTests(unittest.TestCase):
    """
    Use synthetic native outputs so these checks require no phone or model.
    """

    def setUp(self):
        """
        Start with the saved checkpoint results and matching provenance.
        """
        self.manifest = json.loads(MANIFEST_PATH.read_text())
        self.digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
        self.results = prediction_export(self.manifest, self.digest)

    def report(self, results=None, **kwargs):
        """
        Compare one attempt against the unchanged pinned manifest.
        """
        return TOOLS["compare_results"](
            results or self.results, self.manifest, self.digest, **kwargs)

    def test_all_references_and_empty_images(self):
        """
        Cover all three references, full input coverage and successful empty rows.
        """
        report = self.report()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["images"]), len(self.manifest["samples"]))
        self.assertTrue(all(len(row["references"]) == 3 for row in report["images"]))
        negative = report["images"][-1]["references"]["pytorch"]
        self.assertTrue(negative["passed"])
        self.assertEqual(negative["matches"], [])

    def test_model_specific_cli_references(self):
        """
        Select Nano/YOLOX references automatically and reject a Small manifest.
        """
        fields = ("image_id", "file_name", "sha256", "width", "height",
                  "split", "ground_truth")
        frozen_inputs = [{key: row[key] for key in fields}
                         for row in self.manifest["samples"]]
        with tempfile.TemporaryDirectory() as directory:
            for model in ("rfdetr_nano", "yolox_nano"):
                with self.subTest(model=model):
                    manifest_path = TOOLS["MODEL_MANIFESTS"][model]
                    manifest = json.loads(manifest_path.read_text())
                    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                    self.assertEqual([{key: row[key] for key in fields}
                                      for row in manifest["samples"]], frozen_inputs)
                    source = Path(directory) / f"{model}.json"
                    source.write_text(json.dumps(prediction_export(manifest, digest)))
                    output = Path(directory) / f"{model}_report.json"
                    command = [sys.executable, TOOLS["__file__"], str(source),
                               "--output", str(output)]
                    run = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    report = json.loads(output.read_text())
                    self.assertEqual(report["model"], model)
                    self.assertEqual(report["sample_manifest_sha256"], digest)
                    run = subprocess.run(command + ["--manifest", str(MANIFEST_PATH)],
                                         capture_output=True, text=True)
                    self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
                    self.assertEqual(json.loads(output.read_text())["status"], "failed")

    def test_missing_duplicate_and_unexpected_images(self):
        """
        Reject coverage substitutions, including duplicate successful images.
        """
        for change in ("missing", "duplicate", "unexpected"):
            with self.subTest(change=change):
                attempt = deepcopy(self.results)
                if change == "missing":
                    attempt["images"].pop()
                elif change == "duplicate":
                    attempt["images"][-1] = deepcopy(attempt["images"][0])
                else:
                    attempt["images"][-1]["image_id"] = 999999
                self.assertEqual(self.report(attempt)["status"], "failed")

    def test_archived_eight_image_manifest(self):
        """
        Keep historical device exports comparable using their original manifest.
        """
        ids = {6988, 6989, 6990, 6991, 7028, 7030, 7032, 7046}
        self.manifest.pop("scope")
        self.manifest["samples"] = [sample for sample in self.manifest["samples"]
                                    if sample["image_id"] in ids]
        self.results["images"] = [row for row in self.results["images"]
                                  if row["image_id"] in ids]
        self.results["scope"] = "eight_test_examples"
        self.assertEqual(self.report()["status"], "passed")

    def test_invalid_manifest_coverage(self):
        """
        Reject an empty or ambiguous expected set instead of passing vacuously.
        """
        for samples in ([], [self.manifest["samples"][0]] * 2):
            with self.subTest(samples=len(samples)):
                manifest = deepcopy(self.manifest)
                manifest["samples"] = samples
                with self.assertRaises(ValueError):
                    TOOLS["compare_results"](self.results, manifest, self.digest)

    def test_provenance_and_configuration_mismatches(self):
        """
        Reject another checkpoint, resource, threshold, geometry or manifest.
        """
        for key in ("model_sha256", "checkpoint_sha256", "sample_manifest_sha256",
                    "run_id", "model", "split", "confidence_threshold",
                    "max_detections", "box_convention", "scope", "schema_version"):
            with self.subTest(key=key):
                attempt = deepcopy(self.results)
                attempt[key] = "changed"
                self.assertEqual(self.report(attempt)["status"], "failed")
        for key in ("file_name", "image_sha256", "width", "height"):
            with self.subTest(key=key):
                attempt = deepcopy(self.results)
                attempt["images"][0][key] = "changed"
                self.assertEqual(self.report(attempt)["status"], "failed")

    def test_incomplete_or_failed_empty_row(self):
        """
        Prevent a failed negative-image read from appearing to be an empty success.
        """
        self.results["status"] = "incomplete"
        self.results["images"][-1].update(status="error", error="Read failed")
        report = self.report()
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["images"][-1]["passed"])

    def test_host_does_not_count_as_device_parity(self):
        """
        Require an explicit developer override for host and simulator checks.
        """
        self.results["runtime"].update(platform="macos", compute_units="cpuOnly")
        self.assertEqual(self.report()["status"], "failed")
        report = self.report(allow_host=True)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["execution_scope"], "host_or_simulator")

    def test_numerical_count_class_and_threshold_failures(self):
        """
        Catch corner, score, class, count and malformed numeric output failures.
        """
        for change in ("corner", "score", "class", "count", "nan", "infinity",
                       "threshold", "negative_width", "boolean_score"):
            with self.subTest(change=change):
                attempt = deepcopy(self.results)
                row = attempt["images"][0]["detections"][0]
                if change == "corner":
                    row["bbox"][0] += 2.1
                elif change == "score":
                    row["score"] -= 0.021
                elif change == "class":
                    row["category_id"] = 0
                elif change == "count":
                    attempt["images"][0]["detections"] = []
                elif change == "nan":
                    row["score"] = float("nan")
                elif change == "infinity":
                    row["bbox"][0] = float("inf")
                elif change == "threshold":
                    row["score"] = 0.249
                elif change == "negative_width":
                    row["bbox"][2] = -1
                else:
                    row["score"] = True
                self.assertEqual(self.report(attempt)["status"], "failed")

    def test_iou_matching_and_inclusive_tolerances(self):
        """
        Match by IoU rather than array index and retain inclusive limit behavior.
        """
        expected = [
            {"category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.75},
            {"category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.75},
        ]
        # Equal scores permit order reversal while preserving descending output order.
        actual = deepcopy(expected[::-1])
        actual[1]["bbox"][0] += 2
        report = TOOLS["compare_predictions"](actual, expected, 640, 640)
        self.assertTrue(report["passed"])
        self.assertEqual([row["actual_index"] for row in report["matches"]], [1, 0])
        expected = [{"category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.5}]
        actual = [{"category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.484375}]
        self.assertTrue(TOOLS["compare_predictions"](actual, expected, 640, 640)["passed"])

    def test_strict_json_input(self):
        """
        Reject non-finite constants and duplicate keys before comparison.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ('{"score": NaN}', '{"status": "failed", "status": "completed"}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    TOOLS["read_json"](path)


if __name__ == "__main__":
    unittest.main()
