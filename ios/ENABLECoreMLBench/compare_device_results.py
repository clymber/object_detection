"""
Compare precise native test-image results with a matching saved sample manifest.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIRECTORY = ROOT / (
    "ios/ENABLECoreMLBench/ENABLECoreMLBench/Resources/TestImages"
)
MODEL_MANIFESTS = {
    model: MANIFEST_DIRECTORY / f"test_images_{model}_manifest.json"
    for model in ("rfdetr_small", "rfdetr_nano", "yolox_nano")
}
DEFAULT_MANIFEST = MODEL_MANIFESTS["rfdetr_small"]
REFERENCE_KEYS = {
    "pytorch": "expected_checkpoint_detections",
    "onnx": "expected_onnx_detections",
    "mac_coreml": "expected_mac_coreml_detections",
}


def reject_constant(value: str) -> None:
    """
    Reject nonstandard NaN/Infinity JSON instead of silently accepting them.
    """
    raise ValueError(f"Non-finite JSON number: {value}")


def read_json(path: Path) -> dict:
    """
    Read an object with finite JSON numbers and no duplicate object keys.
    """
    def unique_object(pairs: list) -> dict:
        """
        Reject ambiguous duplicate keys at any nesting depth.
        """
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), parse_constant=reject_constant,
                       object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def finite_number(value: object) -> bool:
    """
    Accept finite JSON numbers while excluding booleans.
    """
    return type(value) in (int, float) and math.isfinite(value)


def validate_predictions(rows: object) -> None:
    """
    Reject malformed, unretained or unordered outputs before matching boxes.
    """
    if not isinstance(rows, list) or len(rows) > 100:
        raise ValueError("Detections must be an array of at most 100 entries")
    previous = 1.0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each detection must be an object")
        score, box = row.get("score"), row.get("bbox")
        if type(row.get("category_id")) is not int or row["category_id"] < 0:
            raise ValueError("Invalid category ID")
        if not finite_number(score) or not 0.25 <= score <= previous:
            raise ValueError("Scores must be finite, retained and descending")
        previous = score
        if (not isinstance(box, list) or len(box) != 4
                or not all(finite_number(value) for value in box)
                or box[2] <= 0 or box[3] <= 0
                or not math.isfinite(box[0] + box[2])
                or not math.isfinite(box[1] + box[3])):
            raise ValueError("Invalid source-pixel xywh box")


def corners(box: list) -> list:
    """
    Convert unclipped top-left xywh to the four source-pixel corners.
    """
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def iou(first: list, second: list) -> float:
    """
    Compute continuous source-pixel intersection over union without clipping.
    """
    a, b = corners(first), corners(second)
    intersection = (max(0, min(a[2], b[2]) - max(a[0], b[0]))
                    * max(0, min(a[3], b[3]) - max(a[1], b[1])))
    return intersection / (first[2] * first[3] + second[2] * second[3]
                           - intersection)


def compare_predictions(actual: list, expected: list,
                        width: int, height: int) -> dict:
    """
    Match in reference score order using the highest-IoU unmatched same-class box.
    """
    validate_predictions(actual)
    validate_predictions(expected)
    classes_match = Counter(row["category_id"] for row in actual) == Counter(
        row["category_id"] for row in expected)
    limits = [max(2, 0.002 * width), max(2, 0.002 * height)] * 2
    unmatched = list(range(len(actual)))
    matches = []
    missing = []
    for reference_index, reference in enumerate(expected):
        candidates = [i for i in unmatched
                      if actual[i]["category_id"] == reference["category_id"]]
        if not candidates:
            missing.append(reference_index)
            continue
        index = max(candidates, key=lambda i: iou(actual[i]["bbox"], reference["bbox"]))
        unmatched.remove(index)
        errors = [abs(a - b) for a, b in zip(corners(actual[index]["bbox"]),
                                             corners(reference["bbox"]))]
        score_error = abs(actual[index]["score"] - reference["score"])
        passed = all(error <= limit for error, limit in zip(errors, limits))
        matches.append({
            "reference_index": reference_index, "actual_index": index,
            "category_id": reference["category_id"],
            "iou": iou(actual[index]["bbox"], reference["bbox"]),
            "corner_errors": errors, "corner_tolerances": limits,
            "score_error": score_error,
            "passed": passed and score_error <= 0.02,
        })
    return {
        "passed": classes_match and not missing and not unmatched
        and all(match["passed"] for match in matches),
        "actual_count": len(actual), "reference_count": len(expected),
        "classes_match": classes_match, "matches": matches,
        "unmatched_reference_indices": missing, "unmatched_actual_indices": unmatched,
    }


def compare_results(results: dict, manifest: dict, manifest_sha256: str,
                    *, allow_host: bool = False) -> dict:
    """
    Check provenance, complete input coverage and all three per-image references.
    """
    errors = []
    samples = manifest.get("samples", [])
    if (manifest.get("schema_version") != 3 or manifest.get("split") != "test"
            or not samples or any(sample["split"] != "test" for sample in samples)
            or len({sample["image_id"] for sample in samples}) != len(samples)
            or len({sample["file_name"] for sample in samples}) != len(samples)):
        raise ValueError("Require a nonempty test manifest with unique images")
    # Original schema-3 manifests predate the explicit scope field.
    scope = manifest.get("scope", "eight_test_examples")
    required = {
        "schema_version": 1, "scope": scope, "status": "completed",
        "sample_manifest_sha256": manifest_sha256,
    }
    for key in ("model", "run_id", "model_sha256", "checkpoint_sha256", "split",
                "confidence_threshold", "max_detections", "box_convention"):
        required[key] = manifest[key]
    for key, value in required.items():
        if type(results.get(key)) is not type(value) or results[key] != value:
            errors.append(f"Mismatch in {key}")
    runtime = results.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("Runtime must be an object")
    for key in ("platform", "hardware", "os_version", "compute_units"):
        if not isinstance(runtime.get(key), str) or not runtime[key]:
            errors.append(f"Missing runtime {key}")
    if runtime.get("compute_units") != "all" and not allow_host:
        errors.append("Expected the app's all compute-units configuration")
    if runtime.get("platform") != "ios" and not allow_host:
        errors.append("Physical iPhone results required; host/simulator run is not device parity")
    images = results.get("images")
    if not isinstance(images, list) or not all(isinstance(row, dict) for row in images):
        raise ValueError("Images must be an array of objects")
    ids = [row.get("image_id") for row in images]
    if not all(type(image_id) is int for image_id in ids):
        raise ValueError("Every result needs an integer image_id")
    expected_ids = [sample["image_id"] for sample in manifest["samples"]]
    if Counter(ids) != Counter(expected_ids):
        errors.append("Image coverage differs: missing, duplicate or unexpected IDs")
    comparisons = []
    for sample in manifest["samples"]:
        rows = [row for row in images if row["image_id"] == sample["image_id"]]
        issues = []
        references = {}
        if len(rows) != 1:
            issues.append("Expected exactly one image result")
        else:
            row = rows[0]
            for key, value in (
                ("file_name", sample["file_name"]), ("image_sha256", sample["sha256"]),
                ("width", sample["width"]), ("height", sample["height"]),
                ("status", "completed"),
            ):
                if type(row.get(key)) is not type(value) or row[key] != value:
                    issues.append(f"Mismatch in {key}")
            if row.get("error") is not None:
                issues.append(f"Image failed: {row['error']}")
            if not finite_number(row.get("latency_ms")) or row["latency_ms"] <= 0:
                issues.append("Missing positive finite detection latency")
            try:
                for runtime_name, key in REFERENCE_KEYS.items():
                    references[runtime_name] = compare_predictions(
                        row.get("detections"), sample[key],
                        sample["width"], sample["height"])
            except ValueError as error:
                issues.append(str(error))
        comparisons.append({
            "image_id": sample["image_id"], "file_name": sample["file_name"],
            "passed": not issues and bool(references)
            and all(reference["passed"] for reference in references.values()),
            "errors": issues, "references": references,
        })
    return {
        "schema_version": 1, "scope": scope,
        "status": "passed" if not errors and all(row["passed"] for row in comparisons)
        else "failed",
        "execution_scope": "physical_iphone" if runtime.get("platform") == "ios"
        else "host_or_simulator",
        "runtime": runtime, "model": manifest["model"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "model_sha256": manifest["model_sha256"],
        "sample_manifest_sha256": manifest_sha256,
        "tolerances": {"corner_px": "max(2, 0.002 * source axis)", "score": 0.02},
        "errors": errors, "images": comparisons,
    }


def main() -> int:
    """
    Save a reproducible report and return nonzero for failed or invalid attempts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSON shared from the app")
    parser.add_argument("--manifest", type=Path,
                        help="Override the manifest selected from the exported model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-host", action="store_true",
                        help="Developer checks only; permit host/simulator results")
    args = parser.parse_args()
    output = args.output or ROOT / "outputs/ios_parity" / args.results.stem / "parity_report.json"
    protected = [args.results, *MODEL_MANIFESTS.values()]
    if args.manifest is not None:
        protected.append(args.manifest)
    if output.resolve() in {path.resolve() for path in protected}:
        parser.error("Report output must not overwrite the results or manifest")
    try:
        results = read_json(args.results)
        if args.manifest is None and results.get("model") not in MODEL_MANIFESTS:
            raise ValueError(f"Unsupported exported model: {results.get('model')}")
        manifest_path = args.manifest or MODEL_MANIFESTS[results["model"]]
        report = compare_results(
            results, read_json(manifest_path),
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            allow_host=args.allow_host)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report = {"schema_version": 1, "scope": "test_examples",
                  "status": "failed", "errors": [str(error)], "images": []}
    if args.results.is_file():
        report["results_sha256"] = hashlib.sha256(args.results.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    passed = sum(row["passed"] for row in report["images"])
    scope = report.get("execution_scope", "invalid_input")
    total = len(report["images"])
    print(f"{report['status'].upper()}: {passed}/{total} images ({scope}); report: {output}")
    for error in report["errors"]:
        print(f"  {error}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
