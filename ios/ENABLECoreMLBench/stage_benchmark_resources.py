"""
Hardlink the three selected Core ML models and the 25 frozen test images.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


SELECTED_RUNS = {
    "rfdetr_small": "rfdetr_small_20261106T124419",
    "rfdetr_nano": "rfdetr_nano_20261116T115819",
    "yolox_nano": "yolox_nano_20261105T200341",
}


def sample_manifest_name(model: str) -> str:
    """
    Keep the frozen Small manifest name and use distinct names for other models.
    """
    return "sample_manifest" if model == "rfdetr_small" else f"sample_manifest_{model}"


def file_hash(path: Path) -> str:
    """
    Hash a resource without modifying its source inode.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_record(record: dict) -> Path:
    """
    Reject changed input records before staging resources derived from them.
    """
    path = Path(record["path"])
    if file_hash(path) != record["sha256"]:
        raise ValueError(f"Source record hash changed: {path}")
    return path


def hardlink_file(source: Path, destination: Path) -> None:
    """
    Create a hardlink idempotently, refusing to replace unrelated target files.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise FileExistsError(f"Expected a hardlink, found symlink: {destination}")
    if destination.exists():
        if not os.path.samefile(source, destination):
            raise FileExistsError(
                f"Target exists with a different inode: {destination}"
            )
        return
    os.link(source, destination)
    if not os.path.samefile(source, destination):
        raise RuntimeError(f"Hardlink verification failed: {destination}")


def retained(rows: list[dict]) -> list[dict]:
    """
    Keep the reference's inclusive confidence threshold and stable top-100 cap.
    """
    return sorted(
        (row for row in rows if row["score"] >= 0.25),
        key=lambda row: -row["score"],
    )[:100]


def stage_sample(
    root: Path, app: Path, dataset: Path, kind: str,
    image: dict, annotations: list[dict], references: dict,
) -> dict:
    """
    Link one original image and retain its split, ground truth and saved outputs.
    """
    image_id = image["id"]
    source = dataset / "images/test" / image["file_name"]
    name = f"test_{kind}_{image_id}{source.suffix.lower()}"
    source_hash = file_hash(source)
    destination = app / "Resources/TestImages" / name
    hardlink_file(source, destination)
    if file_hash(destination) != source_hash:
        raise ValueError(f"Image bytes changed during staging: {name}")
    return {
        "kind": kind, "split": "test", "image_id": image_id, "file_name": name,
        "source_path": source.relative_to(root).as_posix(),
        "sha256": source_hash, "width": image["width"], "height": image["height"],
        "ground_truth": [
            {key: row[key] for key in ("category_id", "bbox", "iscrowd")}
            for row in annotations if row["image_id"] == image_id
        ],
        "expected_checkpoint_detections": retained(
            references["pytorch"].get(str(image_id), [])
        ),
        "expected_onnx_detections": retained(
            references["onnx"].get(str(image_id), [])
        ),
        "expected_mac_coreml_detections": retained(
            references["coreml"].get(str(image_id), [])
        ),
    }


def stage(root: Path, model: str, manifest_path: Path) -> None:
    """
    Verify provenance, link resources and write portable sample expectations.
    """
    manifest = json.loads(manifest_path.read_text())
    if model not in SELECTED_RUNS or manifest["model"] != model:
        raise ValueError(f"Require the selected {model} export")
    coreml = manifest["formats"]["coreml"]
    if not coreml["accepted"] or coreml["parity_status"] != "passed":
        raise ValueError("Require an accepted export with development parity passed")
    verified_record(manifest["checkpoint"])
    artifact = coreml["artifact"]
    package = Path(artifact["path"])
    files = {
        path.relative_to(package).as_posix(): file_hash(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != ".DS_Store"
    }
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if files != artifact["files"] or identity != artifact["sha256"]:
        raise ValueError("Selected Core ML package no longer matches its manifest")

    app = root / "ios/ENABLECoreMLBench/ENABLECoreMLBench"
    target = app / f"Models/{model}.mlpackage"
    for relative in files:
        hardlink_file(package / relative, target / relative)

    dataset = Path(manifest["dataset"]["directory"])
    samples = []
    test_record = manifest["dataset"]["annotations"]["test"]
    test_annotations = json.loads(verified_record(test_record).read_text())
    test_images = {image["id"]: image for image in test_annotations["images"]}
    positive_ids = {row["image_id"] for row in test_annotations["annotations"]}
    test_references = {}
    prediction_records = {}
    for runtime in ("pytorch", "onnx", "coreml"):
        record = manifest["formats"][runtime]["predictions"]["test"]
        artifact_data = json.loads(verified_record(record).read_text())
        if (artifact_data["metadata"]["checkpoint_sha256"]
                != manifest["checkpoint"]["sha256"]
                or artifact_data["annotation_sha256"] != test_record["sha256"]
                or set(artifact_data["image_ids"]) != set(test_images)):
            raise ValueError(f"Wrong checkpoint or test coverage for {runtime}")
        rows_by_image = {str(image_id): [] for image_id in test_images}
        for row in artifact_data["predictions"]:
            rows_by_image[str(row["image_id"])].append({
                key: row[key] for key in ("bbox", "category_id", "score")
            })
        test_references[runtime] = rows_by_image
        prediction_records[runtime] = {
            "path": Path(record["path"]).relative_to(root).as_posix(),
            "sha256": record["sha256"],
        }
    # First 20 positive and first 5 negative test examples from the pinned export.
    # The notebook's original 4+4 grid is retained within this expanded selection.
    for kind, ids in (
        ("positive", tuple(range(6988, 7008))),
        ("negative", (7028, 7030, 7032, 7046, 7054)),
    ):
        for image_id in ids:
            if (image_id in positive_ids) != (kind == "positive"):
                raise ValueError(f"Test sample ground truth changed: {image_id}")
            samples.append(stage_sample(
                root, app, dataset, kind, test_images[image_id],
                test_annotations["annotations"], test_references,
            ))

    sample_manifest = {
        "schema_version": 3,
        "scope": "test_examples",
        "model": manifest["model"],
        "run_id": Path(manifest["run_dir"]).name,
        "model_resource": f"{model}.mlpackage",
        "model_sha256": artifact["sha256"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "export_manifest_path": manifest_path.relative_to(root).as_posix(),
        "split": "test",
        "annotation_sha256": test_record["sha256"],
        "prediction_records": prediction_records,
        "confidence_threshold": 0.25,
        "max_detections": 100,
        "box_convention": "unclipped source-pixel top-left xywh",
        "class_mapping": {"0": 1},
        "samples": samples,
    }
    path = app / "Resources/TestImages" / f"{sample_manifest_name(model)}.json"
    path.write_text(json.dumps(sample_manifest, indent=2) + "\n")
    print(f"Hardlinked {len(files)} package files: {target.relative_to(root)}")
    print(f"Staged {len(samples)} frozen test images for {model}")
    print(f"Saved provenance and expected detections: {path.relative_to(root)}")


def main() -> None:
    """
    Stage all selected models or one explicit model from any working directory.
    """
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("all", *SELECTED_RUNS), default="all")
    parser.add_argument(
        "--manifest", type=Path,
        help="Override one export manifest; requires --model other than all",
    )
    args = parser.parse_args()
    if args.manifest is not None and args.model == "all":
        parser.error("--manifest requires an explicit --model")
    models = SELECTED_RUNS if args.model == "all" else (args.model,)
    for model in models:
        path = args.manifest or (
            root / "outputs/exports/basketball" / model
            / SELECTED_RUNS[model] / "model_manifest.json"
        )
        if not path.is_absolute():
            path = root / path
        stage(root, model, path.resolve())


if __name__ == "__main__":
    main()
