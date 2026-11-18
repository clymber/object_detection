"""
Hardlink the selected RF-DETR Small export and two validation images into the app.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


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


def stage(root: Path, manifest_path: Path) -> None:
    """
    Verify provenance, link resources and write portable sample expectations.
    """
    manifest = json.loads(manifest_path.read_text())
    if manifest["model"] != "rfdetr_small":
        raise ValueError("This staging helper requires the RF-DETR Small export")
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
    target = app / "Models/rfdetr_small.mlpackage"
    for relative in files:
        hardlink_file(package / relative, target / relative)

    selection = json.loads(verified_record(manifest["parity_manifest"]).read_text())
    parity_path = verified_record(coreml["parity"])
    parity = json.loads(parity_path.read_text())
    annotation_record = manifest["dataset"]["annotations"]["val"]
    annotations = json.loads(verified_record(annotation_record).read_text())
    images = {image["id"]: image for image in annotations["images"]}
    source_root = Path(manifest["dataset"]["directory"]) / "images/val"
    samples = []
    for kind in ("positive", "negative"):
        image_id = next(
            image_id for image_id in selection[f"{kind}_ids"]
            if bool(retained(parity["reference_detections"][str(image_id)]))
            == (kind == "positive")
        )
        image = images[image_id]
        source = source_root / image["file_name"]
        name = f"{kind}_{image_id}{source.suffix.lower()}"
        destination = app / "Resources/TestImages" / name
        source_hash = file_hash(source)
        hardlink_file(source, destination)
        if file_hash(destination) != source_hash:
            raise ValueError(f"Image bytes changed during staging: {name}")
        samples.append({
            "kind": kind,
            "image_id": image_id,
            "file_name": name,
            "source_path": source.relative_to(root).as_posix(),
            "sha256": source_hash,
            "width": image["width"],
            "height": image["height"],
            "ground_truth": [
                {key: row[key] for key in ("category_id", "bbox", "iscrowd")}
                for row in annotations["annotations"] if row["image_id"] == image_id
            ],
            "expected_checkpoint_detections": retained(
                parity["reference_detections"][str(image_id)]
            ),
            "expected_mac_coreml_detections": retained(
                parity["detections"][str(image_id)]
            ),
        })

    sample_manifest = {
        "schema_version": 1,
        "model": manifest["model"],
        "run_id": Path(manifest["run_dir"]).name,
        "model_resource": "rfdetr_small.mlpackage",
        "model_sha256": artifact["sha256"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "export_manifest_path": manifest_path.relative_to(root).as_posix(),
        "reference_parity_sha256": coreml["parity"]["sha256"],
        "annotation_sha256": annotation_record["sha256"],
        "split": "val",
        "confidence_threshold": 0.25,
        "max_detections": 100,
        "box_convention": "unclipped source-pixel top-left xywh",
        "class_mapping": {"0": 1},
        "samples": samples,
    }
    path = app / "Resources/TestImages/sample_manifest.json"
    path.write_text(json.dumps(sample_manifest, indent=2) + "\n")
    print(f"Hardlinked {len(files)} package files: {target.relative_to(root)}")
    for sample in samples:
        print(f"Staged {sample['kind']} image {sample['image_id']}: "
              f"Resources/TestImages/{sample['file_name']}")
    print(f"Saved provenance and expected detections: {path.relative_to(root)}")


def main() -> None:
    """
    Stage the pinned selection from any working directory without copying weights.
    """
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=root / "outputs/exports/basketball/rfdetr_small"
        / "rfdetr_small_20261106T124419/model_manifest.json",
    )
    args = parser.parse_args()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    stage(root, manifest_path.resolve())


if __name__ == "__main__":
    main()
