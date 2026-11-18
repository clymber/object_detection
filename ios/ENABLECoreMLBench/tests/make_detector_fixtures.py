"""
Write ignored Swift integration fixtures from the existing Python references.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from detection_evaluation.onnx_inference import preprocess_rfdetr, preprocess_yolox


def main():
    """
    Save asymmetric RGB resize cases and the recorded development parity images.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve()
    output = root / "outputs/ios_builds/detection_pipeline/fixtures"
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    rng = np.random.default_rng(42)
    for index, (width, height) in enumerate([(5, 3), (17, 31), (731, 413), (1, 1)]):
        rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        rgb.tofile(output / f"resize_{index}.rgb")
        for family, prepare in [
            ("rfdetr_small", preprocess_rfdetr), ("yolox_nano", preprocess_yolox)
        ]:
            prepared = prepare(Image.fromarray(rgb), 640, 640)
            tensor_file = f"resize_{index}_{family}.f32"
            prepared.tensor.tofile(output / tensor_file)
            cases.append({
                "model": family, "width": width, "height": height,
                "rgb": f"resize_{index}.rgb", "tensor": tensor_file,
            })
    (output / "preprocessing.json").write_text(json.dumps(cases))
    print(f"Saved {len(cases)} Python preprocessing references to {output}")
    models = []
    for manifest_path in sorted((root / "outputs/exports/basketball").glob(
        "*/*/model_manifest.json"
    )):
        manifest = json.loads(manifest_path.read_text())
        parity_path = Path(manifest["formats"]["coreml"]["parity"]["path"])
        parity = json.loads(parity_path.read_text())
        selection = json.loads(Path(manifest["parity_manifest"]["path"]).read_text())
        annotations = json.loads(Path(
            manifest["dataset"]["annotations"]["val"]["path"]
        ).read_text())
        images = {image["id"]: image for image in annotations["images"]}
        records = []
        for image_id in selection["image_ids"]:
            image = images[image_id]
            source = (
                Path(manifest["dataset"]["directory"])
                / "images/val" / image["file_name"]
            )
            with Image.open(source) as decoded:
                rgb = np.asarray(decoded.convert("RGB"))
            name = f"parity_{image_id}.rgb"
            rgb.tofile(output / name)
            records.append({
                "id": image_id, "width": image["width"], "height": image["height"],
                "rgb": name, "expected": parity["reference_detections"][str(image_id)],
            })
        models.append({
            "model": manifest["model"],
            "package": manifest["formats"]["coreml"]["artifact"]["path"],
            "cases": records,
        })
    if len(models) != 3 or {m["model"] for m in models} != {
        "rfdetr_small", "rfdetr_nano", "yolox_nano"
    }:
        raise ValueError("Require one selected export manifest for each model")
    (output / "parity.json").write_text(json.dumps({
        "opencv": cv2.__version__, "models": models,
    }))
    print(f"Saved checkpoint parity references for {len(models)} models")


if __name__ == "__main__":
    main()
