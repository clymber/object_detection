"""
Independent, read-only RF-DETR checkpoint export and runtime validation.
"""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path

from detection_common.utils.json_io import read_json
from detection_evaluation.runtime_workflow import (
    artifact_identity,
    execute_validation,
    publish_export,
    raw_predictor,
    select_run,
    validate_format_settings,
    validate_run,
    workspace_path,
)

from . import rfdetr
from .config import OUTPUT_ROOT, WORKSPACE_ROOT


def export_and_validate(run_dir: Path | None, dataset_dir: Path, *,
                        model: str, runs_dir: Path | None = None,
                        formats: tuple[str, ...] | None = None,
                        device: str = "cpu", cpu_threads: int = 4,
                        coreml_precision: str = "float16",
                        compute_units: str = "all") -> dict:
    """
    Export a completed Small/Nano run through fresh instances for every runtime.
    """
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    dataset_dir = workspace_path(dataset_dir)
    formats = tuple(formats if formats is not None else (
        ("onnx", "coreml") if platform.system() == "Darwin" else ("onnx",)))
    validate_format_settings(formats, coreml_precision, compute_units)
    if model not in {"rfdetr_small", "rfdetr_nano"}:
        raise ValueError(f"Expected RF-DETR Small/Nano, got {model}")
    run = select_run(runs_dir or OUTPUT_ROOT / "runs/basketball", model, run_dir)
    protocol = validate_run(run, model)
    saved = read_json(run / "resolved_config.json")["model_config"]
    resolution = protocol["training_settings"]["resolution"]
    if saved["resolution"] != resolution:
        raise ValueError("Saved model resolution differs from run protocol")
    mapping = read_json(run / "dataset_manifest.json")["category_mapping"]
    categories = [mapping["prediction_to_source"][str(i)]
                  for i in range(len(mapping["class_names"]))]
    if categories != [c["id"] for c in protocol["dataset_identity"]["canonical"][
            "canonical_source"]["category_mapping"]]:
        raise ValueError("Saved RF-DETR category mapping differs from source identity")
    source_categories = protocol["dataset_identity"]["canonical"]["canonical_source"]["category_mapping"]
    if mapping["class_names"] != [category["name"] for category in source_categories]:
        raise ValueError("Saved RF-DETR class names differ from source identity")
    print(f"Run: {run}\nCheckpoint: {run / rfdetr.BEST_CHECKPOINT}", flush=True)
    configuration = {"resolution": resolution, "category_ids": categories,
                     "class_mapping": mapping, "checkpoint_device": device, "cpu_threads": cpu_threads,
                     "coreml_precision": coreml_precision, "compute_units": compute_units,
                     "preprocessing": "RGB float32/255;half-pixel square bilinear;ImageNet normalization",
                     "postprocessing": {"score_floor": .001, "num_select": saved["num_select"],
                                        "nms": None, "background": "rank then discard last slot",
                                        "boxes": "unclipped source xywh"}}

    def factory(name: str, directory: Path) -> tuple:
        """
        Load a new detector before reference inference or an exporter mutation.
        """
        import torch

        torch.set_num_threads(cpu_threads)

        if device.split(":")[0] not in {"cpu", "mps", "cuda"}:
            raise ValueError("Checkpoint device must be cpu, mps or cuda[:index]")
        detector, metadata = rfdetr.load_best_model(
            run, device=device if name == "pytorch" else "cpu", write_metadata=False)
        if detector.model_config.resolution != resolution:
            raise ValueError("Loaded model resolution differs from run protocol")
        if detector.model_config.num_select != saved["num_select"]:
            raise ValueError("Loaded num_select differs from saved model configuration")
        if name == "pytorch":
            def predict(image):
                """
                Run the native checkpoint postprocessor in FP32 without autograd.
                """
                with torch.inference_mode():
                    rows = rfdetr.predictions_for_image(detector, image, 0, categories)
                return [{k: v for k, v in r.items() if k != "image_id"} for r in rows]

            def synchronize():
                """
                Wait for the selected device before ending a host timing sample.
                """
                if device.startswith("cuda"):
                    torch.cuda.synchronize(device)
                elif device == "mps":
                    torch.mps.synchronize()

            return predict, {"backend": "pytorch", "device": device,
                             "precision": "float32", "cpu_threads": cpu_threads, "artifact": artifact_identity(
                                 run / rfdetr.BEST_CHECKPOINT),
                             "selected_weights": metadata["selected_weights"],
                             "source_checkpoint": artifact_identity(Path(metadata["source_checkpoint"]))}, synchronize
        directory.mkdir(parents=True, exist_ok=True)
        if name == "onnx":
            import onnxruntime as ort

            with tempfile.TemporaryDirectory(prefix=".rfdetr-onnx-", dir=directory) as temporary:
                staged = rfdetr.export_onnx_model(detector, Path(temporary), resolution)
                rfdetr.validate_onnx_model(staged)
                path = publish_export(staged, directory, run, model, name)
            del detector
            options = ort.SessionOptions()
            options.intra_op_num_threads = cpu_threads
            session = ort.InferenceSession(str(path), options,
                                           providers=["CPUExecutionProvider"])
            input_info = session.get_inputs()[0]
            if input_info.shape != [1, 3, resolution, resolution]:
                raise ValueError("ONNX input differs from fixed trained dimensions")
            outputs = [o.name for o in session.get_outputs()]
            if set(outputs) != {"dets", "labels"}:
                raise ValueError(f"Unexpected RF-DETR outputs: {outputs}")

            def forward(tensor):
                """
                Execute synchronous CPU ONNX boxes and logits in contract order.
                """
                return session.run(["dets", "labels"], {input_info.name: tensor})

            contract = {"backend": "onnxruntime", "device": "cpu", "precision": "float32",
                        "input": {"name": input_info.name, "shape": input_info.shape,
                                  "dtype": input_info.type},
                        "outputs": [{"name": o.name, "shape": o.shape, "dtype": o.type}
                                    for o in session.get_outputs()]}
        else:
            import coremltools as ct
            import numpy as np

            with tempfile.TemporaryDirectory(prefix=".rfdetr-coreml-", dir=directory) as temporary:
                staged = Path(detector.export(
                    format="coreml", coreml_precision=coreml_precision,
                    batch_size=1, dynamic_batch=False, shape=(resolution, resolution),
                    output_dir=temporary, output_name="model", verbose=False))
                path = publish_export(staged, directory, run, model, name)
            del detector
            runtime = ct.models.MLModel(str(path), compute_units=getattr(
                ct.ComputeUnit, compute_units.upper()))
            spec = runtime.get_spec()
            inputs, outputs = list(spec.description.input), list(spec.description.output)
            if len(inputs) != 1 or len(outputs) != 2:
                raise ValueError("Unexpected Core ML input/output count")
            input_spec = inputs[0].type.multiArrayType
            if list(input_spec.shape) != [1, 3, resolution, resolution]:
                raise ValueError("Core ML input differs from fixed trained dimensions")
            dtype = np.float16 if input_spec.dataType == 65552 else np.float32

            def forward(tensor):
                """
                Use Core ML spec output position, as RF-DETR's exporter requires.
                """
                values = runtime.predict({inputs[0].name: tensor.astype(dtype)})
                return [values[o.name] for o in outputs]

            contract = {"backend": "coreml", "device": "macOS", "precision": coreml_precision,
                        "compute_units": compute_units,
                        "input": {"name": inputs[0].name, "shape": list(input_spec.shape),
                                  "dtype": str(np.dtype(dtype))},
                        "outputs": [{"name": o.name, "shape": list(o.type.multiArrayType.shape),
                                     "dtype": o.type.multiArrayType.dataType} for o in outputs]}
        if name == "onnx":
            contract["cpu_threads"] = cpu_threads
        contract["artifact"] = artifact_identity(path)
        return raw_predictor(forward, family="rfdetr", resolution=resolution,
                             category_ids=categories, num_select=saved["num_select"]), contract, lambda: None

    return execute_validation(run_dir=run, dataset_dir=dataset_dir, model=model,
                              formats=formats, output_root=OUTPUT_ROOT,
                              models_root=WORKSPACE_ROOT / "models",
                              configuration=configuration, factory=factory)
