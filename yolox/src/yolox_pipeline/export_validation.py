"""
Read-only YOLOX Nano export and validation from completed run artifacts.
"""

from __future__ import annotations

import platform
import tempfile
from pathlib import Path

import yaml
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

from . import artifacts, yolox
from .config import OUTPUT_ROOT, WORKSPACE_ROOT


def export_and_validate(run_dir: Path | None, dataset_dir: Path, *,
                        runs_dir: Path | None = None,
                        formats: tuple[str, ...] | None = None,
                        device: str = "cpu", cpu_threads: int = 4,
                        coreml_precision: str = "float16",
                        compute_units: str = "all") -> dict:
    """
    Reload Nano's selected EMA/regular state without training initialization.
    """
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be a positive integer")
    dataset_dir = workspace_path(dataset_dir)
    formats = tuple(formats if formats is not None else (
        ("onnx", "coreml") if platform.system() == "Darwin" else ("onnx",)))
    validate_format_settings(formats, coreml_precision, compute_units)
    run = select_run(runs_dir or OUTPUT_ROOT / "runs/basketball", "yolox_nano", run_dir)
    protocol = validate_run(run, "yolox_nano")
    args = yaml.safe_load((run / "args.yaml").read_text())
    resolution = protocol["training_settings"]["image_size"]
    if args["image_size"] != [resolution, resolution]:
        raise ValueError("Saved YOLOX input dimensions differ from protocol")
    mapping = protocol["dataset_identity"]["canonical"]["canonical_source"]["category_mapping"]
    categories = [c["id"] for c in mapping]
    names = tuple(c["name"] for c in mapping)
    if list(names) != args["classes"]:
        raise ValueError("Saved YOLOX class names differ from source categories")
    nms = float(args["nms_threshold"])
    if not 0 < nms <= 1:
        raise ValueError("Invalid saved YOLOX NMS threshold")
    checkpoint = run / "weights/best_ckpt.pth"
    print(f"Run: {run}\nCheckpoint: {checkpoint}", flush=True)
    configuration = {"resolution": resolution, "category_ids": categories,
                     "class_mapping": mapping, "checkpoint_device": device, "cpu_threads": cpu_threads,
                     "coreml_precision": coreml_precision, "compute_units": compute_units,
                     "preprocessing": "RGB-to-BGR;top-left bilinear letterbox;padding=114;float32 0..255",
                     "postprocessing": {"score_floor": .001, "nms_iou": nms,
                                        "nms": "class-aware CPU", "graph_decoding": False,
                                        "boxes": "unclipped source xywh"}}

    def experiment():
        """
        Construct only the saved model architecture and inference configuration.
        """
        exp = yolox.YOLOXNanoExp(
            dataset_dir=Path(dataset_dir), output_dir=OUTPUT_ROOT,
            max_epoch=protocol["training_settings"]["epochs"], image_size=resolution,
            project_name=run.name, class_names=names,
            seed=protocol["training_settings"]["seed"])
        exp.nmsthre = nms
        return exp

    def factory(name: str, directory: Path) -> tuple:
        """
        Build separate eager, ONNX-export and traced Core ML model instances.
        """
        import torch

        torch.set_num_threads(cpu_threads)

        exp = experiment()
        if name == "pytorch":
            selected_device = torch.device(device)
            if device == "mps":
                yolox.patch_mps_compatibility()
            model = yolox.load_trained_model(exp, checkpoint, selected_device)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            selected_weights = "ema_model" if state.get("ema_model") else "model"
            del state

            def predict(image):
                """
                Execute the native reference adapter with CPU NMS and score floor.
                """
                with torch.inference_mode():
                    return artifacts.predict_image(model, exp, image,
                                                    device=selected_device,
                                                    category_ids=categories)

            def synchronize():
                """
                Complete asynchronous accelerator work at timing boundaries.
                """
                yolox.sync_device(selected_device)

            return predict, {"backend": "pytorch", "device": device,
                             "precision": "float32", "cpu_threads": cpu_threads, "selected_weights": selected_weights,
                             "artifact": artifact_identity(checkpoint)}, synchronize
        directory.mkdir(parents=True, exist_ok=True)
        if name == "onnx":
            import onnxruntime as ort

            with tempfile.TemporaryDirectory(prefix=".yolox-onnx-", dir=directory) as temporary:
                staged = yolox.export_trained_model_to_onnx(
                    exp, checkpoint, Path(temporary) / "model.onnx")
                path = publish_export(staged, directory, run, "yolox_nano", name)
            options = ort.SessionOptions()
            options.intra_op_num_threads = cpu_threads
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if inputs[0].shape != [1, 3, resolution, resolution] or len(outputs) != 1:
                raise ValueError("Unexpected YOLOX ONNX input/output contract")

            def forward(tensor):
                """
                Execute synchronous FP32 CPU ONNX raw grid prediction.
                """
                return session.run(None, {inputs[0].name: tensor})

            contract = {"backend": "onnxruntime", "device": "cpu", "precision": "float32",
                        "input": {"name": inputs[0].name, "shape": inputs[0].shape,
                                  "dtype": inputs[0].type},
                        "outputs": [{"name": o.name, "shape": o.shape, "dtype": o.type}
                                    for o in outputs]}
        else:
            import coremltools as ct
            import numpy as np

            model = yolox.load_trained_model(exp, checkpoint, torch.device("cpu"))
            model.head.decode_in_inference = False
            model.eval()
            with torch.inference_mode():
                traced = torch.jit.trace(model, torch.zeros(1, 3, resolution, resolution))
            converted = ct.convert(
                traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS16,
                inputs=[ct.TensorType(name="images", shape=(1, 3, resolution, resolution),
                                      dtype=np.float32)],
                outputs=[ct.TensorType(name="output", dtype=np.float32)],
                compute_precision=getattr(ct.precision, coreml_precision.upper()),
                compute_units=getattr(ct.ComputeUnit, compute_units.upper()))
            with tempfile.TemporaryDirectory(prefix=".yolox-coreml-", dir=directory) as temporary:
                staged = Path(temporary) / "model.mlpackage"
                converted.save(str(staged))
                path = publish_export(staged, directory, run, "yolox_nano", name)
            del model, traced, converted
            runtime = ct.models.MLModel(str(path), compute_units=getattr(
                ct.ComputeUnit, compute_units.upper()))
            spec = runtime.get_spec()
            inputs, outputs = list(spec.description.input), list(spec.description.output)

            def forward(tensor):
                """
                Execute synchronous Core ML raw grid prediction.
                """
                return [runtime.predict({inputs[0].name: tensor})[outputs[0].name]]

            contract = {"backend": "coreml", "device": "macOS", "precision": coreml_precision,
                        "compute_units": compute_units,
                        "input": {"name": inputs[0].name, "shape": list(inputs[0].type.multiArrayType.shape),
                                  "dtype": "float32"},
                        "outputs": [{"name": o.name, "shape": list(o.type.multiArrayType.shape),
                                     "dtype": o.type.multiArrayType.dataType} for o in outputs]}
        if name == "onnx":
            contract["cpu_threads"] = cpu_threads
        contract["artifact"] = artifact_identity(path)
        return raw_predictor(forward, family="yolox", resolution=resolution,
                             category_ids=categories, nms_iou=nms), contract, lambda: None

    return execute_validation(run_dir=run, dataset_dir=dataset_dir, model="yolox_nano",
                              formats=formats, output_root=OUTPUT_ROOT,
                              models_root=WORKSPACE_ROOT / "models",
                              configuration=configuration, factory=factory)
