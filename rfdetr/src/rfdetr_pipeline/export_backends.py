"""
RF-DETR-owned model conversion and fresh runtime construction.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from detection_evaluation.runtime_workflow import (
    ExportPlan,
    RuntimeAdapter,
    artifact_identity,
    export_artifact_path,
    publish_export,
    raw_predictor,
)

from . import rfdetr


def _checkpoint(plan: ExportPlan, device: str) -> tuple:
    """
    Load a fresh FP32 detector and verify the saved shape and selection count.
    """
    import torch

    config = plan.configuration
    torch.set_num_threads(config["cpu_threads"])
    detector, metadata = rfdetr.load_best_model(
        plan.run_dir, device=device, write_metadata=False
    )
    if detector.model_config.resolution != config["resolution"]:
        raise ValueError("Loaded model resolution differs from run protocol")
    if detector.model_config.num_select != config["postprocessing"]["num_select"]:
        raise ValueError("Loaded num_select differs from saved model configuration")
    return detector, metadata


def export_onnx(plan: ExportPlan, directory: Path) -> dict:
    """
    Convert and check a fresh CPU model before publishing its ONNX artifact.
    """
    detector, _ = _checkpoint(plan, "cpu")
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rfdetr-onnx-", dir=directory
    ) as temporary:
        staged = rfdetr.export_onnx_model(
            detector, Path(temporary), plan.configuration["resolution"]
        )
        rfdetr.validate_onnx_model(staged)
        path = publish_export(staged, directory, plan.run_dir, plan.model, "onnx")
    return artifact_identity(path)


def export_coreml(plan: ExportPlan, directory: Path) -> dict:
    """
    Convert a separate CPU detector into a fixed batch-one Core ML package.
    """
    detector, _ = _checkpoint(plan, "cpu")
    config = plan.configuration
    resolution = config["resolution"]
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rfdetr-coreml-", dir=directory
    ) as temporary:
        staged = Path(
            detector.export(
                format="coreml",
                coreml_precision=config["coreml_precision"],
                batch_size=1,
                dynamic_batch=False,
                shape=(resolution, resolution),
                output_dir=temporary,
                output_name="model",
                verbose=False,
            )
        )
        path = publish_export(staged, directory, plan.run_dir, plan.model, "coreml")
    return artifact_identity(path)


def load_pytorch(plan: ExportPlan, directory: Path) -> RuntimeAdapter:
    """
    Construct a native FP32 reference and device synchronization callback.
    """
    import torch

    config = plan.configuration
    device = config["checkpoint_device"]
    detector, metadata = _checkpoint(plan, device)

    def predict(image):
        """
        Run the native checkpoint postprocessor without autograd.
        """
        with torch.inference_mode():
            rows = rfdetr.predictions_for_image(
                detector, image, 0, config["category_ids"]
            )
        return [{k: v for k, v in row.items() if k != "image_id"} for row in rows]

    def synchronize():
        """
        Wait for accelerator work before ending a host timing sample.
        """
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elif device == "mps":
            torch.mps.synchronize()

    contract = {
        "backend": "pytorch",
        "device": device,
        "precision": "float32",
        "cpu_threads": config["cpu_threads"],
        "artifact": artifact_identity(plan.run_dir / rfdetr.BEST_CHECKPOINT),
        "selected_weights": metadata["selected_weights"],
        "source_checkpoint": artifact_identity(Path(metadata["source_checkpoint"])),
    }
    return RuntimeAdapter(predict, contract, synchronize)


def _raw_runtime(
    plan: ExportPlan, forward, contract: dict, path: Path
) -> RuntimeAdapter:
    """
    Apply RF-DETR's existing common raw-output preprocessing and postprocessing.
    """
    config = plan.configuration
    contract["artifact"] = artifact_identity(path)
    predictor = raw_predictor(
        forward,
        family="rfdetr",
        resolution=config["resolution"],
        category_ids=config["category_ids"],
        num_select=config["postprocessing"]["num_select"],
    )
    return RuntimeAdapter(predictor, contract, lambda: None)


def load_onnx(plan: ExportPlan, directory: Path) -> RuntimeAdapter:
    """
    Load a saved fixed-shape ONNX graph with synchronous CPU execution.
    """
    import onnxruntime as ort

    config = plan.configuration
    path = directory / export_artifact_path(plan.run_dir, plan.model, "onnx").name
    options = ort.SessionOptions()
    options.intra_op_num_threads = config["cpu_threads"]
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    input_info = session.get_inputs()[0]
    if input_info.shape != [1, 3, config["resolution"], config["resolution"]]:
        raise ValueError("ONNX input differs from fixed trained dimensions")
    outputs = session.get_outputs()
    if {o.name for o in outputs} != {"dets", "labels"}:
        raise ValueError(f"Unexpected RF-DETR outputs: {[o.name for o in outputs]}")

    def forward(tensor):
        """
        Return raw boxes and logits in their required contract order.
        """
        return session.run(["dets", "labels"], {input_info.name: tensor})

    contract = {
        "backend": "onnxruntime",
        "device": "cpu",
        "precision": "float32",
        "cpu_threads": config["cpu_threads"],
        "input": {
            "name": input_info.name,
            "shape": input_info.shape,
            "dtype": input_info.type,
        },
        "outputs": [
            {"name": o.name, "shape": o.shape, "dtype": o.type} for o in outputs
        ],
    }
    return _raw_runtime(plan, forward, contract, path)


def load_coreml(plan: ExportPlan, directory: Path) -> RuntimeAdapter:
    """
    Load a saved Core ML package using specification order and input precision.
    """
    import coremltools as ct
    import numpy as np

    config = plan.configuration
    path = directory / export_artifact_path(plan.run_dir, plan.model, "coreml").name
    runtime = ct.models.MLModel(
        str(path),
        compute_units=getattr(ct.ComputeUnit, config["compute_units"].upper()),
    )
    spec = runtime.get_spec()
    inputs, outputs = list(spec.description.input), list(spec.description.output)
    if len(inputs) != 1 or len(outputs) != 2:
        raise ValueError("Unexpected Core ML input/output count")
    input_spec = inputs[0].type.multiArrayType
    if list(input_spec.shape) != [1, 3, config["resolution"], config["resolution"]]:
        raise ValueError("Core ML input differs from fixed trained dimensions")
    dtype = np.float16 if input_spec.dataType == 65552 else np.float32

    def forward(tensor):
        """
        Preserve the exporter's positional boxes/logits convention.
        """
        values = runtime.predict({inputs[0].name: tensor.astype(dtype)})
        return [values[o.name] for o in outputs]

    contract = {
        "backend": "coreml",
        "device": "macOS",
        "precision": config["coreml_precision"],
        "compute_units": config["compute_units"],
        "input": {
            "name": inputs[0].name,
            "shape": list(input_spec.shape),
            "dtype": str(np.dtype(dtype)),
        },
        "outputs": [
            {
                "name": o.name,
                "shape": list(o.type.multiArrayType.shape),
                "dtype": o.type.multiArrayType.dataType,
            }
            for o in outputs
        ],
    }
    return _raw_runtime(plan, forward, contract, path)


def export_model(plan: ExportPlan, name: str, directory: Path) -> dict:
    """
    Dispatch conversion explicitly without constructing an inference runtime.
    """
    exporters = {"onnx": export_onnx, "coreml": export_coreml}
    if name not in exporters:
        raise ValueError(f"Unsupported RF-DETR export: {name}")
    return exporters[name](plan, directory)


def load_runtime(plan: ExportPlan, name: str, directory: Path) -> RuntimeAdapter:
    """
    Dispatch saved-runtime loading without calling any exporter.
    """
    loaders = {"pytorch": load_pytorch, "onnx": load_onnx, "coreml": load_coreml}
    if name not in loaders:
        raise ValueError(f"Unsupported RF-DETR runtime: {name}")
    return loaders[name](plan, directory)
