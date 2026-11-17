"""
YOLOX-owned experiment reconstruction, conversion and fresh runtime loading.
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

from . import artifacts, yolox


def _experiment(plan: ExportPlan):
    """
    Reconstruct only the saved Nano architecture and inference configuration.
    """
    config = plan.configuration
    settings = plan.protocol["training_settings"]
    experiment = yolox.YOLOXNanoExp(
        dataset_dir=plan.dataset_dir,
        output_dir=plan.output_root,
        max_epoch=settings["epochs"],
        image_size=config["resolution"],
        project_name=plan.run_dir.name,
        class_names=tuple(c["name"] for c in config["class_mapping"]),
        seed=settings["seed"],
    )
    experiment.nmsthre = config["postprocessing"]["nms_iou"]
    return experiment


def _threads(plan: ExportPlan) -> None:
    """
    Apply the saved PyTorch CPU thread policy before loading or tracing a model.
    """
    import torch

    torch.set_num_threads(plan.configuration["cpu_threads"])


def export_onnx(plan: ExportPlan, directory: Path) -> dict:
    """
    Export and check a fresh CPU model with raw-grid decoding disabled.
    """
    _threads(plan)
    experiment = _experiment(plan)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".yolox-onnx-", dir=directory) as temporary:
        staged = yolox.export_trained_model_to_onnx(
            experiment,
            plan.run_dir / "weights/best_ckpt.pth",
            Path(temporary) / "model.onnx",
        )
        path = publish_export(staged, directory, plan.run_dir, plan.model, "onnx")
    return artifact_identity(path)


def export_coreml(plan: ExportPlan, directory: Path) -> dict:
    """
    Trace a separate evaluation model into the existing batch-one Core ML contract.
    """
    import coremltools as ct
    import numpy as np
    import torch

    _threads(plan)
    config = plan.configuration
    resolution = config["resolution"]
    model = yolox.load_trained_model(
        _experiment(plan), plan.run_dir / "weights/best_ckpt.pth", torch.device("cpu")
    )
    model.head.decode_in_inference = False
    model.eval()
    with torch.inference_mode():
        traced = torch.jit.trace(model, torch.zeros(1, 3, resolution, resolution))
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        inputs=[
            ct.TensorType(
                name="images", shape=(1, 3, resolution, resolution), dtype=np.float32
            )
        ],
        outputs=[ct.TensorType(name="output", dtype=np.float32)],
        compute_precision=getattr(ct.precision, config["coreml_precision"].upper()),
        compute_units=getattr(ct.ComputeUnit, config["compute_units"].upper()),
    )
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".yolox-coreml-", dir=directory
    ) as temporary:
        staged = Path(temporary) / "model.mlpackage"
        converted.save(str(staged))
        path = publish_export(staged, directory, plan.run_dir, plan.model, "coreml")
    return artifact_identity(path)


def load_pytorch(plan: ExportPlan, directory: Path) -> RuntimeAdapter:
    """
    Load a fresh selected checkpoint with native preprocessing and device handling.
    """
    import torch

    _threads(plan)
    config = plan.configuration
    experiment = _experiment(plan)
    device = torch.device(config["checkpoint_device"])
    if device.type == "mps":
        yolox.patch_mps_compatibility()
    checkpoint = plan.run_dir / "weights/best_ckpt.pth"
    model = yolox.load_trained_model(experiment, checkpoint, device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    selected_weights = "ema_model" if state.get("ema_model") else "model"
    del state

    def predict(image):
        """
        Use the native FP32 reference adapter and saved class-aware CPU NMS.
        """
        with torch.inference_mode():
            return artifacts.predict_image(
                model,
                experiment,
                image,
                device=device,
                category_ids=config["category_ids"],
            )

    def synchronize():
        """
        Complete asynchronous accelerator work at timing boundaries.
        """
        yolox.sync_device(device)

    contract = {
        "backend": "pytorch",
        "device": config["checkpoint_device"],
        "precision": "float32",
        "cpu_threads": config["cpu_threads"],
        "selected_weights": selected_weights,
        "artifact": artifact_identity(checkpoint),
    }
    return RuntimeAdapter(predict, contract, synchronize)


def _raw_runtime(
    plan: ExportPlan, forward, contract: dict, path: Path
) -> RuntimeAdapter:
    """
    Decode raw grids using the saved letterbox, source classes and CPU NMS policy.
    """
    config = plan.configuration
    contract["artifact"] = artifact_identity(path)
    predictor = raw_predictor(
        forward,
        family="yolox",
        resolution=config["resolution"],
        category_ids=config["category_ids"],
        nms_iou=config["postprocessing"]["nms_iou"],
    )
    return RuntimeAdapter(predictor, contract, lambda: None)


def load_onnx(plan: ExportPlan, directory: Path) -> RuntimeAdapter:
    """
    Load a saved raw-grid ONNX graph with the explicit CPU execution policy.
    """
    import onnxruntime as ort

    config = plan.configuration
    path = directory / export_artifact_path(plan.run_dir, plan.model, "onnx").name
    options = ort.SessionOptions()
    options.intra_op_num_threads = config["cpu_threads"]
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (
        inputs[0].shape != [1, 3, config["resolution"], config["resolution"]]
        or len(outputs) != 1
    ):
        raise ValueError("Unexpected YOLOX ONNX input/output contract")

    def forward(tensor):
        """
        Execute synchronous FP32 raw-grid inference.
        """
        return session.run(None, {inputs[0].name: tensor})

    contract = {
        "backend": "onnxruntime",
        "device": "cpu",
        "precision": "float32",
        "cpu_threads": config["cpu_threads"],
        "input": {
            "name": inputs[0].name,
            "shape": inputs[0].shape,
            "dtype": inputs[0].type,
        },
        "outputs": [
            {"name": o.name, "shape": o.shape, "dtype": o.type} for o in outputs
        ],
    }
    return _raw_runtime(plan, forward, contract, path)


def load_coreml(plan: ExportPlan, directory: Path) -> RuntimeAdapter:
    """
    Load an existing raw-grid Core ML package without tracing or conversion.
    """
    import coremltools as ct

    config = plan.configuration
    path = directory / export_artifact_path(plan.run_dir, plan.model, "coreml").name
    runtime = ct.models.MLModel(
        str(path),
        compute_units=getattr(ct.ComputeUnit, config["compute_units"].upper()),
    )
    spec = runtime.get_spec()
    inputs, outputs = list(spec.description.input), list(spec.description.output)

    def forward(tensor):
        """
        Execute synchronous Core ML raw-grid prediction.
        """
        return [runtime.predict({inputs[0].name: tensor})[outputs[0].name]]

    contract = {
        "backend": "coreml",
        "device": "macOS",
        "precision": config["coreml_precision"],
        "compute_units": config["compute_units"],
        "input": {
            "name": inputs[0].name,
            "shape": list(inputs[0].type.multiArrayType.shape),
            "dtype": "float32",
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
    Dispatch conversion explicitly without loading an inference runtime.
    """
    exporters = {"onnx": export_onnx, "coreml": export_coreml}
    if name not in exporters:
        raise ValueError(f"Unsupported YOLOX export: {name}")
    return exporters[name](plan, directory)


def load_runtime(plan: ExportPlan, name: str, directory: Path) -> RuntimeAdapter:
    """
    Dispatch saved-runtime construction without invoking a converter.
    """
    loaders = {"pytorch": load_pytorch, "onnx": load_onnx, "coreml": load_coreml}
    if name not in loaders:
        raise ValueError(f"Unsupported YOLOX runtime: {name}")
    return loaders[name](plan, directory)
