"""
Verify saved YOLOX settings and native Core ML conversion arguments.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
import yaml
from detection_common.utils.json_io import write_json

from yolox_pipeline import export_validation as exports
from yolox_pipeline import export_backends as backends


def test_export_restores_saved_resolution_nms_classes_and_traced_contract(tmp_path, monkeypatch):
    """
    Convert a fresh CPU evaluation model with graph decoding disabled.
    """
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights/best_ckpt.pth").write_bytes(b"selected")
    args = {"image_size": [96, 96], "classes": ["ball"], "nms_threshold": .43}
    (run / "args.yaml").write_text(yaml.safe_dump(args))
    protocol = {"training_settings": {"image_size": 96, "epochs": 7, "seed": 23},
                "dataset_identity": {"canonical": {"canonical_source": {
                    "category_mapping": [{"id": 19, "name": "ball"}]}}}}
    monkeypatch.setattr(exports, "select_run", lambda *a: run)
    monkeypatch.setattr(exports, "validate_run", lambda *a: protocol)
    monkeypatch.setattr(exports, "validate_dataset", lambda *a: {})
    calls = []

    class Model:
        """
        Provide only the inference-state interface required by the converter.
        """
        def __init__(self):
            """
            Start with the native decoding flag enabled.
            """
            self.head = SimpleNamespace(decode_in_inference=True)
            self.evaluated = False

        def eval(self):
            """
            Record evaluation-mode conversion.
            """
            self.evaluated = True
            return self

    def load(exp, checkpoint, device):
        """
        Verify model reconstruction uses the saved classes, shape and NMS.
        """
        assert str(device) == "cpu"
        assert exp.test_size == (96, 96) and exp.nmsthre == .43
        assert exp.num_classes == 1
        model = Model()
        calls.append(model)
        return model

    monkeypatch.setattr(backends.yolox, "load_trained_model", load)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"ema_model": {"weights": 1}})

    def trace(model, tensor):
        """
        Verify static batch-one tracing uses a separate evaluation model.
        """
        assert model is calls[0] and model.evaluated
        assert not model.head.decode_in_inference
        assert list(tensor.shape) == [1, 3, 96, 96] and tensor.device.type == "cpu"
        return "traced"

    monkeypatch.setattr(torch.jit, "trace", trace)
    array = SimpleNamespace(shape=[1, 3, 96, 96], dataType=65568)
    input_spec = SimpleNamespace(name="images", type=SimpleNamespace(multiArrayType=array))
    output_spec = SimpleNamespace(name="output", type=SimpleNamespace(multiArrayType=array))
    spec = SimpleNamespace(description=SimpleNamespace(input=[input_spec], output=[output_spec]))

    def convert(traced, **kwargs):
        """
        Require FP32 conversion, explicit output precision and fixed Core ML target.
        """
        assert traced == "traced"
        assert kwargs["compute_precision"] == "fp32"
        assert kwargs["compute_units"] == "all"
        assert kwargs["inputs"][0]["shape"] == (1, 3, 96, 96)

        def save(path):
            """
            Materialize a fake package with files for checksum validation.
            """
            from pathlib import Path
            directory = Path(path)
            directory.mkdir()
            (directory / "model.bin").write_bytes(b"coreml")

        return SimpleNamespace(save=save)

    module = SimpleNamespace(
        convert=convert, TensorType=lambda **k: k,
        precision=SimpleNamespace(FLOAT32="fp32"), ComputeUnit=SimpleNamespace(ALL="all"),
        target=SimpleNamespace(iOS16="ios16"),
        models=SimpleNamespace(MLModel=lambda *a, **k: SimpleNamespace(get_spec=lambda: spec)))
    monkeypatch.setitem(sys.modules, "coremltools", module)

    plan = exports.prepare_export(
        run, tmp_path / "dataset", formats=("coreml",))
    assert not calls
    artifact = backends.export_coreml(plan, tmp_path / "package")
    assert "sha256" in artifact and len(calls) == 1

    def forbidden(*args, **kwargs):
        """
        Reject tracing or conversion when loading the already exported graph.
        """
        pytest.fail("validation must not trace or convert the model")

    monkeypatch.setattr(torch.jit, "trace", forbidden)
    monkeypatch.setattr(module, "convert", forbidden)
    reference = backends.load_pytorch(plan, tmp_path / "reference").contract
    converted = backends.load_coreml(plan, tmp_path / "package").contract
    assert reference["selected_weights"] == "ema_model"
    assert converted["precision"] == "float32"
    assert len(calls) == 2 and calls[0] is not calls[1]


def test_export_rejects_precision_substitution(tmp_path):
    """
    Reject unsupported requested precision before resolving or loading a run.
    """
    with pytest.raises(ValueError, match="precision"):
        exports.prepare_export(None, tmp_path, coreml_precision="int8")


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    """
    Provide a saved Nano experiment without loading models or dataset pixels.
    """
    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights/best_ckpt.pth").write_bytes(b"selected")
    (run / "args.yaml").write_text(yaml.safe_dump({
        "image_size": [96, 96], "classes": ["ball"], "nms_threshold": .43}))
    protocol = {"training_settings": {"image_size": 96, "epochs": 7, "seed": 23},
                "dataset_identity": {"canonical": {"canonical_source": {
                    "category_mapping": [{"id": 19, "name": "ball"}]}}}}
    monkeypatch.setattr(exports, "select_run", lambda *a: run)
    monkeypatch.setattr(exports, "validate_run", lambda *a: protocol)
    monkeypatch.setattr(exports, "validate_dataset", lambda *a: {})
    return run


def test_preparation_has_no_model_or_filesystem_side_effects(saved_run, tmp_path, monkeypatch):
    """
    Inspect saved architecture and NMS settings without model initialization.
    """
    before = {p: p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        """
        Reject loading, tracing or constructing an experiment during preparation.
        """
        pytest.fail("preparation must not construct model state")

    monkeypatch.setattr(backends.yolox, "load_trained_model", forbidden)
    monkeypatch.setattr(backends.yolox, "YOLOXNanoExp", forbidden)
    plan = exports.prepare_export(saved_run, tmp_path / "coco")
    assert plan.configuration["postprocessing"]["nms_iou"] == .43
    assert plan.configuration["category_ids"] == [19]
    assert before == {p: p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field,value", [
    ("image_size", [64, 64]), ("classes", ["wrong"]),
    ("nms_threshold", 0), ("nms_threshold", 1.5),
])
def test_preparation_rejects_mismatched_saved_settings(saved_run, tmp_path, field, value):
    """
    Reject changed dimensions, class names and invalid saved NMS thresholds.
    """
    path = saved_run / "args.yaml"
    args = yaml.safe_load(path.read_text())
    args[field] = value
    path.write_text(yaml.safe_dump(args))
    with pytest.raises(ValueError):
        exports.prepare_export(saved_run, tmp_path / "coco")


def test_onnx_export_and_loading_keep_raw_grid_contract(saved_run, tmp_path, monkeypatch):
    """
    Preserve saved dimensions, class mapping, NMS and the explicit CPU runtime policy.
    """
    from pathlib import Path
    import onnxruntime as ort

    plan = exports.prepare_export(saved_run, tmp_path / "coco")
    directory = tmp_path / "onnx"

    def convert(experiment, checkpoint, path):
        """
        Require a saved inference experiment and stage the graph outside the run.
        """
        assert experiment.test_size == (96, 96) and experiment.nmsthre == .43
        assert experiment.num_classes == 1
        assert checkpoint == saved_run / "weights/best_ckpt.pth"
        assert path.parent.parent == directory
        path.write_bytes(b"onnx")
        return path

    monkeypatch.setattr(backends.yolox, "export_trained_model_to_onnx", convert)
    artifact = backends.export_onnx(plan, directory)

    class Session:
        """
        Expose the static batch-one raw-grid contract without real ONNX execution.
        """
        def __init__(self, path, options, **kwargs):
            """
            Verify graph identity and the selected CPU thread/provider policy.
            """
            assert Path(path) == Path(artifact["path"])
            assert options.intra_op_num_threads == 4
            assert kwargs["providers"] == ["CPUExecutionProvider"]

        def get_inputs(self):
            """
            Return fixed trained spatial dimensions.
            """
            return [SimpleNamespace(name="images", shape=[1, 3, 96, 96], type="tensor(float)")]

        def get_outputs(self):
            """
            Return a single raw-grid output tensor.
            """
            return [SimpleNamespace(name="output", shape=[1, 189, 6], type="tensor(float)")]

        def run(self, outputs, inputs):
            """
            Keep runtime outputs raw for the shared YOLOX decoder.
            """
            assert outputs is None and inputs == {"images": "tensor"}
            return ["raw-grid"]

    def predictor(forward, **kwargs):
        """
        Check that the existing decoder receives the saved class and NMS settings.
        """
        assert kwargs == {"family": "yolox", "resolution": 96,
                          "category_ids": [19], "nms_iou": .43}
        return forward

    monkeypatch.setattr(ort, "InferenceSession", Session)
    monkeypatch.setattr(backends, "raw_predictor", predictor)
    adapter = backends.load_onnx(plan, directory)
    assert adapter.predict("tensor") == ["raw-grid"]


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:1"])
def test_reference_device_compatibility_and_sync(saved_run, tmp_path, monkeypatch, device):
    """
    Preserve inference mode, MPS patching, native NMS and accelerator synchronization.
    """
    plan = exports.prepare_export(saved_run, tmp_path / "coco", device=device)
    events = []
    monkeypatch.setattr(backends.yolox, "patch_mps_compatibility", lambda: events.append("mps"))
    monkeypatch.setattr(backends.yolox, "sync_device", lambda d: events.append(str(d)))
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"ema_model": {"weights": 1}})
    monkeypatch.setattr(backends.yolox, "load_trained_model", lambda *a: "model")

    def predict(model, experiment, image, **kwargs):
        """
        Verify native prediction is called under inference mode with source classes.
        """
        assert torch.is_inference_mode_enabled()
        assert model == "model" and experiment.nmsthre == .43
        assert str(kwargs["device"]) == device and kwargs["category_ids"] == [19]
        return []

    monkeypatch.setattr(backends.artifacts, "predict_image", predict)
    adapter = backends.load_pytorch(plan, tmp_path)
    assert adapter.predict(None) == []
    adapter.synchronize()
    assert events == (["mps", device] if device == "mps" else [device])
