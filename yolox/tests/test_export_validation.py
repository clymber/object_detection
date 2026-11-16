"""
Verify saved YOLOX settings and native Core ML conversion arguments.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from yolox_pipeline import export_validation as exports


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

    monkeypatch.setattr(exports.yolox, "load_trained_model", load)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"ema_model": {"weights": 1}})

    def trace(model, tensor):
        """
        Verify static batch-one tracing uses a separate evaluation model.
        """
        assert model is calls[1] and model.evaluated
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
        Require FP16 conversion, explicit output precision and fixed Core ML target.
        """
        assert traced == "traced"
        assert kwargs["compute_precision"] == "fp16"
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
        precision=SimpleNamespace(FLOAT16="fp16"), ComputeUnit=SimpleNamespace(ALL="all"),
        target=SimpleNamespace(iOS16="ios16"),
        models=SimpleNamespace(MLModel=lambda *a, **k: SimpleNamespace(get_spec=lambda: spec)))
    monkeypatch.setitem(sys.modules, "coremltools", module)

    def execute(**kwargs):
        """
        Exercise reference and Core ML factories without training or evaluation.
        """
        reference = kwargs["factory"]("pytorch", tmp_path / "reference")
        converted = kwargs["factory"]("coreml", tmp_path / "package")
        return reference[1], converted[1]

    monkeypatch.setattr(exports, "execute_validation", execute)
    reference, converted = exports.export_and_validate(run, tmp_path / "dataset",
                                                       formats=("coreml",))
    assert reference["selected_weights"] == "ema_model"
    assert converted["precision"] == "float16"
    assert len(calls) == 2 and calls[0] is not calls[1]


def test_export_rejects_precision_substitution(tmp_path):
    """
    Reject unsupported requested precision before resolving or loading a run.
    """
    with pytest.raises(ValueError, match="precision"):
        exports.export_and_validate(None, tmp_path, coreml_precision="int8")
