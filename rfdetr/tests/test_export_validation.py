"""
Verify RF-DETR export orchestration preserves source runs and fresh model state.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from detection_common.utils.json_io import write_json

from rfdetr_pipeline import export_validation as exports


def test_export_reference_and_onnx_use_fresh_read_only_models(tmp_path, monkeypatch):
    """
    Load separate FP32 instances and export only into the attempt package directory.
    """
    run = tmp_path / "archived"
    run.mkdir()
    checkpoint = run / "checkpoint_best_total.pth"
    checkpoint.write_bytes(b"selected")
    source = run / "checkpoint_best_ema.pth"
    source.write_bytes(b"ema")
    legacy = run / "checkpoint_best_total.onnx"
    legacy.write_bytes(b"previous export")
    write_json(run / "resolved_config.json", {"model_config": {"resolution": 64, "num_select": 7}})
    write_json(run / "dataset_manifest.json", {"category_mapping": {
        "class_names": ["ball"], "prediction_to_source": {"0": 19}}})
    protocol = {"training_settings": {"resolution": 64}, "dataset_identity": {
        "canonical": {"canonical_source": {"category_mapping": [{"id": 19, "name": "ball"}]}}}}
    monkeypatch.setattr(exports, "select_run", lambda *a: run)
    monkeypatch.setattr(exports, "validate_run", lambda *a: protocol)
    loaded = []
    canonical_directory = tmp_path / "exports/onnx"

    def load(path, **kwargs):
        """
        Create and record a distinct checkpoint model for every factory call.
        """
        assert path == run and kwargs == {"device": "cpu", "write_metadata": False}
        model = SimpleNamespace(model_config=SimpleNamespace(num_select=7, resolution=64))
        loaded.append(model)
        return model, {"selected_weights": "ema", "source_checkpoint": str(source)}

    monkeypatch.setattr(exports.rfdetr, "load_best_model", load)

    def forbidden(*args, **kwargs):
        """
        Reject fitting, publication and attempts from the independent exporter.
        """
        pytest.fail("export must not train or publish source-run artifacts")

    monkeypatch.setattr(exports.rfdetr, "fit_model", forbidden)
    monkeypatch.setattr(exports.rfdetr, "start_training_attempt", forbidden)

    def fake_export(model, directory, resolution):
        """
        Require the second fresh model and the saved input resolution.
        """
        assert model is loaded[1] and resolution == 64
        assert directory.parent == canonical_directory
        assert legacy.read_bytes() == b"previous export"
        path = directory / "model.onnx"
        path.write_bytes(b"onnx")
        return path

    monkeypatch.setattr(exports.rfdetr, "export_onnx_model", fake_export)
    monkeypatch.setattr(exports.rfdetr, "validate_onnx_model", lambda p: p)
    import onnxruntime

    class Session:
        """
        Expose a fixed batch-one RF-DETR session contract without executing it.
        """
        def __init__(self, *args, **kwargs):
            """
            Check that CPU execution is explicitly requested.
            """
            assert kwargs["providers"] == ["CPUExecutionProvider"]

        def get_inputs(self):
            """
            Return the trained fixed input dimensions.
            """
            return [SimpleNamespace(name="input", shape=[1, 3, 64, 64], type="tensor(float)")]

        def get_outputs(self):
            """
            Expose raw boxes/logits in arbitrary session order.
            """
            return [SimpleNamespace(name=name, shape=[1, 7, size], type="tensor(float)")
                    for name, size in (("labels", 2), ("dets", 4))]

    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)

    def execute(**kwargs):
        """
        Exercise each requested factory and return its recorded runtime contract.
        """
        assert kwargs["configuration"]["resolution"] == 64
        assert kwargs["configuration"]["category_ids"] == [19]
        reference = kwargs["factory"]("pytorch", tmp_path / "reference")
        onnx = kwargs["factory"]("onnx", canonical_directory)
        return {"reference": reference[1], "onnx": onnx[1]}

    monkeypatch.setattr(exports, "execute_validation", execute)
    result = exports.export_and_validate(run, tmp_path / "coco", model="rfdetr_nano",
                                         formats=("onnx",))
    assert len(loaded) == 2 and loaded[0] is not loaded[1]
    assert result["reference"]["selected_weights"] == "ema"
    assert result["onnx"]["precision"] == "float32"
    assert not (run / "best_checkpoint.json").exists()
    assert Path(result["onnx"]["artifact"]["path"]) == (
        canonical_directory / "checkpoint_best_total.onnx")
    assert (run / "weights/checkpoint_best_total.onnx").resolve() == (
        canonical_directory / "checkpoint_best_total.onnx")
    assert not legacy.exists()


def test_reject_unsupported_format_settings_before_loading(tmp_path):
    """
    Reject silent precision substitutions and unknown runtime policies.
    """
    with pytest.raises(ValueError, match="precision"):
        exports.export_and_validate(None, tmp_path, model="rfdetr_small",
                                     coreml_precision="int8")
    with pytest.raises(ValueError, match="compute"):
        exports.export_and_validate(None, tmp_path, model="rfdetr_small",
                                     compute_units="cuda")
