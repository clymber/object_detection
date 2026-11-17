"""
Verify RF-DETR export orchestration preserves source runs and fresh model state.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from detection_common.utils.json_io import write_json

from rfdetr_pipeline import export_validation as exports
from rfdetr_pipeline import export_backends as backends


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
    monkeypatch.setattr(exports, "validate_dataset", lambda *a: {})
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

    monkeypatch.setattr(backends.rfdetr, "load_best_model", load)

    def forbidden(*args, **kwargs):
        """
        Reject fitting, publication and attempts from the independent exporter.
        """
        pytest.fail("export must not train or publish source-run artifacts")

    monkeypatch.setattr(backends.rfdetr, "fit_model", forbidden)
    monkeypatch.setattr(backends.rfdetr, "start_training_attempt", forbidden)

    def fake_export(model, directory, resolution):
        """
        Require the export model and the saved input resolution.
        """
        assert model is loaded[0] and resolution == 64
        assert directory.parent == canonical_directory
        assert legacy.read_bytes() == b"previous export"
        path = directory / "model.onnx"
        path.write_bytes(b"onnx")
        return path

    monkeypatch.setattr(backends.rfdetr, "export_onnx_model", fake_export)
    monkeypatch.setattr(backends.rfdetr, "validate_onnx_model", lambda p: p)
    import onnxruntime

    class Session:
        """
        Expose a fixed batch-one RF-DETR session contract without executing it.
        """
        def __init__(self, *args, **kwargs):
            """
            Check that CPU execution is explicitly requested.
            """
            assert len(loaded) == 2
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

    plan = exports.prepare_export(
        run, tmp_path / "coco", model="rfdetr_nano", formats=("onnx",))
    assert not loaded
    artifact = backends.export_onnx(plan, canonical_directory)
    assert "sha256" in artifact and len(loaded) == 1
    monkeypatch.setattr(backends.rfdetr, "export_onnx_model", forbidden)
    reference = backends.load_pytorch(plan, tmp_path / "reference")
    onnx = backends.load_onnx(plan, canonical_directory)
    result = {"reference": reference.contract, "onnx": onnx.contract}
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
        exports.prepare_export(None, tmp_path, model="rfdetr_small",
                                     coreml_precision="int8")
    with pytest.raises(ValueError, match="compute"):
        exports.prepare_export(None, tmp_path, model="rfdetr_small",
                                     compute_units="cuda")


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    """
    Provide minimal saved model metadata without requiring trained model weights.
    """
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoint_best_total.pth").write_bytes(b"selected")
    (run / "checkpoint_best_ema.pth").write_bytes(b"ema")
    write_json(run / "resolved_config.json", {
        "model_config": {"resolution": 64, "num_select": 7}})
    write_json(run / "dataset_manifest.json", {"category_mapping": {
        "class_names": ["ball"], "prediction_to_source": {"0": 19}}})
    protocol = {"training_settings": {"resolution": 64}, "dataset_identity": {
        "canonical": {"canonical_source": {
            "category_mapping": [{"id": 19, "name": "ball"}]}}}}
    monkeypatch.setattr(exports, "select_run", lambda *a: run)
    monkeypatch.setattr(exports, "validate_run", lambda *a: protocol)
    monkeypatch.setattr(exports, "validate_dataset", lambda *a: {})
    return run


def test_preparation_reads_metadata_without_loading_or_writing(saved_run, tmp_path, monkeypatch):
    """
    Verify preparation has no detector, conversion or filesystem side effects.
    """
    before = {p: p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        """
        Reject any attempt to initialize model state during preparation.
        """
        pytest.fail("preparation must not load or export a detector")

    monkeypatch.setattr(backends.rfdetr, "load_best_model", forbidden)
    plan = exports.prepare_export(saved_run, tmp_path / "coco", model="rfdetr_small")
    assert plan.configuration["postprocessing"]["num_select"] == 7
    assert plan.configuration["category_ids"] == [19]
    assert before == {p: p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", ["resolution", "category", "class_name"])
def test_preparation_rejects_saved_metadata_mismatch(saved_run, tmp_path, change):
    """
    Reject source identity mismatches before model loading or artifact publication.
    """
    if change == "resolution":
        write_json(saved_run / "resolved_config.json", {
            "model_config": {"resolution": 32, "num_select": 7}})
    else:
        write_json(saved_run / "dataset_manifest.json", {"category_mapping": {
            "class_names": ["wrong" if change == "class_name" else "ball"],
            "prediction_to_source": {"0": 1 if change == "category" else 19}}})
    with pytest.raises(ValueError, match="differs|differ"):
        exports.prepare_export(saved_run, tmp_path / "coco", model="rfdetr_nano")


@pytest.mark.parametrize("loaded_resolution,loaded_count", [(32, 7), (64, 8)])
def test_loaded_checkpoint_must_match_prepared_configuration(
        saved_run, tmp_path, monkeypatch, loaded_resolution, loaded_count):
    """
    Verify fresh models retain the trained input size and saved selection count.
    """
    plan = exports.prepare_export(saved_run, tmp_path / "coco", model="rfdetr_nano")
    detector = SimpleNamespace(model_config=SimpleNamespace(
        resolution=loaded_resolution, num_select=loaded_count))
    monkeypatch.setattr(backends.rfdetr, "load_best_model", lambda *a, **k: (detector, {}))
    with pytest.raises(ValueError, match="resolution|num_select"):
        backends.load_pytorch(plan, tmp_path)


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:1"])
def test_reference_preserves_native_prediction_and_synchronization(
        saved_run, tmp_path, monkeypatch, device):
    """
    Keep native category mapping, inference mode and selected device synchronization.
    """
    import torch

    plan = exports.prepare_export(
        saved_run, tmp_path / "coco", model="rfdetr_small", device=device)
    detector = SimpleNamespace(model_config=SimpleNamespace(resolution=64, num_select=7))

    def load(path, **kwargs):
        """
        Require explicit device selection and read-only metadata loading.
        """
        assert kwargs == {"device": device, "write_metadata": False}
        return detector, {"selected_weights": "ema", "source_checkpoint": str(
            saved_run / "checkpoint_best_ema.pth")}

    def predictions(model, image, image_id, category_ids):
        """
        Assert the native postprocessor receives the original class identifiers.
        """
        assert torch.is_inference_mode_enabled()
        assert model is detector and category_ids == [19]
        return [{"image_id": image_id, "category_id": 19, "score": .9}]

    monkeypatch.setattr(backends.rfdetr, "load_best_model", load)
    monkeypatch.setattr(backends.rfdetr, "predictions_for_image", predictions)
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: synchronized.append(a))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: synchronized.append(()))
    adapter = backends.load_pytorch(plan, tmp_path)
    assert adapter.predict(None) == [{"category_id": 19, "score": .9}]
    adapter.synchronize()
    assert synchronized == ([] if device == "cpu" else [() if device == "mps" else (device,)])


def test_coreml_conversion_and_spec_order_are_preserved(saved_run, tmp_path, monkeypatch):
    """
    Preserve fixed-shape conversion, FP16 input casting and spec-position outputs.
    """
    import sys
    import numpy as np

    plan = exports.prepare_export(saved_run, tmp_path / "coco", model="rfdetr_small")

    def convert(**kwargs):
        """
        Materialize a fake package while checking the original conversion settings.
        """
        assert kwargs["format"] == "coreml"
        assert kwargs["coreml_precision"] == "float16"
        assert kwargs["batch_size"] == 1 and not kwargs["dynamic_batch"]
        assert kwargs["shape"] == (64, 64)
        path = Path(kwargs["output_dir"]) / "model.mlpackage"
        path.mkdir()
        (path / "model.bin").write_bytes(b"coreml")
        return path

    detector = SimpleNamespace(
        model_config=SimpleNamespace(resolution=64, num_select=7), export=convert)
    monkeypatch.setattr(backends.rfdetr, "load_best_model", lambda *a, **k: (detector, {}))
    directory = tmp_path / "coreml"
    backends.export_coreml(plan, directory)
    array = SimpleNamespace(shape=[1, 3, 64, 64], dataType=65552)
    inputs = [SimpleNamespace(name="input", type=SimpleNamespace(multiArrayType=array))]
    outputs = [SimpleNamespace(name=name, type=SimpleNamespace(multiArrayType=array))
               for name in ("z_boxes", "a_logits")]
    spec = SimpleNamespace(description=SimpleNamespace(input=inputs, output=outputs))

    def predict(values):
        """
        Check dtype conversion and return names whose alphabetical order is wrong.
        """
        assert values["input"].dtype == np.float16
        return {"a_logits": "logits", "z_boxes": "boxes"}

    module = SimpleNamespace(ComputeUnit=SimpleNamespace(ALL="all"), models=SimpleNamespace(
        MLModel=lambda *a, **k: SimpleNamespace(get_spec=lambda: spec, predict=predict)))
    monkeypatch.setitem(sys.modules, "coremltools", module)
    monkeypatch.setattr(backends, "raw_predictor", lambda forward, **k: forward)
    adapter = backends.load_coreml(plan, directory)
    assert adapter.predict(np.zeros((1, 3, 64, 64))) == ["boxes", "logits"]
