"""
Execute export notebook sources with small saved-result fixtures.
"""

from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

import pytest
from detection_common.utils.json_io import write_json

from yolox_pipeline import export_validation as exports


@pytest.mark.parametrize(
    "notebook,model",
    [
        ("nb03.03.02-yolox_nano-basketball-export.py", "yolox_nano"),
    ],
)
@pytest.mark.parametrize("failed", [False, True])
def test_export_notebook_executes_explicit_steps(
    tmp_path, monkeypatch, capsys, notebook, model, failed
):
    """
    Exercise every real presentation cell and retain diagnostic failure completion.
    """
    import detection_common
    import IPython.display

    calls = []
    plan = SimpleNamespace(
        run_dir=tmp_path / "run",
        model=model,
        formats=("onnx",),
        protocol={
            "dataset_identity": {
                "canonical": {
                    "canonical_source": {
                        "category_mapping": [{"id": 19, "name": "ball"}]
                    }
                }
            }
        },
        configuration={
            "resolution": 64,
            "category_ids": [19],
            "checkpoint_device": "cpu",
            "cpu_threads": 4,
            "coreml_precision": "float32",
            "compute_units": "all",
        },
    )
    timing = tmp_path / "host_timing.json"
    write_json(timing, {"mean_ms": 1.0, "p50_ms": 1.0, "p95_ms": 1.0})
    outcome = {
        "status": "failed" if failed else "completed",
        "accepted": not failed,
        "parity_status": "failed" if failed else "passed",
        "stages": {
            stage: {"status": "completed"} for stage in ("parity", "accuracy", "timing")
        },
        "metrics": {
            "val": {
                "ap50": 1.0,
                "ap50_95": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            }
        },
        "timing": {"path": str(timing)},
    }
    result = {
        "manifest_path": str(tmp_path / "manifest.json"),
        "report_dir": str(tmp_path),
        "export_dir": str(tmp_path),
        "exports": {"onnx": {"status": "exported"}},
        "formats": {"onnx": outcome},
        "source_run_unchanged": True,
        "attempt_status": "finished",
    }

    def prepare(*args, **kwargs):
        """
        Record preparation and ensure each notebook selects its intended model.
        """
        assert kwargs.get("model", "yolox_nano") == model
        calls.append("prepare")
        return plan

    def export(prepared):
        """
        Require the prepared plan without running any validation callback.
        """
        assert prepared is plan
        calls.append("export")
        return "selected-export"

    def stage(name):
        """
        Build independent stage fixtures that record their execution order.
        """

        def run(selected):
            """
            Return displayable fixture results for exactly the requested stage.
            """
            assert selected == "selected-export"
            calls.append(name)
            return result

        return run

    monkeypatch.setattr(exports, "prepare_export", prepare)
    monkeypatch.setattr(exports, "export_models", export)
    monkeypatch.setattr(exports, "check_parity", stage("parity"))
    monkeypatch.setattr(exports, "evaluate_accuracy", stage("accuracy"))
    monkeypatch.setattr(exports, "benchmark_host", stage("timing"))
    monkeypatch.setattr(exports, "read_results", lambda selected: result)
    monkeypatch.setattr(
        detection_common, "configure_stdio_relative_path", lambda *a: None
    )
    monkeypatch.setattr(IPython.display, "display", lambda *a: None)
    path = Path(__file__).parents[1] / "notebooks" / notebook
    run_path(str(path))
    assert calls == ["prepare", "export", "parity", "accuracy", "timing"]
    assert ("ERROR:" in capsys.readouterr().out) is failed
