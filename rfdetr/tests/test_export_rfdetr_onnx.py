"""
Tests for the standalone RF-DETR ONNX export worker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfdetr_pipeline import onnx_cli as export_rfdetr_onnx


def test_main_ensures_and_reports_the_selected_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Resolve the CLI run directory and report its validated ONNX artifact.
    """
    run_dir = tmp_path / "run"
    expected = run_dir / "checkpoint_best_total.onnx"
    calls = []

    def fake_ensure(path: Path) -> Path:
        """
        Record the requested run and return its fixture artifact path.
        """
        calls.append(path)
        return expected

    monkeypatch.setattr(export_rfdetr_onnx.rfdetr, "ensure_onnx_model", fake_ensure)

    assert export_rfdetr_onnx.main(["--run-dir", str(run_dir)]) == 0
    assert calls == [run_dir]
    assert capsys.readouterr().out.strip() == f"ONNX model: {expected}"
