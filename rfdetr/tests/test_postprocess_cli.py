"""
Tests for the RF-DETR post-training recovery command.
"""

from __future__ import annotations

from pathlib import Path

from rfdetr_pipeline import postprocess_cli


def test_main_recovers_the_requested_run(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """
    Delegate recovery to the one model-owned postprocessing helper.
    """
    run_dir = tmp_path / "run"
    calls = []

    def fake_postprocess(path: Path) -> dict:
        """
        Record the requested run and return a compact publication result.
        """
        calls.append(path)
        return {"bundle": {"generation": "fixture-generation"}}

    monkeypatch.setattr(postprocess_cli.rfdetr, "postprocess_run", fake_postprocess)

    assert postprocess_cli.main(["--run-dir", str(run_dir)]) == 0
    assert calls == [run_dir]
    assert capsys.readouterr().out.strip() == "Published bundle: fixture-generation"