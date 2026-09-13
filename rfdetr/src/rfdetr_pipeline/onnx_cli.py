#!/usr/bin/env python3
"""
Export and validate an RF-DETR run's selected checkpoint outside IPython.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rfdetr_pipeline import rfdetr


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Require the completed RF-DETR run directory to export.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="RF-DETR run containing checkpoint_best_total.pth",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Ensure the run has a structurally valid, atomically installed ONNX model.
    """
    args = parse_args(argv)
    onnx_path = rfdetr.ensure_onnx_model(args.run_dir)
    print(f"ONNX model: {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
