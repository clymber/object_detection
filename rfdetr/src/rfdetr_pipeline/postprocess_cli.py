"""
Regenerate and publish outputs for a completed new-protocol RF-DETR run.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from . import rfdetr


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Require the completed RF-DETR run whose artifacts will be republished.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="completed RF-DETR run containing run_protocol.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Recover model-owned outputs without invoking fitting or appending attempts.
    """
    result = rfdetr.postprocess_run(parse_args(argv).run_dir)
    print(f"Published bundle: {result['bundle']['generation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())