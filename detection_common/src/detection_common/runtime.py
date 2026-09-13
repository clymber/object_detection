"""
Small runtime helpers shared by notebook-owning projects.
"""

from __future__ import annotations

import os
import sys
from enum import StrEnum
from pathlib import Path

from .utils.text_stream import set_text_stream_filter


class Device(StrEnum):
    """
    Supported compute device names.
    """

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

    @staticmethod
    def auto_choose() -> Device:
        """
        Prefer CUDA, then available local MPS, then CPU.
        """
        import torch

        if torch.cuda.is_available():
            return Device.CUDA
        if torch.backends.mps.is_available():
            return Device.MPS
        return Device.CPU


def configure_stdio_relative_path(base: Path) -> None:
    """
    Display paths beneath a selected workspace base relatively in both streams.
    """
    substitution = {
        f"{Path(base).resolve()}{os.sep}": "",
        f"{Path.home().resolve()}{os.sep}": f"~{os.sep}",
    }
    sys.stdout = set_text_stream_filter(sys.stdout, map=substitution)
    sys.stderr = set_text_stream_filter(sys.stderr, map=substitution)
