"""
Validate the shared Linux GPU runtime and one model project's import.
"""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata

import cv2
import torch
import torchvision
from torchvision.ops import nms

PROJECT_MODULES = {
    "rfdetr": "rfdetr",
    "ultralytics": "ultralytics",
    "yolox": "yolox.layers.fast_cocoeval",
}


def parse_args() -> argparse.Namespace:
    """
    Parse the model project whose import should be validated.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=tuple(PROJECT_MODULES))
    return parser.parse_args()


def validate_gpu_runtime() -> None:
    """
    Exercise CUDA, Torchvision NMS, and headless OpenCV.
    """
    if not torch.version.cuda or not torch.cuda.is_available():
        raise SystemExit("CUDA-enabled PyTorch cannot access the allocated GPU")

    device = torch.device("cuda:0")
    assert torch.tensor([1.0, 2.0], device=device).square().sum().item() == 5.0
    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0]], device=device)
    scores = torch.tensor([0.9], device=device)
    assert nms(boxes, scores, 0.5).numel() == 1

    opencv_names = {
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    }
    installed = {
        distribution.metadata["Name"].lower()
        for distribution in metadata.distributions()
    }
    assert installed & opencv_names == {"opencv-python-headless"}, (
        installed & opencv_names
    )

    gui_backends = [
        line.split(":", 1)[1].strip()
        for line in cv2.getBuildInformation().splitlines()
        if line.strip().startswith("GUI:")
    ]
    assert gui_backends == ["NONE"], gui_backends

    print(
        torch.__version__,
        torchvision.__version__,
        torch.version.cuda,
        torch.cuda.get_device_name(0),
    )


def validate_project_import(project: str) -> None:
    """
    Import the selected model project after the shared runtime checks.
    """
    module = importlib.import_module(PROJECT_MODULES[project])
    if project == "rfdetr" and not hasattr(module, "RFDETRSmall"):
        raise ImportError("rfdetr.RFDETRSmall is unavailable")


def main() -> None:
    """
    Run shared GPU validation and the selected project import.
    """
    args = parse_args()
    validate_gpu_runtime()
    validate_project_import(args.project)


if __name__ == "__main__":
    main()
