"""
Export a frozen YOLO11n run to versioned, framework-neutral artifacts.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from detection_evaluation import export_baseline, latest_run_dir, prepare_baseline

from .config import OUTPUT_ROOT
from .ultralytics import configure_privacy

CHECKPOINT = Path("weights/best.pt")
RUN_PATTERN = "yolo11n_*"


def framework_version() -> str:
    """
    Identify the installed standard or headless Ultralytics distribution.
    """
    for distribution in ("ultralytics", "ultralytics-opencv-headless"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    raise PackageNotFoundError("ultralytics or ultralytics-opencv-headless")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse paths without loading a checkpoint or touching a GPU.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--runs-dir", type=Path, default=OUTPUT_ROOT / "runs" / "basketball"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=("val", "test", "both"), default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args(argv)


def export(args: argparse.Namespace) -> list[Path]:
    """
    Validate the run, load its best checkpoint, and export held-out splits.
    """
    run_dir = args.run_dir or latest_run_dir(args.runs_dir, RUN_PATTERN, CHECKPOINT)
    output_dir = args.output_dir or run_dir / "evaluation"
    context = prepare_baseline(
        run_dir,
        args.dataset_dir,
        output_dir,
        checkpoint=CHECKPOINT,
        model_name="yolo11n",
        resolution=args.resolution,
        device=args.device,
        split=args.split,
    )
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU or use --device cpu")
    configure_privacy()
    from ultralytics import YOLO
    from .artifacts import predict_image

    model = YOLO(context.checkpoint)
    if [model.names[index] for index in range(len(model.names))] != list(
        context.class_names
    ):
        raise ValueError(f"Unexpected checkpoint classes: {model.names}")
    inner_model = model.model
    if not isinstance(inner_model, torch.nn.Module):
        raise RuntimeError("Ultralytics did not load a local PyTorch model")
    settings = context.training_settings
    postprocessing = {
        "score_floor": 0.001,
        "nms_iou": float(settings.get("iou", 0.7)),
        "max_det": int(settings.get("max_det", 300)),
        "agnostic_nms": bool(settings.get("agnostic_nms", False)),
        "rect": True,
        "precision": "float32",
        "resize": "Ultralytics native letterbox",
    }

    def predict_one(image):
        """
        Run native Ultralytics prediction with recorded NMS settings.
        """
        return predict_image(
            model,
            image,
            category_ids=list(context.category_ids),
            resolution=context.resolution,
            device=context.device,
            score_floor=postprocessing["score_floor"],
            nms_iou=postprocessing["nms_iou"],
            max_det=postprocessing["max_det"],
            agnostic_nms=postprocessing["agnostic_nms"],
        )

    return export_baseline(
        context,
        predict_one,
        metadata={
            "framework_version": framework_version(),
            "parameters": sum(
                parameter.numel() for parameter in inner_model.parameters()
            ),
            "postprocessing": postprocessing,
        },
        benchmark=args.benchmark,
    )


def main() -> None:
    """
    Run the command-line exporter and print its artifact paths.
    """
    for path in export(parse_args()):
        print(path)


if __name__ == "__main__":
    main()
