"""
Export a frozen YOLOX Tiny or Nano run to neutral prediction artifacts.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path

from detection_evaluation import export_baseline, latest_run_dir, prepare_baseline

from .config import OUTPUT_ROOT

CHECKPOINT = Path("weights/best_ckpt.pth")
MODEL_RUNS = {
    "tiny": (
        "yolox_tiny",
        "yolox_tiny_*",
        "YOLOXTinyExp",
    ),
    "nano": (
        "yolox_nano",
        "yolox_nano_*",
        "YOLOXNanoExp",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse a model-specific export without loading YOLOX or a checkpoint.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_RUNS), required=True)
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
    Validate the selected run, load YOLOX, and export held-out splits.
    """
    model_name, run_pattern, experiment_name = MODEL_RUNS[args.model]
    run_dir = args.run_dir or latest_run_dir(args.runs_dir, run_pattern, CHECKPOINT)
    output_dir = args.output_dir or run_dir / "evaluation"
    context = prepare_baseline(
        run_dir,
        args.dataset_dir,
        output_dir,
        checkpoint=CHECKPOINT,
        model_name=model_name,
        resolution=args.resolution,
        device=args.device,
        split=args.split,
    )
    import torch

    from . import yolox
    from .artifacts import predict_image

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a GPU or use --device cpu")
    settings = context.training_settings
    if settings.get("classes") != list(context.class_names):
        raise ValueError("YOLOX run classes differ from the dataset")
    experiment = getattr(yolox, experiment_name)(
        dataset_dir=context.dataset_dir,
        output_dir=context.run_dir.parent,
        max_epoch=int(settings["epochs"]),
        image_size=context.resolution,
        project_name=context.run_dir.name,
        class_names=context.class_names,
    )
    experiment.test_conf = 0.001
    experiment.nmsthre = float(settings.get("nms_threshold", experiment.nmsthre))
    device = torch.device(args.device)
    model = yolox.load_trained_model(experiment, context.checkpoint, device)

    def predict_one(image):
        """
        Use the native YOLOX FP32 forward and CPU-NMS path.
        """
        return predict_image(
            model,
            experiment,
            image,
            device=device,
            category_ids=list(context.category_ids),
            score_floor=0.001,
        )

    return export_baseline(
        context,
        predict_one,
        metadata={
            "framework_version": version("yolox"),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "postprocessing": {
                "score_floor": 0.001,
                "nms_iou": experiment.nmsthre,
                "precision": "float32",
                "nms_device": "cpu",
                "resize": "YOLOX native ValTransform letterbox",
            },
        },
        benchmark=args.benchmark,
    )


def main() -> None:
    """
    Run the model-owned exporter and print artifact paths.
    """
    for path in export(parse_args()):
        print(path)


if __name__ == "__main__":
    main()
