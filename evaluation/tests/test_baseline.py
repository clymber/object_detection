"""
Exercise the framework-neutral half of the split baseline exporters.
"""

from pathlib import Path

import pytest
import yaml
from detection_common.utils.json_io import read_json, write_json
from PIL import Image

from detection_evaluation import (
    export_baseline,
    latest_run_dir,
    prepare_baseline,
    read_prediction_artifact,
)


@pytest.fixture
def baseline_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """
    Make a tiny complete training run and frozen train/val/test COCO splits.
    """
    run_dir = tmp_path / "runs" / "yolo11n_basketball_large_dataset_1"
    checkpoint = run_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"selected checkpoint")
    (run_dir / "results.csv").write_text("epoch,metric\n1,0.5\n2,0.7\n")
    dataset_dir = tmp_path / "coco_basketball_large_dataset"
    (run_dir / "args.yaml").write_text(
        yaml.safe_dump({"data": str(dataset_dir), "fraction": 1.0})
    )
    annotation_dir = dataset_dir / "annotations"
    annotation_dir.mkdir(parents=True)
    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        image_dir.mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(image_dir / "example.png")
        write_json(
            annotation_dir / f"instances_{split}.json",
            {
                "images": [
                    {"id": 10, "file_name": "example.png", "width": 64, "height": 64}
                ],
                "categories": [{"id": 7, "name": "basketball"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 10,
                        "category_id": 7,
                        "bbox": [2, 3, 10, 12],
                        "area": 120,
                        "iscrowd": 0,
                    }
                ],
            },
        )
    return run_dir, dataset_dir, tmp_path / "artifacts"


def test_prepare_and_export_without_detector_frameworks(
    baseline_files: tuple[Path, Path, Path],
) -> None:
    """
    Consume one model callback and write complete split artifacts and metrics.
    """
    run_dir, dataset_dir, output_dir = baseline_files
    assert (
        latest_run_dir(run_dir.parent, "yolo11n_*", Path("weights/best.pt"))
        == run_dir
    )
    context = prepare_baseline(
        run_dir,
        dataset_dir,
        output_dir,
        checkpoint=Path("weights/best.pt"),
        model_name="yolo11n",
        resolution=640,
        device="cpu",
    )
    assert context.category_id == 7
    assert context.epochs_completed == 2

    def predict_one(image: Image.Image) -> list[dict]:
        """
        Return one original-pixel COCO box for every fixture image.
        """
        assert image.size == (64, 64)
        return [{"category_id": 7, "bbox": [2, 3, 10, 12], "score": 0.9}]

    paths = export_baseline(
        context,
        predict_one,
        metadata={"postprocessing": {"score_floor": 0.001}},
    )
    assert len(paths) == 2
    for split, path in zip(("val", "test"), paths, strict=True):
        annotation_path = dataset_dir / "annotations" / f"instances_{split}.json"
        artifact = read_prediction_artifact(path, annotation_path)
        assert artifact["image_ids"] == [10]
        assert artifact["metadata"]["checkpoint"] == str(context.checkpoint)
        assert read_json(output_dir / split / f"yolo11n_{split}_metrics.json")[
            "ap50"
        ] == pytest.approx(1)
    with pytest.raises(FileExistsError, match="new output directory"):
        prepare_baseline(
            run_dir,
            dataset_dir,
            output_dir,
            checkpoint=Path("weights/best.pt"),
            model_name="yolo11n",
            resolution=640,
            device="cpu",
        )


def test_prepare_rejects_truncated_run(
    baseline_files: tuple[Path, Path, Path],
) -> None:
    """
    Exclude partial training from full-dataset comparisons.
    """
    run_dir, dataset_dir, output_dir = baseline_files
    (run_dir / "args.yaml").write_text(
        yaml.safe_dump({"data": str(dataset_dir), "fraction": 0.5})
    )
    with pytest.raises(ValueError, match="Truncated"):
        prepare_baseline(
            run_dir,
            dataset_dir,
            output_dir,
            checkpoint=Path("weights/best.pt"),
            model_name="yolo11n",
            resolution=640,
            device="cpu",
        )


def test_multiclass_baseline_preserves_sorted_category_mapping(baseline_files) -> None:
    """Accept multiple classes regardless of category list order across splits."""
    run_dir, dataset_dir, output_dir = baseline_files
    for split in ("train", "val", "test"):
        path = dataset_dir / "annotations" / f"instances_{split}.json"
        document = read_json(path)
        document["categories"].append({"id": 19, "name": "football"})
        if split == "train":
            document["categories"].reverse()
        write_json(path, document)
    context = prepare_baseline(
        run_dir,
        dataset_dir,
        output_dir,
        checkpoint=Path("weights/best.pt"),
        model_name="yolo11n",
        resolution=640,
        device="cpu",
    )
    assert context.category_ids == (7, 19)
    assert context.class_names == ("basketball", "football")
