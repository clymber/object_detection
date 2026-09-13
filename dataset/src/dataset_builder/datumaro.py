"""
Datumaro dataset utilities.
"""

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from io import IOBase
from pathlib import Path

import datumaro as dm
import pandas as pd
from datumaro.components import media as datumaro_media


class ImportFormat(StrEnum):
    """
    Common Datumaro dataset importer plugin names.
    """

    COCO = "coco"
    COCO_INSTANCES = "coco_instances"
    CVAT = "cvat"
    DATUMARO = "datumaro"
    IMAGE_DIR = "image_dir"
    IMAGENET = "imagenet"
    KITTI = "kitti"
    LABELME = "label_me"
    OPEN_IMAGES = "open_images"
    PASCAL_VOC = "voc"
    ROBOFLOW_COCO = "roboflow_coco"
    ROBOFLOW_VOC = "roboflow_voc"
    ROBOFLOW_YOLO = "roboflow_yolo"
    YOLO = "yolo"


class ExportFormat(StrEnum):
    """
    Common Datumaro dataset exporter plugin names.
    """

    COCO = "coco"
    COCO_INSTANCES = "coco_instances"
    CVAT = "cvat"
    DATUMARO = "datumaro"
    IMAGE_DIR = "image_dir"
    IMAGENET = "imagenet"
    KITTI = "kitti"
    LABELME = "label_me"
    OPEN_IMAGES = "open_images"
    PASCAL_VOC = "voc"
    YOLO = "yolo"
    YOLO_ULTRALYTICS = "yolo_ultralytics"


def hardlink_or_copy_file(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
) -> None:
    """
    Hardlink a file to a destination, falling back to copy when unavailable.
    """
    src_path = Path(src)
    dst_path = Path(dst)

    if dst_path.exists() and src_path.samefile(dst_path):
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()

    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)


@contextmanager
def prefer_hardlinked_datumaro_media() -> Iterator[None]:
    """
    Make Datumaro media exports use hardlinks before falling back to copies.
    """
    original_copyto_image = datumaro_media.copyto_image

    def copyto_image(src: str | IOBase, dst: str | IOBase) -> None:
        """
        Copy Datumaro image data with a hardlink-first path for file paths.
        """
        if isinstance(src, (str, os.PathLike)) and isinstance(
            dst,
            (str, os.PathLike),
        ):
            hardlink_or_copy_file(src, dst)
            return

        original_copyto_image(src, dst)

    datumaro_media.copyto_image = copyto_image
    try:
        yield
    finally:
        datumaro_media.copyto_image = original_copyto_image


def summarize_datumaro_label_counts(dataset: dm.Dataset) -> pd.DataFrame:
    """
    Return dataset labels and annotation counts in Datumaro category order.
    """
    label_categories = dataset.categories()[
        dm.AnnotationType.label
    ].items  # pyright: ignore[reportAttributeAccessIssue]
    label_counts = {label.name: 0 for label in label_categories}

    for item in dataset:
        for annotation in item.annotations:
            label_id = annotation.label  # pyright: ignore[reportAttributeAccessIssue]
            if label_id is not None:
                label_counts[label_categories[label_id].name] += 1

    return pd.DataFrame(
        label_counts.items(),
        columns=["label", "annotation_count"],
    )
