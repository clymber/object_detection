"""
Dataset-related utilities and classes.
"""

from .coco import (
    filter_coco_annotation_by_labels,
    summarize_coco_dataset,
    summarize_coco_datasets,
)
from .identity import (
    canonical_coco_identity,
    capture_dataset_identity,
    prepare_smoke_layouts,
    smoke_layout_directory,
    validate_rfdetr_layout,
    validate_yolo_layout,
    verify_dataset_identity,
)


def __getattr__(name: str):
    """
    Load optional Datumaro helpers only when preprocessing callers request them.
    """
    if name in {"prefer_hardlinked_datumaro_media", "summarize_datumaro_label_counts"}:
        from . import datumaro

        value = getattr(datumaro, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "filter_coco_annotation_by_labels",
    "canonical_coco_identity",
    "capture_dataset_identity",
    "prefer_hardlinked_datumaro_media",
    "prepare_smoke_layouts",
    "smoke_layout_directory",
    "summarize_coco_dataset",
    "summarize_coco_datasets",
    "summarize_datumaro_label_counts",
    "validate_rfdetr_layout",
    "validate_yolo_layout",
    "verify_dataset_identity",
]
