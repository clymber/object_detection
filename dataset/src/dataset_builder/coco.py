"""
COCO JSON format dataset utilities.
"""

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from detection_common.utils.json_io import read_json


def normalize_split_name(split_name: str) -> str:
    """
    Return the split name normalized to a standard COCO JSON format.
    """
    if split_name in {"valid", "val"}:
        return "val"
    return split_name


def summarize_coco_annotation_file(
    annotation_path: Path,
) -> dict[str, object]:
    """
    Summarize one COCO annotation file.
    """
    coco = read_json(annotation_path)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    category_names = {
        category["id"]: category["name"] for category in coco.get("categories", [])
    }
    category_counts = Counter(
        annotation.get("category_id")
        for annotation in annotations
        if annotation.get("category_id") is not None
    )

    labels = [
        category_names.get(category_id, f"unknown:{category_id}")
        for category_id in category_counts
    ]

    return {
        "dataset_id": annotation_path.parents[2].name,
        "split": normalize_split_name(annotation_path.parent.name),
        "images": len(images),
        "annotations": len(annotations),
        "labels": labels,
    }


def format_class_names(class_names: pd.Series) -> str:
    """
    Format class names in a predictable alphabetical order.
    """
    return ", ".join(sorted(set(class_names), key=str.casefold))


def summarize_coco_dataset(dataset_path: Path | str) -> pd.DataFrame:
    """
    Summarize all COCO annotation files in a dataset.
    """
    dataset_path = Path(dataset_path)
    annotation_paths = list(dataset_path.rglob("*.coco.json"))
    split_summary_df = pd.DataFrame(
        summarize_coco_annotation_file(annotation_path)
        for annotation_path in annotation_paths
    )

    image_summary_df = split_summary_df.pivot_table(
        index="dataset_id",
        columns="split",
        values="images",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    image_columns = ["train", "val", "test"]
    for column in image_columns:
        if column not in image_summary_df.columns:
            image_summary_df[column] = 0

    label_summary_df = (
        split_summary_df.explode("labels")
        .groupby("dataset_id", as_index=False)
        .agg(
            classes=("labels", "nunique"),
            labels=("labels", format_class_names),
        )
    )
    annotation_summary_df = (
        split_summary_df.groupby("dataset_id", as_index=False)
        .agg(annotations=("annotations", "sum"))
    )
    dataset_summary_df = (
        image_summary_df[["dataset_id", *image_columns]]
        .merge(label_summary_df, on="dataset_id")
        .merge(annotation_summary_df, on="dataset_id")
        .sort_values("dataset_id")
        .reset_index(drop=True)
    )

    return dataset_summary_df


def summarize_coco_datasets(
    ds_paths: Iterable[Path | str],
) -> pd.DataFrame:
    """
    Summarize all COCO datasets in an iterable of dataset paths.
    """
    summaries = [summarize_coco_dataset(ds_path) for ds_path in ds_paths]
    if not summaries:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "train",
                "val",
                "test",
                "classes",
                "labels",
                "annotations",
            ]
        )

    return (
        pd.concat(summaries, ignore_index=True)
        .sort_values("dataset_id")
        .reset_index(drop=True)
    )


def filter_coco_annotation_by_labels(
    src_annotation: dict[str, Any],
    labels_include: set[str] | None = None,
    labels_exclude: set[str] | None = None,
) -> dict[str, Any]:
    """
    Filter a COCO annotation dictionary by included or excluded labels.

    If `labels_exclude` is provided, annotations and images associated with
    those labels are removed. If `labels_include` is provided, only annotations
    and images associated with those labels are kept. If both are provided, both
    rules are applied.
    """
    categories = src_annotation.get("categories", [])
    images = src_annotation.get("images", [])
    annotations = src_annotation.get("annotations", [])
    category_id_to_name = {
        category["id"]: category["name"] for category in categories
    }
    name_to_category_id = {v: k for k, v in category_id_to_name.items()}

    include_ids = (
        {
            name_to_category_id[label]
            for label in labels_include
            if label in name_to_category_id
        }
        if labels_include is not None
        else None
    )
    exclude_ids = (
        {
            name_to_category_id[label]
            for label in labels_exclude
            if label in name_to_category_id
        }
        if labels_exclude is not None
        else None
    )

    allowed_category_ids = set(category_id_to_name)
    if include_ids is not None:
        allowed_category_ids &= include_ids
    if exclude_ids is not None:
        allowed_category_ids -= exclude_ids

    image_ids = {image.get("id") for image in images}
    if include_ids is not None:
        image_ids &= {
            annotation.get("image_id")
            for annotation in annotations
            if annotation.get("category_id") in include_ids
        }
    if exclude_ids is not None:
        image_ids -= {
            annotation.get("image_id")
            for annotation in annotations
            if annotation.get("category_id") in exclude_ids
        }

    filtered_fields = {
        "categories": [
            category
            for category in categories
            if category.get("id") in allowed_category_ids
        ],
        "images": [image for image in images if image.get("id") in image_ids],
        "annotations": [
            annotation
            for annotation in annotations
            if annotation.get("image_id") in image_ids
            and annotation.get("category_id") in allowed_category_ids
        ],
    }

    new_annotation = {}
    for key, value in src_annotation.items():
        new_annotation[key] = filtered_fields.get(key, value)

    for key, value in filtered_fields.items():
        if key not in new_annotation:
            new_annotation[key] = value

    return new_annotation
