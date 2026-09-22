"""
Exercise canonical source identities and framework-neutral smoke layouts.
"""

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from dataset_builder.identity import (
    IDENTITY_FILE_NAME,
    canonical_coco_identity,
    capture_dataset_identity,
    prepare_smoke_layouts,
    validate_rfdetr_layout,
    validate_yolo_layout,
    verify_dataset_identity,
)
from dataset_builder.rfdetr import prepare_coco_dataset


def _source(
    root: Path, counts: dict[str, int] | None = None, *, class_name: str = "basketball"
) -> Path:
    """
    Write a one-class source with deterministic positive and negative images.
    """
    counts = counts or {"train": 18, "val": 9, "test": 9}
    for split_index, (split, count) in enumerate(counts.items()):
        images = []
        annotations = []
        for index in range(count):
            relative = Path("nested") / f"image_{index:02}.png"
            image_path = root / "images" / split / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new(
                "RGB",
                (40, 20),
                (split_index * 50, index * 11, 30),
            ).save(image_path)
            image_id = split_index * 100 + index + 1
            images.append(
                {
                    "id": image_id,
                    "file_name": relative.as_posix(),
                    "width": 40,
                    "height": 20,
                }
            )
            if index % 2 == 0:
                annotations.append(
                    {
                        "id": image_id,
                        "image_id": image_id,
                        "category_id": 7,
                        "bbox": [4.0, 2.0, 20.0, 10.0],
                        "area": 200.0,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
        annotation_path = root / "annotations" / f"instances_{split}.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            json.dumps(
                {
                    "info": {"created": "not part of identity"},
                    "categories": [
                        {
                            "id": 7,
                            "name": class_name,
                            "supercategory": "ball",
                        }
                    ],
                    "images": images,
                    "annotations": annotations,
                }
            ),
            encoding="utf-8",
        )
    return root


def _edit_annotation(path: Path, edit) -> None:
    """
    Apply one test mutation to a COCO annotation document.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_canonical_identity_is_relocation_independent_and_content_sensitive(
    tmp_path: Path,
) -> None:
    """
    Ignore roots and metadata while hashing source boxes, mappings, and image bytes.
    """
    source = _source(tmp_path / "first")
    relocated = tmp_path / "second"
    shutil.copytree(source, relocated)
    original = canonical_coco_identity(source)
    assert canonical_coco_identity(relocated) == original

    _edit_annotation(
        relocated / "annotations" / "instances_train.json",
        lambda payload: payload["annotations"][0].update(bbox=[5.0, 2.0, 20.0, 10.0]),
    )
    assert canonical_coco_identity(relocated)["source_fingerprint"] != original[
        "source_fingerprint"
    ]

    shutil.rmtree(relocated)
    shutil.copytree(source, relocated)
    Image.new("RGB", (40, 20), (1, 2, 3)).save(
        relocated / "images" / "test" / "nested" / "image_01.png"
    )
    assert canonical_coco_identity(relocated)["source_fingerprint"] != original[
        "source_fingerprint"
    ]

    shutil.rmtree(relocated)
    shutil.copytree(source, relocated)
    for split in ("train", "val", "test"):
        _edit_annotation(
            relocated / "annotations" / f"instances_{split}.json",
            lambda payload: (
                payload["categories"][0].update(id=9),
                [
                    annotation.update(category_id=9)
                    for annotation in payload["annotations"]
                ],
            ),
        )
    assert canonical_coco_identity(relocated)["source_fingerprint"] != original[
        "source_fingerprint"
    ]


@pytest.mark.parametrize("class_name", ["basketball", "football"])
def test_smoke_layouts_are_deterministic_and_retain_negatives(
    tmp_path: Path, class_name: str
) -> None:
    """
    Build 16/8/8 COCO, YOLO, and RF-DETR layouts with original IDs and boxes.
    """
    source = _source(tmp_path / "source", class_name=class_name)
    destination = tmp_path / "processed" / f"{class_name}_smoke"
    record = prepare_smoke_layouts(source, destination, link_images=False)

    manifest = json.loads((destination / "rfdetr" / "manifest.json").read_text())
    assert manifest["category_mapping"]["class_names"] == [class_name]
    assert (
        canonical_coco_identity(source)["canonical_source"]["category_mapping"][0][
            "name"
        ]
        == class_name
    )
    assert record["selected_ids"] == {
        "train": list(range(1, 17)),
        "val": list(range(101, 109)),
        "test": list(range(201, 209)),
    }
    assert record["smoke_limits"] == {"train": 16, "val": 8, "test": 8}
    assert (destination / IDENTITY_FILE_NAME).is_file()
    for split, count in {"train": 16, "val": 8, "test": 8}.items():
        payload = json.loads(
            (
                destination / "coco" / "annotations" / f"instances_{split}.json"
            ).read_text(
                encoding="utf-8"
            )
        )
        assert len(payload["images"]) == count
        assert any(not annotation for annotation in [
            [
                entry
                for entry in payload["annotations"]
                if entry["image_id"] == image["id"]
            ]
            for image in payload["images"]
        ])
        assert payload["annotations"][0]["bbox"] == [4.0, 2.0, 20.0, 10.0]

    yolo = validate_yolo_layout(destination / "coco", destination / "yolo")
    rfdetr = validate_rfdetr_layout(destination / "coco", destination / "rfdetr")
    assert yolo["source_fingerprint"] == record["subset_source_fingerprint"]
    assert rfdetr["source_fingerprint"] == record["subset_source_fingerprint"]
    assert prepare_smoke_layouts(source, destination, link_images=False) == record


def test_linked_smoke_layouts_remain_valid_after_staging_promotion(
    tmp_path: Path,
) -> None:
    """
    Keep default linked YOLO media valid after the staging directory is renamed.
    """
    source = _source(tmp_path / "source")
    destination = tmp_path / "processed"
    prepare_smoke_layouts(source, destination)

    validate_yolo_layout(destination / "coco", destination / "yolo")
    validate_rfdetr_layout(destination / "coco", destination / "rfdetr")
    manifest = json.loads((destination / "rfdetr" / "manifest.json").read_text())
    assert Path(manifest["source_dir"]) == destination / "coco"
    assert Path(manifest["dataset_dir"]) == destination / "rfdetr"
    # Reconstructing the manifest must agree with the promoted layout too.
    assert prepare_coco_dataset(
        destination / "coco", destination / "rfdetr", link_images=False
    ) == manifest


def test_yolo_layout_accepts_image_lists_and_tracks_their_contents(
    tmp_path: Path,
) -> None:
    """
    Validate Datumaro-style split lists and fingerprint the images they select.
    """
    source = _source(tmp_path / "source")
    destination = tmp_path / "processed"
    prepare_smoke_layouts(source, destination, link_images=False)
    yolo_dir = destination / "yolo"
    yaml = yolo_dir / "data.yaml"
    yaml.write_text(
        "path: .\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
        "names:\n  0: basketball\n",
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        images = sorted((yolo_dir / "images" / split).rglob("*.png"))
        (yolo_dir / f"{split}.txt").write_text(
            "".join(
                f"./{image.relative_to(yolo_dir).as_posix()}\n"
                for image in images
            ),
            encoding="utf-8",
        )

    original = validate_yolo_layout(destination / "coco", yolo_dir)
    assert set(original["loader"]["image_list_sha256"]) == {
        "train", "val", "test"
    }
    train_list = yolo_dir / "train.txt"
    lines = train_list.read_text(encoding="utf-8").splitlines()
    train_list.write_text(
        "\n".join([lines[1], lines[0], *lines[2:]]) + "\n",
        encoding="utf-8",
    )
    reordered = validate_yolo_layout(destination / "coco", yolo_dir)
    assert reordered["loader_fingerprint"] != original["loader_fingerprint"]

    train_list.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list membership differs"):
        validate_yolo_layout(destination / "coco", yolo_dir)

    train_list.write_text(
        "\n".join([*lines, lines[0]]) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="list membership differs"):
        validate_yolo_layout(destination / "coco", yolo_dir)


def test_yolo_layout_rejects_changed_labels_images_and_membership(
    tmp_path: Path,
) -> None:
    """
    Reject loader changes that affect boxes, image bytes, or split membership.
    """
    source = _source(tmp_path / "source")
    destination = tmp_path / "processed"
    prepare_smoke_layouts(source, destination, link_images=False)
    label = destination / "yolo" / "labels" / "train" / "nested" / "image_00.txt"
    label.write_text("0 0.6 0.35 0.5 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="labels disagree"):
        validate_yolo_layout(destination / "coco", destination / "yolo")

    shutil.rmtree(destination)
    prepare_smoke_layouts(source, destination, link_images=False)
    image = destination / "yolo" / "images" / "train" / "nested" / "image_01.png"
    Image.new("RGB", (40, 20), (1, 2, 3)).save(image)
    with pytest.raises(ValueError, match="membership differs"):
        validate_yolo_layout(destination / "coco", destination / "yolo")

    shutil.rmtree(destination)
    prepare_smoke_layouts(source, destination, link_images=False)
    source_image = destination / "yolo" / "images" / "train" / "nested" / "image_01.png"
    source_label = destination / "yolo" / "labels" / "train" / "nested" / "image_01.txt"
    moved_image = destination / "yolo" / "images" / "val" / "nested" / "image_01.png"
    moved_label = destination / "yolo" / "labels" / "val" / "nested" / "image_01.txt"
    moved_image.parent.mkdir(parents=True, exist_ok=True)
    moved_label.parent.mkdir(parents=True, exist_ok=True)
    source_image.rename(moved_image)
    source_label.rename(moved_label)
    with pytest.raises(ValueError, match="train image membership differs"):
        validate_yolo_layout(destination / "coco", destination / "yolo")


def test_yolo_layout_allows_missing_negative_labels_and_rejects_positives(
    tmp_path: Path,
) -> None:
    """
    Accept standard absent negative labels but reject missing annotated targets.
    """
    source = _source(tmp_path / "source")
    destination = tmp_path / "processed"
    prepare_smoke_layouts(source, destination, link_images=False)
    negative_label = destination / "yolo" / "labels" / "train" / "nested" / "image_01.txt"
    negative_label.unlink()
    validate_yolo_layout(destination / "coco", destination / "yolo")

    positive_label = destination / "yolo" / "labels" / "train" / "nested" / "image_00.txt"
    positive_label.unlink()
    with pytest.raises(ValueError, match="labels disagree"):
        validate_yolo_layout(destination / "coco", destination / "yolo")


def test_rfdetr_layout_and_identity_record_reject_changes_without_overwrite(
    tmp_path: Path,
) -> None:
    """
    Preserve an identity record while stale RF-DETR annotations fail validation.
    """
    source = _source(tmp_path / "source")
    destination = tmp_path / "processed"
    record = prepare_smoke_layouts(source, destination, link_images=False)
    identity_path = destination / IDENTITY_FILE_NAME
    saved = identity_path.read_bytes()
    with pytest.raises(ValueError, match="original record was preserved"):
        verify_dataset_identity(identity_path, {"unexpected": "identity"})
    assert identity_path.read_bytes() == saved
    with pytest.raises(ValueError, match="original record was preserved"):
        capture_dataset_identity(identity_path, {"unexpected": "identity"})
    assert identity_path.read_bytes() == saved
    assert capture_dataset_identity(identity_path, record) == record

    annotation_path = destination / "rfdetr" / "train" / "_annotations.coco.json"
    _edit_annotation(
        annotation_path,
        lambda payload: payload["annotations"][0].update(category_id=99),
    )
    with pytest.raises(ValueError, match="annotations changed"):
        validate_rfdetr_layout(destination / "coco", destination / "rfdetr")


def test_identity_records_normalize_equivalent_json_collections(tmp_path: Path) -> None:
    """
    Treat tuples and lists alike after their persisted JSON representation.
    """
    identity_path = tmp_path / IDENTITY_FILE_NAME
    capture_dataset_identity(identity_path, {"boxes": [(0, 0.5)]})

    assert verify_dataset_identity(identity_path, {"boxes": [[0, 0.5]]}) == {
        "boxes": [[0, 0.5]]
    }

def test_default_smoke_paths_preserve_subsets_of_previous_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Select another smoke directory when source contents change, preserving old data.
    """
    from dataset_builder import identity

    monkeypatch.setattr(identity, "DATA_ROOT", tmp_path)
    source = _source(tmp_path / "source")
    old_root = identity.smoke_layout_directory(source)
    old_record = prepare_smoke_layouts(source)
    annotation = source / "annotations" / "instances_train.json"
    payload = json.loads(annotation.read_text())
    payload["annotations"][0]["bbox"][0] += 1
    annotation.write_text(json.dumps(payload))
    new_root = identity.smoke_layout_directory(source)
    assert new_root != old_root
    new_record = prepare_smoke_layouts(source)
    assert new_record["parent_source_fingerprint"] != old_record["parent_source_fingerprint"]
    assert json.loads((old_root / IDENTITY_FILE_NAME).read_text()) == old_record


@pytest.mark.parametrize("name", [None, "", "   ", 42])
def test_invalid_category_names_are_rejected(tmp_path: Path, name: object) -> None:
    """Reject malformed category names while permitting non-basketball classes."""
    source = _source(tmp_path / "source")
    _edit_annotation(
        source / "annotations" / "instances_train.json",
        lambda payload: payload["categories"][0].update(name=name),
    )
    with pytest.raises(ValueError, match="nonempty category name"):
        canonical_coco_identity(source)


def test_category_names_must_match_across_splits(tmp_path: Path) -> None:
    """Reject a changed class meaning even when category IDs match."""
    source = _source(tmp_path / "source", class_name="football")
    _edit_annotation(
        source / "annotations" / "instances_val.json",
        lambda payload: payload["categories"][0].update(name="basketball"),
    )
    with pytest.raises(ValueError, match="category mapping differs"):
        canonical_coco_identity(source)
    with pytest.raises(ValueError, match="Category names or IDs differ"):
        prepare_coco_dataset(source, tmp_path / "derived")


@pytest.mark.parametrize("first_training_id", [None, 1])
def test_multiclass_layouts_preserve_all_categories(
    tmp_path: Path, first_training_id: int | None
) -> None:
    """Round-trip noncontiguous IDs through COCO, YOLO and RF-DETR smoke layouts."""
    source = _source(tmp_path / "source", class_name="football")
    for split in ("train", "val", "test"):

        def add_class(payload):
            payload["categories"].insert(
                0, {"id": 19, "name": "basketball", "supercategory": "ball"}
            )
            for index, annotation in enumerate(payload["annotations"]):
                if index % 2:
                    annotation["category_id"] = 19

        _edit_annotation(source / "annotations" / f"instances_{split}.json", add_class)
    identity = canonical_coco_identity(source)
    assert [
        item["id"] for item in identity["canonical_source"]["category_mapping"]
    ] == [7, 19]
    derived = tmp_path / "rfdetr"
    manifest = prepare_coco_dataset(
        source, derived, category_id=first_training_id, link_images=False
    )
    expected = {"7": 7, "19": 19} if first_training_id is None else {"7": 1, "19": 2}
    assert manifest["category_mapping"]["source_to_training"] == expected
    assert manifest["category_mapping"]["prediction_to_source"] == {"0": 7, "1": 19}
    assert manifest["category_mapping"]["class_names"] == ["football", "basketball"]
    assert (
        validate_rfdetr_layout(source, derived)["source_fingerprint"]
        == identity["source_fingerprint"]
    )
    smoke = tmp_path / "smoke"
    record = prepare_smoke_layouts(source, smoke, link_images=False)
    assert prepare_smoke_layouts(source, smoke, link_images=False) == record
    assert (
        validate_yolo_layout(smoke / "coco", smoke / "yolo")["source_fingerprint"]
        == record["subset_source_fingerprint"]
    )
    assert (
        validate_rfdetr_layout(smoke / "coco", smoke / "rfdetr")["source_fingerprint"]
        == record["subset_source_fingerprint"]
    )
    labels = "".join(
        path.read_text() for path in (smoke / "yolo/labels/train").rglob("*.txt")
    )
    assert {line.split()[0] for line in labels.splitlines()} == {"0", "1"}


@pytest.mark.parametrize(
    "category",
    [
        {"id": 7, "name": "other"},
        {"id": 19, "name": "basketball"},
    ],
)
def test_duplicate_categories_are_rejected(tmp_path: Path, category: dict) -> None:
    """Reject duplicate IDs or names instead of conflating model labels."""
    source = _source(tmp_path / "source")
    _edit_annotation(
        source / "annotations/instances_train.json",
        lambda payload: payload["categories"].append(category),
    )
    with pytest.raises(ValueError, match="duplicate category"):
        canonical_coco_identity(source)
