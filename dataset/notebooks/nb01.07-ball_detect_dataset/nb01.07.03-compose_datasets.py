# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     formats: ipynb,py:percent
#     notebook_metadata_filter: jupytext,title,authors,-kernelspec,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Compose sport datasets
#
# Run `nb01.07.01` to import sources and `nb01.07.02` to review duplicates first.
# This notebook reads `sports_objects_master` and exports four versioned COCO
# datasets. Samples tagged `review:*duplicate` are excluded; source annotations,
# images, and review decisions remain unchanged.
#
# Each export lives at `DATA_ROOT/composed/<target-name>/coco/`, with
# `annotations/instances_{train,val,test}.json` and `images/{train,val,test}/`.
# Images use their cached `pixel_digest` plus the original extension. Media is
# hardlinked when possible, with a copy fallback. Existing targets stop the build.
#
# `Dataset(root).coco` returns the COCO directory. `Dataset(root).yolo` creates
# a matching YOLO export on first access, then reuses it. Keep each version's
# COCO content fixed so both formats stay consistent.

# %%
from fnmatch import fnmatchcase
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import datumaro as dm
import fiftyone as fo
import pandas as pd
from detection_common import configure_stdio_relative_path
from IPython.display import display

from dataset_builder import Dataset, prefer_hardlinked_datumaro_media
from dataset_builder.config import DATA_ROOT, WORKSPACE_ROOT
from dataset_builder.datumaro import ExportFormat

configure_stdio_relative_path(WORKSPACE_ROOT)
COMPOSED_ROOT = DATA_ROOT / "composed"
SPLITS = ("train", "val", "test")

# %% [markdown]
# ## 1. Select retained images
#
# Filtering uses existing review tags only. Basketball, Gaelic football, and
# tennis keep their original splits (`valid` becomes `val`). Football uses the
# fixed source assignments shown below.

# %%
master = fo.load_dataset("sports_objects_master")
duplicate_tags = [
    tag for tag in cast(list[str], master.distinct("tags"))
    if fnmatchcase(tag, "review:*duplicate")
]
retained_view = (
    master.match_tags(duplicate_tags, bool=False) if duplicate_tags else master.view()
)
print(f"Retained images: {retained_view.count():,} / {master.count():,}")

# %% [markdown]
# ### Shared export helper
#
# Check source labels before exporting, then build independent Datumaro items.
# A mapping value of `None` removes an annotation; an image with no remaining
# annotations is kept as background. Whole-image exclusions run first.

# %%
def export_dataset(
    collection, name, label_maps, *, classes=None, split_by_source=None,
    exclude_images=None,
):
    """
    Export mapped detections as canonical COCO without modifying the collection.
    """
    root = COMPOSED_ROOT / name
    if root.exists():
        raise FileExistsError(f"Target already exists: {root}")
    view = collection.match(fo.ViewField("source_dataset").is_in(list(label_maps)))
    exclude_images = exclude_images or {}
    inventory = []
    for source, mapping in label_maps.items():
        source_view = view.match(fo.ViewField("source_dataset") == source)
        count = source_view.count()
        if not count:
            continue
        labels = sorted(source_view.distinct("original_labels.detections.label"))
        inventory.append({"source": source, "images": count, "labels": labels})
        if mapping is not None:
            unknown = set(labels) - set(mapping) - set(exclude_images.get(source, ()))
            if unknown:
                raise ValueError(f"Unexpected labels in {source}: {sorted(unknown)}")
    display(pd.DataFrame(inventory))
    classes = classes or sorted(view.distinct("original_labels.detections.label"))
    if not classes:
        raise ValueError(f"No labels available for {name}")
    label_ids = {label: index for index, label in enumerate(classes)}
    items, records, seen_digests = [], [], set()
    fields = ["source_dataset", "original_split", "pixel_digest", "metadata",
              "original_labels"]
    for sample in view.select_fields(fields).iter_samples():
        source = sample.source_dataset
        split = (split_by_source or {}).get(source, sample.original_split)
        split = "val" if split == "valid" else split
        if split not in SPLITS:
            raise ValueError(f"Unknown split for {source}: {split!r}")
        detections = sample.original_labels.detections if sample.original_labels else []
        excluded = any(d.label in exclude_images.get(source, ()) for d in detections)
        record = {"source": source, "split": split, "images": 0,
                  "annotations": 0, "backgrounds": 0, "excluded_images": int(excluded)}
        records.append(record)
        if excluded:
            continue
        path = Path(sample.filepath)
        digest = sample.pixel_digest
        if (
            not digest or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"Missing or invalid pixel_digest: {path}")
        filename = digest + path.suffix
        if digest in seen_digests:
            raise ValueError(f"Duplicate pixel_digest: {digest}")
        seen_digests.add(digest)
        width, height = sample.metadata.width, sample.metadata.height
        mapping, annotations = label_maps[source], []
        for detection in detections:
            label = detection.label if mapping is None else mapping[detection.label]
            if label is None:
                continue
            x, y, w, h = detection.bounding_box
            annotations.append(dm.Bbox(
                x * width, y * height, w * width, h * height,
                label=label_ids[label],
                attributes={
                    "is_crowd": bool(detection.get_attribute_value("iscrowd", 0))
                },
            ))
        items.append(dm.DatasetItem(
            id=digest, subset=split,
            media=dm.Image.from_file(str(path), size=(height, width)),
            annotations=annotations,
        ))
        record.update(
            images=1, annotations=len(annotations), backgrounds=int(not annotations)
        )
    dataset = dm.Dataset.from_iterable(items, categories=classes)
    COMPOSED_ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{name}-", dir=COMPOSED_ROOT) as temporary:
        staging = Path(temporary) / name
        with prefer_hardlinked_datumaro_media():
            dataset.export(str(staging / "coco"), format=ExportFormat.COCO_INSTANCES,
                           save_media=True, reindex=True)
        staging.rename(root)
    summary = pd.DataFrame(records).groupby(["source", "split"]).sum()
    display(summary)
    totals = summary.groupby("split").sum().reindex(SPLITS, fill_value=0)
    totals["image_percent"] = (100 * totals["images"] / totals["images"].sum()).round(2)
    display(totals)
    result = Dataset(root)
    print(f"COCO export: {result.coco}")
    return result

# %% [markdown]
# ## 2. Basketball — `basketball-v0.1`
#
# Combine `basketball-lokg2` and `basketball-xil7x`, preserving original splits.
# Source `ball-tracker-jpjs8` is excluded.
#
# | Original labels | Target label |
# | --- | --- |
# | `ball`, `basketball` | `basketball` |
# | `people`, `human` | `person` |
# | `rim` | `rim` |

# %%
basketball_labels = {
    "ball": "basketball", "basketball": "basketball",
    "people": "person", "human": "person", "rim": "rim",
}
basketball = export_dataset(
    retained_view, "basketball-v0.1",
    {source: basketball_labels for source in ("basketball-lokg2", "basketball-xil7x")},
    classes=["basketball", "person", "rim"],
)

# %% [markdown]
# ## 3. Football — `football-v0.1`
#
# Label matching is case-sensitive. Remove an entire image when `football-csdy0`
# contains `otherball` or `football-tnb17` contains `other`. Other removals below
# affect annotations only; keep the image even when it becomes background.
#
# | Source | Labels mapped to `football` | Other rules |
# | --- | --- | --- |
# | `football-2k4ty` | `football` | |
# | `football-5beob` | `ball` | Keep `person` |
# | `football-ablqb` | `football` | |
# | `football-csdy0` | `football` | Exclude images with `otherball` |
# | `football-detection-ftt4q` | `Football` | |
# | `football-detection-gjrtq` | `Football`, `ball` | `Player`, `referee` → `person`; remove `big` |
# | `football-gsgma` | `football`, `soccer`, `soccer ball`, `soccer_ball` | Remove `football2` |
# | `football-rmhvy` | `Football`, `football_1` | Remove `Not football`, `hball` |
# | `football-tnb17` | `football` | Exclude images with `other` |
# | `football-wsef2` | `football` | |
# | `football-ywyvk` | `Football` | |
# | `tracer-football` | `Soccer ball`, `football`, `soccer` | Remove `ball`, `other`, `others` |

# %%
football_labels = {
    "football-2k4ty": {"football": "football"},
    "football-5beob": {"ball": "football", "person": "person"},
    "football-ablqb": {"football": "football"},
    "football-csdy0": {"football": "football"},
    "football-detection-ftt4q": {"Football": "football"},
    "football-detection-gjrtq": {
        "Football": "football", "ball": "football",
        "Player": "person", "referee": "person", "big": None,
    },
    "football-gsgma": {
        "football": "football", "soccer": "football", "soccer ball": "football",
        "soccer_ball": "football", "football2": None,
    },
    "football-rmhvy": {
        "Football": "football", "football_1": "football",
        "Not football": None, "hball": None,
    },
    "football-tnb17": {"football": "football"},
    "football-wsef2": {"football": "football"},
    "football-ywyvk": {"Football": "football"},
    "tracer-football": {
        "Soccer ball": "football", "football": "football", "soccer": "football",
        "ball": None, "other": None, "others": None,
    },
}

# %% [markdown]
# ### Fixed source assignments
#
# As in `nb01.03`, each source belongs wholly to one target split. Combine its
# original partitions using these fixed choices. After filtering, the planned
# totals are 3,288 / 697 / 685 images, approximately 70.41:14.93:14.67.
#
# | Source | Retained images | Target split |
# | --- | ---: | --- |
# | `football-detection-ftt4q` | 799 | train |
# | `football-detection-gjrtq` | 592 | train |
# | `football-gsgma` | 663 | train |
# | `football-rmhvy` | 1,234 | train |
# | `football-2k4ty` | 191 | val |
# | `football-ablqb` | 100 | val |
# | `football-csdy0` | 81 | val |
# | `football-ywyvk` | 325 | val |
# | `football-5beob` | 28 | test |
# | `football-tnb17` | 43 | test |
# | `football-wsef2` | 133 | test |
# | `tracer-football` | 481 | test |

# %%
football_splits = {
    "football-detection-ftt4q": "train", "football-detection-gjrtq": "train",
    "football-gsgma": "train", "football-rmhvy": "train",
    "football-2k4ty": "val", "football-ablqb": "val",
    "football-csdy0": "val", "football-ywyvk": "val",
    "football-5beob": "test", "football-tnb17": "test",
    "football-wsef2": "test", "tracer-football": "test",
}
football = export_dataset(
    retained_view, "football-v0.1", football_labels,
    classes=["football", "person"], split_by_source=football_splits,
    exclude_images={"football-csdy0": {"otherball"}, "football-tnb17": {"other"}},
)

# %% [markdown]
# ## 4. Gaelic football — `gaelic_football-v0.1`
#
# Export retained samples from `coco_gaelic_football`, preserving their current
# label names, boxes, and original splits. No label remapping is applied.

# %%
gaelic_football = export_dataset(
    retained_view, "gaelic_football-v0.1", {"coco_gaelic_football": None},
)

# %% [markdown]
# ## 5. Tennis — `tennis-v0.1`
#
# Export retained samples from `tennis-racket-r6mgq`, preserving their current
# label names, boxes, and original splits. No label remapping is applied.

# %%
tennis = export_dataset(
    retained_view, "tennis-v0.1", {"tennis-racket-r6mgq": None},
)

# %% [markdown]
# ## Use an export
#
# Each returned object is a `dataset_builder.Dataset`. Use `football.coco` for
# the canonical COCO path. Request `football.yolo` when YOLO is needed; the first
# access converts COCO and subsequent accesses reuse the existing directory.
# A YOLO training configuration is then at `football.yolo / "data.yaml"`.
