# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags
#     notebook_metadata_filter: jupytext,title,authors,-kernelspec,-jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %%
# %load_ext autoreload
# %autoreload 2
# %aimport -fiftyone

from detection_common import configure_stdio_relative_path

from dataset_builder.config import WORKSPACE_ROOT

# Display project paths relatively for consistent output across environments.
configure_stdio_relative_path(WORKSPACE_ROOT)

# %%
from collections import defaultdict
from typing import cast

import fiftyone as fo
from IPython.display import display

MASTER_DATASET = "sports_objects_master"

# %% [markdown]
# # Dataset Deduplication
#
# 1. **Exact-duplicate detection**. Select one representative per exact group for
# subsequent analysis.
# 2. **Compute and cache a separate perceptual_hash field**. Compare hashes using
# Hamming distance, then display candidate pairs or groups for review.
# 3. **Run embedding-based detection on the remaining images**. Exclude samples
#    with any tag matching `review*duplicate` before computing embeddings.
# 4. **Record each method separately**, for example with `review:perceptual_duplicate`
# and `review:embedding_duplicate` tags. Show references alongside candidates and retain
# all samples during review.
#
# Tag exact pixel duplicates using the `pixel_digest` field cached during import.
# This does not detect pixel changes introduced by lossy recompression.

# %% [markdown]
# ## 1. Load the master sport object detection dataset
#
# The "sports_objects_master" should have been created by `nb01.07.01`.

# %%
import logging

logging.getLogger("fiftyone.core.config").setLevel(logging.ERROR)

if not fo.dataset_exists(MASTER_DATASET):
    raise ValueError(f"Dataset {MASTER_DATASET!r} not found. Run nb01.07.01 first.")

master = fo.load_dataset(MASTER_DATASET)
master.persistent = True
if not master.has_sample_field("pixel_digest"):
    raise ValueError("Run nb01.07.01 to populate the pixel_digest field first.")

# %% [markdown]
# ## 2. Detecting exact duplicates (image pixel hash)
#
# ### 2.1 Tag exact pixel duplicates for review
#
# Group the stored digests without reopening images. Keep the first sample in
# each group untagged and tag the remaining copies. Refresh this review tag on
# reruns. This includes byte-identical files and other encodings with identical
# normalized pixels. Samples and media files are retained for review.

# %%
# FiftyOne's inferred aggregation type does not describe this two-field result.
sample_ids, pixel_digests = cast(
    tuple[list[str], list[str | None]],
    master.values(["id", "pixel_digest"]),
)
if any(not digest for digest in pixel_digests):
    raise ValueError("Every sample must have a pixel_digest before tagging.")

ids_by_digest = defaultdict(list)
for sample_id, digest in zip(sample_ids, pixel_digests):
    ids_by_digest[digest].append(sample_id)

duplicate_ids = [sid for ids in ids_by_digest.values() for sid in ids[1:]]
master.untag_samples("review:exact_duplicate")
if duplicate_ids:
    master.select(duplicate_ids).tag_samples("review:exact_duplicate")

dup_groups_cnt = sum(len(ids) > 1 for ids in ids_by_digest.values())
print("Exact pixel duplicate groups:", dup_groups_cnt)
print("Tagged extra copies:", len(duplicate_ids))

# %% [markdown]
# ### 2.2 Run FiftyOne Apps for visual reviews of exact duplicates.
#
# #### 2.2.1 Show all duplicate samples
#
# Show all duplicate groups, including their untagged first samples. Rebuild the
# selection from the digest groups so earlier reviews cannot consume it. Sort by
# digest to keep matching samples adjacent in the App.

# %%
all_duplicate_ids = [
    sid for ids in ids_by_digest.values() if len(ids) > 1 for sid in ids
]
review_view = master.select(all_duplicate_ids).sort_by("pixel_digest")
session = fo.launch_app(view=review_view)

# %% [marksown] [markdown]
# #### 2.2.2 Iteratively review
# %%
session.freeze()
duplicate_samples = (ids for ids in ids_by_digest.values() if len(ids) > 1)

# %% [markdown]
# Iteratively run the following cell to do visual review on the duplicated samples:

# %%
group_ids = next(duplicate_samples, None)
if group_ids:
    review_view = master.select(group_ids)
    session = fo.launch_app(view=review_view)

# %%
session.freeze()

# %% [markdown]
# ## 3. Detecting perceptually duplicates (perceptual hash)
#
# ### 3.1 Set up pHash
#
# Use `imagehash.phash(image, hash_size=PHASH_SIZE)` for 256-bit fingerprints.
# ImageHash hashes and compares images on CPU without a model download.

# %%
import imagehash

PHASH_SIZE = 16

# %% [markdown]
# ### 3.2 Select images to compare
#
# Keep the first sample from every pixel-digest group, including singletons.
# Reuse the digest groups so changes to review tags do not affect selection.
# Map each representative to its other exact copies; singletons have no copies.
# This creates a view without changing samples, tags, or source images.

# %%
exact_copies_by_representative = {ids[0]: ids[1:] for ids in ids_by_digest.values()}
representative_ids = list(exact_copies_by_representative)
print("Images selected for comparison:", len(representative_ids))

representative_view = master.select(representative_ids, ordered=True)
exclude_cnt = sum(len(copies) for copies in exact_copies_by_representative.values())
print("Exact copies excluded from comparison:", exclude_cnt)

# %% [markdown]
# ### 3.3 Compute and cache perceptual hashes
#
# Open only representatives without a cached hash. Apply EXIF orientation before
# computing the 256-bit pHash, then store its hexadecimal representation in FiftyOne.
# Source images and `pixel_digest` stay unchanged. Report unreadable images and
# leave their hashes missing so they are excluded from the comparison view.

# %%
from PIL import Image, ImageOps

if not master.has_sample_field("perceptual_hash"):
    master.add_sample_field("perceptual_hash", fo.StringField)

unhashed_ids, unhashed_paths = cast(
    tuple[list[str], list[str]],
    representative_view.exists("perceptual_hash", False).values(["id", "filepath"]),
)
new_perceptual_hashes = {}
unreadable_ids = []

with fo.ProgressBar(total=len(unhashed_ids)) as progress:
    for sample_id, image_path in progress(zip(unhashed_ids, unhashed_paths)):
        try:
            with Image.open(image_path) as image:
                normalized = ImageOps.exif_transpose(image)
                new_perceptual_hashes[sample_id] = str(
                    imagehash.phash(normalized, hash_size=PHASH_SIZE)
                )
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            unreadable_ids.append(sample_id)
            print(f"Skipping unreadable image {sample_id} ({image_path}): {error}")

if new_perceptual_hashes:
    master.set_values("perceptual_hash", new_perceptual_hashes, key_field="id")

hashed_representative_view = representative_view.exists("perceptual_hash")
print("Reused cached hashes:", len(representative_ids) - len(unhashed_ids))
print("Computed hashes:", len(new_perceptual_hashes))
print("Unreadable images skipped:", len(unreadable_ids))
print("Representatives ready for comparison:", hashed_representative_view.count())

# %% [markdown]
# ### 3.4 Find candidate pairs
#
# Compare cached hashes in batches, keeping only pairs within the largest trial
# threshold. XOR marks differing bits, and `bit_count()` gives Hamming distance.
# Compare each pair once and skip self-matches. No source images are opened.

# %%
from time import perf_counter

import pandas as pd

PHASH_THRESHOLDS = (0, 2, 4, 6)
PHASH_MAX_DISTANCE = max(PHASH_THRESHOLDS)
PAIR_BATCH_SIZE = 256

hashed_ids, cached_hashes = cast(
    tuple[list[str], list[str]],
    hashed_representative_view.values(["id", "perceptual_hash"]),
)
expected_hash_length = (PHASH_SIZE**2 + 3) // 4
if any(len(fingerprint) != expected_hash_length for fingerprint in cached_hashes):
    raise ValueError(
        "Cached pHash lengths do not match PHASH_SIZE. Recompute them first."
    )

hash_records = sorted(
    (sample_id, int(fingerprint, 16))
    for sample_id, fingerprint in zip(hashed_ids, cached_hashes)
)
pair_rows = []
comparison_started = perf_counter()
batch_starts = range(0, len(hash_records), PAIR_BATCH_SIZE)

with fo.ProgressBar(total=len(batch_starts), iters_str="batches") as progress:
    for batch_start in progress(batch_starts):
        batch_stop = min(batch_start + PAIR_BATCH_SIZE, len(hash_records))
        for index_a in range(batch_start, batch_stop):
            sample_id_a, hash_a = hash_records[index_a]
            for index_b in range(index_a + 1, len(hash_records)):
                sample_id_b, hash_b = hash_records[index_b]
                distance = (hash_a ^ hash_b).bit_count()
                if distance <= PHASH_MAX_DISTANCE:
                    pair_rows.append((sample_id_a, sample_id_b, distance))

comparison_seconds = perf_counter() - comparison_started
perceptual_pairs = pd.DataFrame(
    pair_rows,
    columns=["sample_id_a", "sample_id_b", "hamming_distance"],
).astype({"hamming_distance": "int64"})
perceptual_pairs = perceptual_pairs.sort_values(
    ["hamming_distance", "sample_id_a", "sample_id_b"]
).reset_index(drop=True)
del pair_rows

print("Hash pairs compared:", len(hash_records) * (len(hash_records) - 1) // 2)
print(f"Comparison time: {comparison_seconds:.2f} seconds")
threshold_counts = pd.DataFrame(
    [
        (threshold, int((perceptual_pairs["hamming_distance"] <= threshold).sum()))
        for threshold in PHASH_THRESHOLDS
    ],
    columns=["hamming_threshold", "candidate_pairs"],
)
display(threshold_counts)

# %% [markdown]
# ### 3.5 Tag perceptual duplicates automatically
#
# Use the calibrated distance threshold of 4. Prefer higher pixel area, then more
# original detections, then the first image in the representative selection.
# Every tagged image must directly match an untagged keeper within the threshold.
# A chain-linked review group can retain more than one image when necessary.
# Refresh only the automatic perceptual-duplicate tag; preserve all other tags.

# %%
from dataset_builder.deduplication import select_perceptual_duplicates

PHASH_DUPLICATE_THRESHOLD = 4
PERCEPTUAL_DUPLICATE_TAG = "review:perceptual_duplicate"

if not 0 <= PHASH_DUPLICATE_THRESHOLD <= PHASH_MAX_DISTANCE:
    raise ValueError("The duplicate threshold must be within the computed range.")

duplicate_candidate_pairs = perceptual_pairs.loc[
    perceptual_pairs["hamming_distance"] <= PHASH_DUPLICATE_THRESHOLD
]
duplicate_candidate_ids = set(
    duplicate_candidate_pairs[["sample_id_a", "sample_id_b"]].to_numpy().ravel()
)
quality_view = master.select(
    [sid for sid in representative_ids if sid in duplicate_candidate_ids], ordered=True
)
quality_ids, widths, heights, label_counts = cast(
    tuple[list[str], list[int], list[int], list[int]],
    quality_view.values(
        [
            "id", "metadata.width", "metadata.height",
            fo.ViewField("original_labels.detections").length(),
        ]
    ),
)
if any(not width or not height for width, height in zip(widths, heights)):
    raise ValueError(
        "Image dimensions are required. Run master.compute_metadata() first."
    )

sample_quality = {
    sid: (width * height, label_count)
    for sid, width, height, label_count in zip(
        quality_ids, widths, heights, label_counts
    )
}
perceptual_keeper_ids, duplicate_assignments = select_perceptual_duplicates(
    duplicate_candidate_pairs.itertuples(index=False, name=None),
    sample_quality,
    max_distance=PHASH_DUPLICATE_THRESHOLD,
)
perceptual_duplicate_ids = list(duplicate_assignments)
perceptual_duplicate_assignments = pd.DataFrame(
    [
        (sid, keeper, distance)
        for sid, (keeper, distance) in duplicate_assignments.items()
    ],
    columns=["duplicate_id", "keeper_id", "hamming_distance"],
)

master.untag_samples(PERCEPTUAL_DUPLICATE_TAG)
if perceptual_duplicate_ids:
    master.select(perceptual_duplicate_ids).tag_samples(PERCEPTUAL_DUPLICATE_TAG)

print("Automatic duplicate threshold:", PHASH_DUPLICATE_THRESHOLD)
print("Retained candidate images:", len(perceptual_keeper_ids))
print("Tagged perceptual duplicates:", len(perceptual_duplicate_ids))
display(perceptual_duplicate_assignments.head(20))

# %% [markdown]
# ### 3.6 Review overview in FiftyOne
#
# Manually review distances greater than 4 and at most 8 after automatic tagging.
# Exclude pairs involving an automatically tagged duplicate. Rerun this filter
# after changing the range; the cached pair table is reused without opening images.

# %%
PHASH_REVIEW_MIN_DISTANCE = 4
PHASH_REVIEW_MAX_DISTANCE = 6

if not 0 <= PHASH_REVIEW_MIN_DISTANCE < PHASH_REVIEW_MAX_DISTANCE <= PHASH_MAX_DISTANCE:
    raise ValueError("The review range must be within the computed distance range.")

review_excluded_ids = set(master.match_tags(PERCEPTUAL_DUPLICATE_TAG).values("id"))
manual_band_pairs = perceptual_pairs.loc[
    (perceptual_pairs["hamming_distance"] > PHASH_REVIEW_MIN_DISTANCE)
    & (perceptual_pairs["hamming_distance"] <= PHASH_REVIEW_MAX_DISTANCE)
]
candidate_pairs = manual_band_pairs.loc[
    ~manual_band_pairs["sample_id_a"].isin(review_excluded_ids)
    & ~manual_band_pairs["sample_id_b"].isin(review_excluded_ids)
].reset_index(drop=True)
print(
    f"Manual review distances: ({PHASH_REVIEW_MIN_DISTANCE}, "
    f"{PHASH_REVIEW_MAX_DISTANCE}]"
)
print("Pairs in distance range:", len(manual_band_pairs))
print("Pairs remaining after automatic tagging:", len(candidate_pairs))
display(candidate_pairs.head(20))

# %% [markdown]
# Show every image participating in the selected candidate pairs, once per image.
# Preserve the order of first appearance in the pair table, closest matches first.
# Rerun the filter above and this cell after changing the range.

# %%
perceptual_review_ids = list(
    dict.fromkeys(
        candidate_pairs[["sample_id_a", "sample_id_b"]].to_numpy().ravel()
    )
)
perceptual_review_view = master.select(perceptual_review_ids, ordered=True)

print(
    f"Manual review distances: ({PHASH_REVIEW_MIN_DISTANCE}, "
    f"{PHASH_REVIEW_MAX_DISTANCE}]"
)
print("Candidate pairs:", len(candidate_pairs))
print("Unique images in candidate pairs:", len(perceptual_review_ids))

if perceptual_review_ids:
    session = fo.launch_app(view=perceptual_review_view)
else:
    print("No candidate images to review in this distance range.")

# %%
session.freeze()

# %% [markdown]
# ### 3.7 Iteratively review candidate groups
#
# Run this setup cell once to start or restart the review queue. Rerun it after
# changing the review range and rebuilding `candidate_pairs`. The next cell uses the
# App session opened by the overview above.
#
# Images connected by candidate pairs form one review group. A-B and B-C put
# A, B, and C together even when A-C falls outside the review range. Inspect groups
# visually. Groups with lower maximum matched-pair distance appear first.

# %%
perceptual_neighbors = defaultdict(dict)
for pair in candidate_pairs.itertuples(index=False):
    perceptual_neighbors[pair.sample_id_a][pair.sample_id_b] = pair.hamming_distance
    perceptual_neighbors[pair.sample_id_b][pair.sample_id_a] = pair.hamming_distance

unvisited_ids = set(perceptual_neighbors)
group_rows = []
for seed_id in sorted(perceptual_neighbors):
    if seed_id not in unvisited_ids:
        continue

    pending_ids = [seed_id]
    unvisited_ids.remove(seed_id)
    group_ids = []
    while pending_ids:
        sample_id = pending_ids.pop()
        group_ids.append(sample_id)
        new_ids = set(perceptual_neighbors[sample_id]) & unvisited_ids
        unvisited_ids.difference_update(new_ids)
        pending_ids.extend(new_ids)

    # Each matched pair occurs twice in the undirected neighbor map.
    matched_distances = [
        distance
        for sample_id in group_ids
        for distance in perceptual_neighbors[sample_id].values()
    ]
    group_rows.append(
        {
            "sample_ids": sorted(group_ids),
            "image_count": len(group_ids),
            "candidate_pair_count": len(matched_distances) // 2,
            "min_distance": min(matched_distances),
            "max_distance": max(matched_distances),
        }
    )

perceptual_groups_to_review = pd.DataFrame(
    group_rows,
    columns=[
        "sample_ids", "image_count", "candidate_pair_count",
        "min_distance", "max_distance",
    ],
).sort_values(
    ["max_distance", "min_distance"], kind="stable"
).reset_index(drop=True)
perceptual_group_iterator = enumerate(
    perceptual_groups_to_review.itertuples(index=False), start=1
)
print("Candidate groups queued:", len(perceptual_groups_to_review))
print("Images queued:", int(perceptual_groups_to_review["image_count"].sum()))

# %% [markdown]
# Rerun the following cell to show the next group. Each image appears in one
# review group. The distance range describes matched pairs, rather than every
# possible pair within the group. Review progress stays in memory until the
# setup cell is rerun or the kernel restarts.
# Apply sample tags such as `decision:keep`, `decision:duplicate`, or
# `decision:uncertain` before advancing; selecting a thumbnail alone is temporary.

# %%
review_item = next(perceptual_group_iterator, None)
if review_item is None:
    print("No more candidate groups. Rerun the setup cell to restart the review.")
else:
    group_number, group = review_item
    perceptual_group_view = master.select(group.sample_ids, ordered=True)
    print(f"Group {group_number}/{len(perceptual_groups_to_review)}")
    print("Images:", group.image_count)
    print("Candidate pairs within group:", group.candidate_pair_count)
    print("Matched-pair distance range:", group.min_distance, "to", group.max_distance)
    session.view = perceptual_group_view

# %%
session.show()

# %% [markdown]
# ## 4. Detecting near duplicates (embedding similarity)
#
# Follow `nb01.05`: let FiftyOne compute embeddings with its default pretrained
# model, compare distance thresholds on the same index, and review candidates.

# %% [markdown]
# ### 4.1 Select images without duplicate review tags
#
# Refresh the matching tag list from the master dataset, including any earlier
# embedding-duplicate tags. Exclude each matching sample once. Snapshot the
# excluded IDs so adding review tags does not change this run's comparison view.
# Rerun this cell and the computation cell to refresh the selection.

# %%
embedding_excluded_tags = sorted(
    tag for tag in cast(dict[str, int], master.count_sample_tags())
    if fnmatchcase(tag, "review*duplicate")
)
embedding_excluded_ids = (
    cast(list[str], master.match_tags(embedding_excluded_tags).values("id"))
    if embedding_excluded_tags else []
)
embedding_view = master.exclude(embedding_excluded_ids)
embedding_sample_count = embedding_view.count()

print("Excluded tags:", embedding_excluded_tags)
print("Already duplicate-tagged samples excluded:", len(embedding_excluded_ids))
print("Images selected for embedding comparison:", embedding_sample_count)

# %% [markdown]
# ### 4.2 Compute near duplicates
#
# Compute embeddings only for the selected images. The first run may download
# the default model. Rerunning this cell recomputes embeddings; use the following
# threshold cells to explore results without repeating that work.
# A distance threshold of 0.2 is a starting point for visual review.
# If the kernel's working directory was removed, restore it before model loading.
# Restart the kernel if an earlier attempt failed while importing PyTorch.

# %%
import os

try:
    os.getcwd()
except FileNotFoundError:
    os.chdir(WORKSPACE_ROOT)
    print("Restored kernel working directory to:", WORKSPACE_ROOT)

import fiftyone.brain as fob

EMBEDDING_INITIAL_THRESHOLD = 0.2

embedding_index = None
if embedding_sample_count < 2:
    print("At least two eligible images are required for duplicate comparison.")
else:
    embedding_index = fob.compute_near_duplicates(
        embedding_view, threshold=EMBEDDING_INITIAL_THRESHOLD
    )
    print("Potential duplicate copies:", len(embedding_index.duplicate_ids or []))

# %% [markdown]
# ### 4.3 Compare distance thresholds
#
# Reuse the index to compare thresholds, as in `nb01.05`. Larger distances admit
# less similar images. Counts refer to candidate extra copies, not all images
# in duplicate groups. These candidates still require visual review.
# Restore the initial threshold after the sweep.

# %%
EMBEDDING_THRESHOLDS = (0.02, 0.05, 0.10, 0.20)

embedding_threshold_rows = []
if embedding_index is not None:
    for threshold in EMBEDDING_THRESHOLDS:
        embedding_index.find_duplicates(thresh=threshold)
        embedding_threshold_rows.append(
            (threshold, len(embedding_index.duplicate_ids or []))
        )
    embedding_index.find_duplicates(thresh=EMBEDDING_INITIAL_THRESHOLD)

embedding_threshold_summary = pd.DataFrame(
    embedding_threshold_rows, columns=["distance_threshold", "candidate_copies"]
)
display(embedding_threshold_summary)

# %% [markdown]
# ### 4.4 Review candidates in FiftyOne
#
# Change the review threshold and rerun this cell to inspect another distance
# range without recomputing embeddings. Show reference images alongside copies,
# with group and distance fields available for inspection.
# In the App, tag confirmed extra copies as `review:embedding_duplicate` and
# leave their keepers untagged. Samples and source images are retained.
# Rerun section 3.8 to refresh the combined duplicate-tag summary after review.

# %%
EMBEDDING_REVIEW_THRESHOLD = 0.02

if embedding_index is None:
    print("No embedding index. Check the eligible image count in section 4.1.")
else:
    embedding_index.find_duplicates(thresh=EMBEDDING_REVIEW_THRESHOLD)
    print("Review distance threshold:", EMBEDDING_REVIEW_THRESHOLD)
    print("Potential duplicate copies:", len(embedding_index.duplicate_ids or []))
    embedding_review_view = embedding_index.duplicates_view(
        type_field="embedding_duplicate_type",
        id_field="embedding_duplicate_group",
        dist_field="embedding_duplicate_distance",
    )
    if embedding_index.duplicate_ids:
        session = fo.launch_app(view=embedding_review_view)
    else:
        print("No candidate duplicates at this threshold.")

# %%
session.freeze()


# %% [markdown]
# ## 5. Deduplicate results
#
# ### 5.1 Summarize duplicate tags
#
# Count sample tags matching `review:*duplicate`. The total counts each sample
# once even when it has multiple matching tags. This summary does not change tags.

# %%
from fnmatch import fnmatchcase

import pandas as pd

duplicate_review_tag_counts = {
    tag: count
    for tag, count in cast(dict[str, int], master.count_sample_tags()).items()
    if fnmatchcase(tag, "review:*duplicate")
}
duplicate_tag_summary = pd.DataFrame(
    sorted(duplicate_review_tag_counts.items()), columns=["tag", "samples"]
)
display(duplicate_tag_summary)

duplicate_sample_count = (
    master.match_tags(list(duplicate_review_tag_counts)).count()
    if duplicate_review_tag_counts else 0
)
total_sample_count = master.count()
duplicate_sample_percent = (
    100 * duplicate_sample_count / total_sample_count if total_sample_count else 0
)
print(
    f"Unique duplicate-tagged samples: {duplicate_sample_count:,} / "
    f"{total_sample_count:,} ({duplicate_sample_percent:.2f}%)"
)
print("Samples without matching tags:", total_sample_count - duplicate_sample_count)

# %% [markdown]
# ### 5.2 Compare source datasets before and after deduplication
#
# Compare each source's imported image count with the count remaining after
# excluding samples tagged `review:*duplicate`. Overlapping tags count once.
# Include the overall reduction; samples and source images are retained.

# %%
source_original_counts = cast(
    dict[str | None, int], master.count_values("source_dataset")
)
retained_view = (
    master.match_tags(list(duplicate_review_tag_counts), bool=False)
    if duplicate_review_tag_counts else master.view()
)
source_retained_counts = cast(
    dict[str | None, int], retained_view.count_values("source_dataset")
)
source_deduplication_summary = pd.DataFrame(
    [
        (
            source if source is not None else "(unknown)",
            count,
            source_retained_counts.get(source, 0),
        )
        for source, count in sorted(
            source_original_counts.items(), key=lambda item: item[0] or ""
        )
    ],
    columns=["source_dataset", "original_images", "after_deduplication"],
)
source_deduplication_summary.loc[len(source_deduplication_summary)] = (
    "Total",
    sum(source_original_counts.values()),
    sum(source_retained_counts.values()),
)
source_deduplication_summary["duplicate_images"] = (
    source_deduplication_summary["original_images"]
    - source_deduplication_summary["after_deduplication"]
)
source_deduplication_summary["reduction_percent"] = (
    100 * source_deduplication_summary["duplicate_images"]
    / source_deduplication_summary["original_images"].replace(0, float("nan"))
).fillna(0).round(2)
display(source_deduplication_summary)
