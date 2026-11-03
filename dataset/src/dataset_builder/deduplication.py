"""
Select quality-ranked keepers from perceptual duplicate candidates.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping


def select_perceptual_duplicates(
    candidate_pairs: Iterable[tuple[str, str, int]],
    sample_quality: Mapping[str, tuple[int, int]],
    max_distance: int = 4,
) -> tuple[list[str], dict[str, tuple[str, int]]]:
    """
    Keep images by pixel area, then label count, with stable input-order ties.

    Return keeper IDs and a mapping from duplicate IDs to (keeper ID, distance).
    Each duplicate directly matches its keeper within ``max_distance``. Indirect
    chains can therefore retain multiple keepers. The mapping's iteration order
    determines which image wins when both quality measures are tied.
    """
    if max_distance < 0:
        raise ValueError("The duplicate distance threshold must be nonnegative.")

    neighbors = defaultdict(dict)
    for sample_a, sample_b, distance in candidate_pairs:
        if distance < 0:
            raise ValueError("Candidate distances must be nonnegative.")
        if sample_a == sample_b or distance > max_distance:
            continue
        if sample_a not in sample_quality or sample_b not in sample_quality:
            raise ValueError("Every candidate image must have quality information.")
        distance = min(distance, neighbors[sample_a].get(sample_b, distance))
        neighbors[sample_a][sample_b] = distance
        neighbors[sample_b][sample_a] = distance

    candidate_ids = [
        sample_id for sample_id in sample_quality if sample_id in neighbors
    ]
    ranked_ids = sorted(candidate_ids, key=sample_quality.__getitem__, reverse=True)
    remaining_ids = set(candidate_ids)
    keeper_ids = []
    assignments = {}
    for keeper_id in ranked_ids:
        if keeper_id not in remaining_ids:
            continue
        keeper_ids.append(keeper_id)
        remaining_ids.remove(keeper_id)
        for duplicate_id, distance in neighbors[keeper_id].items():
            if duplicate_id in remaining_ids:
                assignments[duplicate_id] = (keeper_id, distance)
                remaining_ids.remove(duplicate_id)

    return keeper_ids, assignments
