"""
Check quality ranking and direct-match duplicate assignments.
"""

import pytest

from dataset_builder.deduplication import select_perceptual_duplicates


@pytest.mark.parametrize(
    ("quality", "keeper", "duplicate"),
    [
        ({"A": (100, 99), "B": (200, 1)}, "B", "A"),
        ({"A": (200, 1), "B": (200, 3)}, "B", "A"),
        ({"B": (200, 3), "A": (200, 3)}, "B", "A"),
        ({"A": (0, 1), "B": (0, 3)}, "B", "A"),
    ],
)
def test_quality_ranking_uses_resolution_labels_and_first_order(
    quality: dict[str, tuple[int, int]], keeper: str, duplicate: str
) -> None:
    """
    Apply each level of the quality ranking without relying on pair ordering.
    """
    keepers, assignments = select_perceptual_duplicates([("A", "B", 4)], quality)
    assert keepers == [keeper]
    assert assignments == {duplicate: (keeper, 4)}


def test_indirect_chain_retains_images_outside_the_keeper_threshold() -> None:
    """
    Keep a distinct endpoint instead of inferring a missing direct match.
    """
    pairs = [("A", "B", 4), ("B", "C", 4)]
    quality = {"A": (300, 1), "B": (200, 1), "C": (100, 1)}
    keepers, assignments = select_perceptual_duplicates(iter(pairs), quality)
    assert keepers == ["A", "C"]
    assert assignments == {"B": ("A", 4)}


def test_best_central_image_can_represent_all_direct_neighbors() -> None:
    """
    Retain one image when the highest-quality image matches both endpoints.
    """
    pairs = [("A", "B", 4), ("B", "C", 4)]
    quality = {"A": (200, 1), "B": (300, 1), "C": (100, 1)}
    keepers, assignments = select_perceptual_duplicates(pairs, quality)
    assert keepers == ["B"]
    assert assignments == {"A": ("B", 4), "C": ("B", 4)}


def test_threshold_ignores_far_pairs_and_self_matches() -> None:
    """
    Include exact boundary matches and leave unrelated images unassigned.
    """
    quality = {"A": (300, 1), "B": (200, 1), "C": (100, 1)}
    pairs = [("A", "A", 0), ("A", "B", 0), ("B", "C", 1)]
    keepers, assignments = select_perceptual_duplicates(pairs, quality, max_distance=0)
    assert keepers == ["A"]
    assert assignments == {"B": ("A", 0)}


def test_repeated_edges_do_not_repeat_duplicate_assignments() -> None:
    """
    Treat duplicate or reversed edges as one direct candidate relationship.
    """
    quality = {"A": (300, 1), "B": (200, 1)}
    pairs = [("A", "B", 4), ("B", "A", 2), ("A", "B", 4)]
    assert select_perceptual_duplicates(pairs, quality) == (
        ["A"], {"B": ("A", 2)}
    )


def test_empty_candidates_produce_no_tagging_decisions() -> None:
    """
    Handle an empty review queue without selecting unrelated samples.
    """
    assert select_perceptual_duplicates([], {"A": (100, 1)}) == ([], {})


def test_missing_quality_is_rejected_before_tagging() -> None:
    """
    Refuse incomplete quality data instead of choosing an arbitrary keeper.
    """
    with pytest.raises(ValueError, match="quality information"):
        select_perceptual_duplicates([("A", "B", 0)], {"A": (100, 1)})


def test_negative_threshold_is_rejected() -> None:
    """
    Reject invalid threshold configuration before making any assignments.
    """
    with pytest.raises(ValueError, match="threshold"):
        select_perceptual_duplicates([], {}, max_distance=-1)


def test_negative_candidate_distance_is_rejected() -> None:
    """
    Reject invalid pair data before producing duplicate assignments.
    """
    with pytest.raises(ValueError, match="Candidate distances"):
        select_perceptual_duplicates([("A", "B", -1)], {})
