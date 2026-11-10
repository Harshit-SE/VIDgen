"""Reciprocal Rank Fusion (RRF) for combining multi-modal search results.

Combines ranked result lists from diverse modalities (such as visual search via CLIP
and spoken text search via MiniLM) without relying on raw score calibration.

Why RRF is used instead of weighted score averaging:
1. Scale-invariant: CLIP cosine similarities and MiniLM cosine similarities follow
   different distributions; averaging raw scores introduces bias towards whichever
   model outputs higher numerical values.
2. Zero hyperparameter tuning: Rank positions (1st, 2nd, 3rd) are standardized and
   require no weighting multipliers.
3. Robustness: Items that appear near the top of both lists receive a massive boost,
   while items appearing in only one list still get preserved fairly.

Formula:
    RRF_score(item) = SUM( 1 / (k + rank_i) )
    where rank_i is the 1-based position in list i, and k is a smoothing constant (default 60).
"""

from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Combine multiple ranked lists of (item_id, score) tuples into a unified ranking.

    Args:
        ranked_lists: A sequence of ranked lists. Each list contains (item_id, score)
            tuples sorted in descending order of relevance (rank 1 is highest relevance).
        k: Smoothing constant to dampen the impact of high-ranking outliers
            (standard default is 60 from Cormack, Clarke, and Buettcher, 2009).

    Returns:
        Combined list of (item_id, fused_score) tuples sorted in descending order of fused_score.
        If no inputs or all empty lists are supplied, returns an empty list.
    """
    if not ranked_lists:
        return []

    # Dictionary accumulating reciprocal rank scores for each unique item_id
    rrf_scores: dict[int, float] = {}

    for ranked_list in ranked_lists:
        # Enumerate with 1-based rank (rank 1, 2, 3...)
        for rank, (item_id, _original_score) in enumerate(ranked_list, start=1):
            # Reciprocal rank formula: 1 / (k + rank)
            reciprocal_rank = 1.0 / (k + rank)
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + reciprocal_rank

    # Sort items by accumulated RRF score descending (highest score first)
    return sorted(rrf_scores.items(), key=lambda entry: entry[1], reverse=True)

