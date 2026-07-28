from typing import Iterable, Optional, Sequence


def average_precision_at_k(ranking: Sequence[dict], testing_set: Iterable, k: Optional[int] = None) -> float:
    """Compute average precision at k for a ranking of dict rows."""
    if not ranking:
        return 0.0

    relevant = set(testing_set)
    if not relevant:
        return 0.0

    if k is None:
        k = len(ranking)
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    k = min(k, len(ranking))

    num_hits = 0
    precision_sum = 0.0

    for rank_idx in range(k):
        gene_id = ranking[rank_idx]["gene_id"]
        if gene_id in relevant:
            num_hits += 1
            precision_sum += num_hits / (rank_idx + 1)

    if num_hits == 0:
        return 0.0

    return precision_sum / min(len(relevant), k)


def recall_at_k(ranking: Sequence[dict], testing_set: Iterable, k: Optional[int] = None) -> float:
    """Compute recall at k for a ranking of dict rows."""
    if not ranking:
        return 0.0

    relevant = set(testing_set)
    if not relevant:
        return 0.0

    if k is None:
        k = len(ranking)
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    k = min(k, len(ranking))
    true_positives = sum(1 for i in range(k) if ranking[i]["gene_id"] in relevant)
    return true_positives / len(relevant)


def f1_score_at_k(ranking: Sequence[dict], testing_set: Iterable, k: Optional[int] = None) -> float:
    """Compute F1 score at k for a ranking of dict rows."""
    if not ranking:
        return 0.0

    relevant = set(testing_set)
    if not relevant:
        return 0.0

    if k is None:
        k = len(ranking)
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    k = min(k, len(ranking))
    true_positives = sum(1 for i in range(k) if ranking[i]["gene_id"] in relevant)

    if true_positives == 0:
        return 0.0

    precision_at_k = true_positives / k
    recall_value = true_positives / len(relevant)
    return 2 * precision_at_k * recall_value / (precision_at_k + recall_value)


 


