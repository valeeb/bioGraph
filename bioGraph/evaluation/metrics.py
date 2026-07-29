from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


def compute_ranking_metrics(
    ranking: Sequence[dict], testing_set: Iterable, k: Optional[int] = None,
) -> dict[str, float]:
    """Compute the main binary-ranking metrics from one relevance vector.

    The ranking must be ordered from highest to lowest score and contain a
    ``gene_id`` in every row. ``testing_set`` contains the relevant gene IDs.
    The returned metrics are average precision, recall, F1, ROC-AUC, and area
    under the precision-recall curve (PR-AUC).

    When ``k`` is supplied, all calculations use only the first ``k`` ranking
    rows. Recall and AP still use the complete testing-set size as their
    denominator, matching :func:`recall_at_k` and
    :func:`average_precision_at_k`. ROC-AUC and PR-AUC are most meaningful when
    called on the complete candidate ranking.
    """

    if k is not None and k <= 0:
        raise ValueError("k must be a positive integer.")
    relevant = set(testing_set)
    if not ranking or not relevant:
        return {
            "average_precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.0,
            "precision_recall_auc": 0.0,
        }

    cutoff = len(ranking) if k is None else min(k, len(ranking))
    labels = np.fromiter(
        (row["gene_id"] in relevant for row in ranking[:cutoff]),
        dtype=np.float64,
        count=cutoff,
    )
    ranks = np.arange(1, cutoff + 1, dtype=np.float64)
    cumulative_hits = np.cumsum(labels)
    precision_curve = cumulative_hits / ranks
    recall_curve = cumulative_hits / len(relevant)
    true_positives = float(cumulative_hits[-1])

    average_precision = float(
        np.sum(precision_curve * labels) / min(len(relevant), cutoff)
    )
    recall = true_positives / len(relevant)
    precision_at_k = true_positives / cutoff
    f1 = (
        0.0
        if true_positives == 0.0
        else 2.0 * precision_at_k * recall / (precision_at_k + recall)
    )

    # ROC curve: cumulative true- and false-positive rates. AUC is undefined
    # with only one observed class; return 0.0 consistently with this module's
    # behavior for unavailable ranking metrics.
    observed_positives = int(labels.sum())
    observed_negatives = cutoff - observed_positives
    if observed_positives and observed_negatives:
        false_positives = np.cumsum(1.0 - labels)
        true_positive_rate = cumulative_hits / observed_positives
        false_positive_rate = false_positives / observed_negatives
        roc_auc = float(
            np.trapz(
                np.concatenate(([0.0], true_positive_rate)),
                np.concatenate(([0.0], false_positive_rate)),
            )
        )
    else:
        roc_auc = 0.0

    # Trapezoidal PR-AUC. The curve begins at recall=0, precision=1.
    precision_recall_auc = float(
        np.trapz(
            np.concatenate(([1.0], precision_curve)),
            np.concatenate(([0.0], recall_curve)),
        )
    )

    return {
        "average_precision": average_precision,
        "recall": recall,
        "f1_score": float(f1),
        "roc_auc": roc_auc,
        "precision_recall_auc": precision_recall_auc,
    }


def average_precision_at_k(
    ranking: Sequence[dict], testing_set: Iterable, k: Optional[int] = None
) -> float:
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


def recall_at_k(
    ranking: Sequence[dict], testing_set: Iterable, k: Optional[int] = None
) -> float:
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


def mean_reciprocal_rank_at_k(
    ranking: Sequence[dict], testing_set: Iterable, k: Optional[int] = None
) -> float:
    """Return the reciprocal rank of the first relevant gene within the top k."""

    if not ranking:
        return 0.0
    relevant = set(testing_set)
    if not relevant:
        return 0.0
    if k is None:
        k = len(ranking)
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    for rank, row in enumerate(ranking[:k], start=1):
        if row["gene_id"] in relevant:
            return 1.0 / rank
    return 0.0


def f1_score_at_k(
    ranking: Sequence[dict], testing_set: Iterable, k: Optional[int] = None
) -> float:
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
