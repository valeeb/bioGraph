"""Equivalence contracts between aggregate and dedicated metric functions."""

import pytest

from bioGraph.evaluation.metrics import (
    average_precision_at_k,
    compute_ranking_metrics,
    recall_at_k,
)


@pytest.mark.parametrize("k", [None, 1, 3, 5, 100])
@pytest.mark.parametrize(
    "relevant_genes",
    [
        {2, 4},
        {1, 3, 5},
        {5},
        {99},
        set(),
    ],
)
def test_aggregate_ap_and_recall_equal_dedicated_metrics(
    ranking_factory, relevant_genes, k
):
    """Aggregate AP/recall must use exactly the dedicated metric semantics."""

    ranking = ranking_factory({1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2})

    aggregate = compute_ranking_metrics(ranking, relevant_genes, k=k)

    assert aggregate["average_precision"] == pytest.approx(
        average_precision_at_k(ranking, relevant_genes, k=k)
    )
    assert aggregate["recall"] == pytest.approx(
        recall_at_k(ranking, relevant_genes, k=k)
    )


@pytest.mark.parametrize("k", [None, 1, 10])
def test_empty_ranking_metric_implementations_are_equivalent(k):
    aggregate = compute_ranking_metrics([], {1, 2}, k=k)

    assert aggregate["average_precision"] == average_precision_at_k([], {1, 2}, k=k)
    assert aggregate["recall"] == recall_at_k([], {1, 2}, k=k)
