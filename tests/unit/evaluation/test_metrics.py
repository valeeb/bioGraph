"""Unit tests for disease-gene ranking metrics."""

import pytest

from bioGraph.evaluation.metrics import (
    average_precision_at_k,
    compute_ranking_metrics,
    f1_score_at_k,
    mean_reciprocal_rank_at_k,
    recall_at_k,
)


@pytest.fixture
def ranking(ranking_factory):
    """Five candidates ordered as gene IDs 1, 2, 3, 4, 5."""

    return ranking_factory({1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2})


def test_average_precision_uses_precision_at_each_relevant_rank(ranking):
    assert average_precision_at_k(ranking, {2, 4}) == pytest.approx(0.5)


def test_average_precision_respects_cutoff_and_all_relevant_denominator(ranking):
    assert average_precision_at_k(ranking, {2, 4}, k=3) == pytest.approx(0.25)


def test_average_precision_is_zero_when_cutoff_contains_no_hits(ranking):
    assert average_precision_at_k(ranking, {4}, k=3) == 0.0


def test_recall_counts_fraction_of_all_relevant_genes_retrieved(ranking):
    assert recall_at_k(ranking, {2, 4}, k=3) == pytest.approx(0.5)
    assert recall_at_k(ranking, {2, 4}) == pytest.approx(1.0)


def test_mean_reciprocal_rank_uses_first_relevant_position(ranking):
    assert mean_reciprocal_rank_at_k(ranking, {2, 4}) == pytest.approx(0.5)


def test_mean_reciprocal_rank_is_zero_when_first_hit_is_beyond_cutoff(ranking):
    assert mean_reciprocal_rank_at_k(ranking, {2, 4}, k=1) == 0.0


def test_f1_combines_precision_at_k_and_recall(ranking):
    # At k=4, precision is 2/4 and recall is 2/2.
    assert f1_score_at_k(ranking, {2, 4}, k=4) == pytest.approx(2.0 / 3.0)


def test_f1_is_zero_without_a_true_positive(ranking):
    assert f1_score_at_k(ranking, {4}, k=3) == 0.0


@pytest.mark.parametrize(
    "metric",
    [average_precision_at_k, recall_at_k, mean_reciprocal_rank_at_k, f1_score_at_k],
)
@pytest.mark.parametrize(
    "ranking_value, relevant", [([], {1}), ([{"gene_id": 1}], set())]
)
def test_individual_metrics_are_zero_for_unavailable_inputs(
    metric, ranking_value, relevant
):
    assert metric(ranking_value, relevant) == 0.0


@pytest.mark.parametrize(
    "metric",
    [
        average_precision_at_k,
        recall_at_k,
        mean_reciprocal_rank_at_k,
        f1_score_at_k,
        compute_ranking_metrics,
    ],
)
@pytest.mark.parametrize("invalid_k", [0, -1])
def test_metrics_reject_nonpositive_cutoffs(metric, invalid_k, ranking):
    with pytest.raises(ValueError, match="positive integer"):
        metric(ranking, {2, 4}, k=invalid_k)


def test_cutoff_larger_than_ranking_uses_complete_ranking(ranking):
    assert recall_at_k(ranking, {2, 4}, k=100) == recall_at_k(ranking, {2, 4})
    assert average_precision_at_k(ranking, {2, 4}, k=100) == average_precision_at_k(
        ranking, {2, 4}
    )


def test_compute_ranking_metrics_returns_expected_full_result(ranking):
    result = compute_ranking_metrics(ranking, {2, 4})

    assert result == pytest.approx(
        {
            "average_precision": 0.5,
            "recall": 1.0,
            "f1_score": 4.0 / 7.0,
            "roc_auc": 0.5,
            "precision_recall_auc": 1.0 / 3.0,
        }
    )


def test_compute_ranking_metrics_matches_individual_metrics_at_cutoff(ranking):
    result = compute_ranking_metrics(ranking, {2, 4}, k=3)

    assert result["average_precision"] == pytest.approx(
        average_precision_at_k(ranking, {2, 4}, k=3)
    )
    assert result["recall"] == pytest.approx(recall_at_k(ranking, {2, 4}, k=3))
    assert result["f1_score"] == pytest.approx(f1_score_at_k(ranking, {2, 4}, k=3))


@pytest.mark.parametrize(
    "ranking_value, relevant", [([], {1}), ([{"gene_id": 1}], set())]
)
def test_compute_ranking_metrics_returns_zeroes_for_unavailable_inputs(
    ranking_value, relevant
):
    assert compute_ranking_metrics(ranking_value, relevant) == {
        "average_precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "roc_auc": 0.0,
        "precision_recall_auc": 0.0,
    }


@pytest.mark.parametrize(
    "relevant",
    [
        {1, 2, 3, 4, 5},  # all observed candidates are positive
        {99},  # all observed candidates are negative
    ],
)
def test_roc_auc_is_zero_when_only_one_class_is_observed(ranking, relevant):
    assert compute_ranking_metrics(ranking, relevant)["roc_auc"] == 0.0


def test_repeated_relevant_ids_are_counted_once(ranking):
    assert recall_at_k(ranking, [2, 2, 4]) == pytest.approx(1.0)
