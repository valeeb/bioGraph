from tests.helpers.assertions import (
    assert_metrics_close,
    assert_same_ranked_genes,
    assert_same_top_genes,
)


def test_ranking_assertions_accept_equivalent_outcomes(ranking_factory):
    first = ranking_factory({1: 0.9, 2: 0.8, 3: 0.1})
    second = ranking_factory({1: 9.0, 2: 8.0, 3: 1.0})

    assert_same_ranked_genes(first, second)
    assert_same_top_genes(first, list(reversed(second[:2])) + second[2:], top_k=2)


def test_metric_assertion_accepts_small_numerical_difference():
    assert_metrics_close(
        {"recall": 0.5, "average_precision": 0.75},
        {"recall": 0.5 + 1e-10, "average_precision": 0.75},
    )
