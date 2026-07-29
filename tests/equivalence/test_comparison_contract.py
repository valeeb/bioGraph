"""Examples of explicit equivalence contracts for future method comparisons."""

from tests.helpers.assertions import assert_same_top_genes


def test_top_k_contract_allows_different_order_with_same_candidates(ranking_factory):
    reference = ranking_factory({1: 0.9, 2: 0.8, 3: 0.2, 4: 0.1})
    alternative = ranking_factory({2: 0.95, 1: 0.85, 4: 0.2, 3: 0.1})

    assert_same_top_genes(alternative, reference, top_k=2)
