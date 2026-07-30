"""Examples of explicit equivalence contracts for future method comparisons."""

import pytest

from bioGraph.data.splitting import split_disease_genes, split_known_genes
from tests.helpers.assertions import assert_same_top_genes


def test_top_k_contract_allows_different_order_with_same_candidates(ranking_factory):
    reference = ranking_factory({1: 0.9, 2: 0.8, 3: 0.2, 4: 0.1})
    alternative = ranking_factory({2: 0.95, 1: 0.85, 4: 0.2, 3: 0.1})

    assert_same_top_genes(alternative, reference, top_k=2)


@pytest.mark.parametrize("number_of_genes", [2, 3, 4, 7, 12, 23])
def test_gcn_and_deterministic_calls_receive_the_same_outer_split(number_of_genes):

    known_genes = list(range(1, number_of_genes + 1))
    diseases = {"disease": list(reversed(known_genes))}
    random_state = 17

    gcn_split = split_known_genes(known_genes, 0.75, random_state=random_state)
    deterministic_training, deterministic_test = split_disease_genes(
        "disease",
        split_fraction=0.50 + 0.25,
        diseases_dict=diseases,
        random_state=random_state,
    )

    assert set(deterministic_training) == set(gcn_split["train_genes"])
    assert set(deterministic_test) == set(gcn_split["test_genes"])
