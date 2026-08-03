import numpy as np
import pytest

from bioGraph.data.splitting import (
    split_disease_genes,
    split_disease_genes_three_way,
    split_known_genes,
    split_training_genes,
)


def test_outer_and_inner_splits_are_complete_and_disjoint():
    positives = list(range(1, 13))
    outer = split_known_genes(positives, random_state=7)
    inner = split_training_genes(outer["train_genes"], random_state=8)

    assert set(outer["train_genes"]).isdisjoint(outer["test_genes"])
    assert set(outer["train_genes"]) | set(outer["test_genes"]) == set(positives)
    assert set(inner["seed_genes"]).isdisjoint(inner["label_genes"])
    assert set(inner["seed_genes"]) | set(inner["label_genes"]) == set(
        outer["train_genes"]
    )


def test_split_is_reproducible_and_removes_duplicates():
    first = split_known_genes([4, 1, 2, 2, 3, 1], 0.5, random_state=9)
    second = split_known_genes({1, 2, 3, 4}, 0.5, random_state=9)

    assert first == second
    assert len(first["train_genes"] + first["test_genes"]) == 4


def test_split_accepts_generator_and_existing_rng():
    rng = np.random.default_rng(4)
    split = split_known_genes((gene for gene in range(5)), 0.6, random_state=rng)

    assert len(split["train_genes"]) == 3
    assert len(split["test_genes"]) == 2


@pytest.mark.parametrize("genes", [[], [1], [1, 1]])
def test_split_requires_two_unique_genes(genes):
    with pytest.raises(ValueError, match="At least two unique"):
        split_known_genes(genes)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, np.nan, np.inf])
def test_split_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="between zero and one"):
        split_known_genes([1, 2, 3], fraction)


def test_compatibility_split_wrappers_partition_genes():
    seeds, labels, tests = split_disease_genes_three_way(
        range(1, 7), 0.5, 0.25, random_state=3
    )
    train, test = split_disease_genes(
        "disease", 0.5, {"disease": [1, 2, 3, 4]}, random_state=3
    )

    assert set(seeds) | set(labels) | set(tests) == set(range(1, 7))
    assert set(seeds).isdisjoint(labels)
    assert set(train) | set(test) == {1, 2, 3, 4}


def test_compatibility_split_rejects_unknown_disease():
    with pytest.raises(ValueError, match="not found"):
        split_disease_genes("missing", 0.5, {})
