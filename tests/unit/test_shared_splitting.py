import numpy as np

from bioGraph.data.splitting import split_known_genes, split_training_genes


def test_outer_split_is_reproducible_disjoint_and_complete():
    known = list(range(12))
    first = split_known_genes(known, train_fraction=0.75, random_state=7)
    second = split_known_genes(
        set(reversed(known)), train_fraction=0.75, random_state=7
    )

    assert first == second
    assert set(first) == {"train_genes", "test_genes"}
    assert set(first["train_genes"]).isdisjoint(first["test_genes"])
    assert set(first["train_genes"]) | set(first["test_genes"]) == set(known)
    assert [len(first["train_genes"]), len(first["test_genes"])] == [9, 3]


def test_inner_splits_partition_only_outer_training_genes_and_vary():
    outer = split_known_genes(range(12), random_state=3)
    rng = np.random.default_rng(11)
    inner_splits = [
        split_training_genes(outer["train_genes"], random_state=rng)
        for _ in range(4)
    ]

    for inner in inner_splits:
        assert set(inner) == {"seed_genes", "label_genes"}
        assert set(inner["seed_genes"]).isdisjoint(inner["label_genes"])
        assert set(inner["seed_genes"]) | set(inner["label_genes"]) == set(
            outer["train_genes"]
        )
        assert set(inner["seed_genes"]).isdisjoint(outer["test_genes"])
        assert set(inner["label_genes"]).isdisjoint(outer["test_genes"])
    assert len({tuple(inner["seed_genes"]) for inner in inner_splits}) > 1


def test_outer_split_rejects_invalid_inputs():
    with np.testing.assert_raises_regex(ValueError, "between zero and one"):
        split_known_genes(range(4), train_fraction=1.0)
    with np.testing.assert_raises_regex(ValueError, "At least two"):
        split_known_genes([1])
