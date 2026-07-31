from bioGraph.data.splitting import split_known_genes, split_training_genes


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
