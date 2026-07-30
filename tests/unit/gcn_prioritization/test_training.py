import pytest

pytest.importorskip("torch")

from bioGraph.data.splitting import split_known_genes  # noqa: E402
from bioGraph.gcn_prioritization.training import train_single_disease  # noqa: E402


def test_single_disease_training_returns_a_complete_ranking(small_graph):
    result = train_single_disease(
        small_graph,
        [1, 2, 4],
        hidden_dim=4,
        epochs=1,
        negative_ratio=1,
        train_fraction=0.67,
        seed=7,
    )

    assert len(result["losses"]) == 1
    assert result["ranking"]
    assert set(result["visible_seed_genes"]).isdisjoint(result["test_genes"])


def test_gcn_consumes_the_exact_shared_split(small_graph):
    split = split_known_genes([1, 2, 4], train_fraction=0.67, random_state=7)
    result = train_single_disease(
        small_graph, [1, 2, 4], hidden_dim=4, epochs=1, negative_ratio=1,
        seed=7, outer_split=split,
    )

    assert result["train_genes"] == split["train_genes"]
    assert result["visible_seed_genes"] == split["train_genes"]
    assert result["test_genes"] == split["test_genes"]
    assert len(result["inner_splits"]) == 1
    inner = result["inner_splits"][0]
    assert set(inner["seed_genes"]) | set(inner["label_genes"]) == set(
        split["train_genes"]
    )
