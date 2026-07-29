import pytest

pytest.importorskip("torch")

from bioGraph.gcn_prioritization.training import train_single_disease  # noqa: E402


def test_single_disease_training_returns_a_complete_ranking(small_graph):
    result = train_single_disease(
        small_graph,
        [1, 2, 4],
        hidden_dim=4,
        epochs=1,
        negative_ratio=1,
        seed_fraction=0.34,
        training_target_fraction=0.33,
        seed=7,
    )

    assert len(result["losses"]) == 1
    assert result["ranking"]
    assert set(result["visible_seed_genes"]).isdisjoint(result["test_genes"])
