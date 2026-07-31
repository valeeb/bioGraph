import pytest

torch = pytest.importorskip("torch")

from bioGraph.gcn_prioritization.data import (  # noqa: E402
    make_features,
    prepare_graph,
)


def test_prepare_graph_builds_normalized_adjacency_and_stable_node_order(small_graph):
    data = prepare_graph(small_graph)

    assert data.nodelist == [1, 2, 3, 4, 5, 6]
    assert data.node_to_index == {gene: gene - 1 for gene in data.nodelist}
    assert data.adjacency.shape == (6, 6)
    assert data.adjacency.is_coalesced()


def test_make_features_marks_only_visible_seed_genes(small_graph):
    data = prepare_graph(small_graph)

    features = make_features(data, visible_seed_genes=[1, 4])

    assert features.shape == (6, 2)
    assert torch.equal(features[:, 0], torch.ones(6))
    assert features[:, 1].tolist() == [1, 0, 0, 1, 0, 0]
