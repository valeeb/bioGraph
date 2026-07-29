import pytest

torch = pytest.importorskip("torch")

from bioGraph.gcn_prioritization.model import (  # noqa: E402
    GCN,
    make_features,
    prepare_graph,
)


def test_prepare_graph_features_and_model_output_have_expected_shapes(small_graph):
    data = prepare_graph(small_graph)
    features = make_features(data, visible_seed_genes=[1, 2])
    model = GCN(hidden_dim=4, dropout=0.0)

    output = model(features, data.adjacency)

    assert data.adjacency.shape == (6, 6)
    assert features.shape == (6, 2)
    assert output.shape == (6,)
