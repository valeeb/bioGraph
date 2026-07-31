import pytest

torch = pytest.importorskip("torch")

from bioGraph.gcn_prioritization.data import prepare_graph  # noqa: E402
from bioGraph.gcn_prioritization.inference import predict_from_seed_genes  # noqa: E402
from bioGraph.gcn_prioritization.model import DiseaseConditionedGCN  # noqa: E402


def test_conditioned_inference_requires_a_disease_id(small_graph):
    data = prepare_graph(small_graph)
    model = DiseaseConditionedGCN(2, hidden_dim=4, dropout=0.0)

    with pytest.raises(ValueError, match="disease_id is required"):
        predict_from_seed_genes(model, data, [1, 2])


def test_conditioned_inference_excludes_visible_seeds(small_graph):
    data = prepare_graph(small_graph)
    model = DiseaseConditionedGCN(2, hidden_dim=4, dropout=0.0)

    ranking = predict_from_seed_genes(model, data, [1, 2], disease_id=0)

    assert len(ranking) == 4
    assert {row["gene_id"] for row in ranking}.isdisjoint({1, 2})
