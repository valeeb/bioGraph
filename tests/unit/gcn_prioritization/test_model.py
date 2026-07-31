import pytest

torch = pytest.importorskip("torch")

from bioGraph.gcn_prioritization.model import (  # noqa: E402
    DiseaseConditionedGCN,
    GCN,
)
from bioGraph.gcn_prioritization.data import make_features, prepare_graph  # noqa: E402


def test_single_task_model_output_has_one_score_per_gene(small_graph):
    data = prepare_graph(small_graph)
    features = make_features(data, visible_seed_genes=[1, 2])
    model = GCN(hidden_dim=4, dropout=0.0)

    output = model(features, data.adjacency)

    assert data.adjacency.shape == (6, 6)
    assert features.shape == (6, 2)
    assert output.shape == (6,)


def test_disease_conditioned_model_scores_each_gene_for_each_disease(small_graph):
    data = prepare_graph(small_graph)
    features = torch.stack(
        [make_features(data, [1]), make_features(data, [2])], dim=1
    )
    model = DiseaseConditionedGCN(
        num_diseases=2, hidden_dim=4, disease_embedding_dim=3, dropout=0.0
    )

    output = model(features, data.adjacency, torch.tensor([0, 1]))

    assert output.shape == (6, 2)
    assert model.disease_embeddings.weight.shape == (2, 3)


def test_pairwise_loss_backpropagates_to_disease_embeddings(small_graph):
    data = prepare_graph(small_graph)
    features = make_features(data, [1])
    model = DiseaseConditionedGCN(
        num_diseases=2, hidden_dim=4, disease_embedding_dim=3, dropout=0.0
    )
    scores = model(features, data.adjacency, torch.tensor(0))

    torch.nn.functional.softplus(-(scores[1] - scores[4])).backward()

    assert model.disease_embeddings.weight.grad[0].abs().sum() > 0
