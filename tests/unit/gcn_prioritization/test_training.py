import pytest

pytest.importorskip("torch")

from bioGraph.data.splitting import split_known_genes  # noqa: E402
from bioGraph.gcn_prioritization.model import DiseaseConditionedGCN  # noqa: E402
from bioGraph.gcn_prioritization.training import (  # noqa: E402
    evaluate_all_diseases,
    train_all_diseases,
    train_single_disease,
)


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


def test_all_diseases_jointly_train_one_conditioned_model_without_test_leakage(
    small_graph, disease_genes, disease_outer_splits,
):
    trained = train_all_diseases(
        small_graph,
        disease_genes,
        epochs=1,
        hidden_dim=4,
        disease_embedding_dim=3,
        negative_ratio=1,
        outer_splits=disease_outer_splits,
        verbose=False,
    )

    assert isinstance(trained["model"], DiseaseConditionedGCN)
    assert "disease_results" not in trained
    result = evaluate_all_diseases(trained)
    assert set(result) == {
        "model",
        "disease_to_id",
        "graph_data",
        "disease_results",
        "losses",
        "device",
    }
    assert result["disease_to_id"] == {"disease_a": 0, "disease_b": 1}
    assert result["model"].disease_embeddings.weight.shape == (2, 3)
    for disease, split in disease_outer_splits.items():
        disease_result = result["disease_results"][disease]
        assert disease_result["train_genes"] == split["train_genes"]
        assert disease_result["test_genes"] == split["test_genes"]
        assert disease_result["ranking"]
        assert "scores" not in disease_result
        assert "metrics" not in disease_result


def test_each_training_run_initializes_a_new_model_for_its_outer_splits(
    small_graph, disease_genes, disease_outer_splits,
):
    arguments = dict(
        epochs=1,
        hidden_dim=4,
        disease_embedding_dim=3,
        negative_ratio=1,
        outer_splits=disease_outer_splits,
        verbose=False,
    )

    first = train_all_diseases(small_graph, disease_genes, **arguments)
    second = train_all_diseases(small_graph, disease_genes, **arguments)

    assert first["model"] is not second["model"]
    assert next(first["model"].parameters()).data_ptr() != next(
        second["model"].parameters()
    ).data_ptr()


def test_evaluation_is_repeatable_and_does_not_mutate_training_state(
    small_graph, disease_genes, disease_outer_splits,
):
    trained = train_all_diseases(
        small_graph, disease_genes, epochs=1, hidden_dim=4,
        disease_embedding_dim=3, negative_ratio=1,
        outer_splits=disease_outer_splits, verbose=False,
    )

    first = evaluate_all_diseases(trained)
    second = evaluate_all_diseases(trained)

    assert first["disease_results"] == second["disease_results"]
    assert "disease_results" not in trained
