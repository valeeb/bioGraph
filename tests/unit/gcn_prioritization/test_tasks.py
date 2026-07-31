import pytest

pytest.importorskip("torch")

from bioGraph.gcn_prioritization.data import prepare_graph  # noqa: E402
from bioGraph.gcn_prioritization.tasks import (  # noqa: E402
    build_disease_tasks,
    validate_outer_split,
)


def test_validate_outer_split_rejects_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        validate_outer_split(
            {"train_genes": [1, 2], "test_genes": [2, 3]},
            known_genes=[1, 2, 3],
        )


def test_build_tasks_never_uses_held_out_genes_as_negatives(
    small_graph, disease_genes
):
    data = prepare_graph(small_graph)
    splits = {
        "disease_a": {"train_genes": [1, 2], "test_genes": [4]},
        "disease_b": {"train_genes": [3, 5], "test_genes": [6]},
    }

    tasks = build_disease_tasks(data, disease_genes, 0.75, 42, splits)

    for disease_name, known_genes in disease_genes.items():
        negative_gene_ids = {
            data.nodelist[index] for index in tasks[disease_name]["negative_pool"]
        }
        assert negative_gene_ids.isdisjoint(known_genes)
        assert tasks[disease_name]["test_genes"] == splits[disease_name]["test_genes"]
