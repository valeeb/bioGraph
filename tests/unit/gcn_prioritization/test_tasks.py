import pytest

pytest.importorskip("torch")

from bioGraph.gcn_prioritization.data import prepare_graph  # noqa: E402
from bioGraph.gcn_prioritization.tasks import (  # noqa: E402
    build_disease_tasks,
    comparison_gene_pool,
    validate_outer_split,
)


def test_validate_outer_split_rejects_overlap():
    with pytest.raises(ValueError, match="disjoint"):
        validate_outer_split(
            {"train_genes": [1, 2], "test_genes": [2, 3]},
            known_genes=[1, 2, 3],
        )


def test_build_tasks_preserves_outer_split_without_a_filtered_negative_pool(
    small_graph, disease_genes, disease_outer_splits
):
    data = prepare_graph(small_graph)
    tasks = build_disease_tasks(
        data, disease_genes, 0.75, 42, disease_outer_splits
    )

    for disease_name, split in disease_outer_splits.items():
        assert "negative_pool" not in tasks[disease_name]
        assert tasks[disease_name]["train_genes"] == split["train_genes"]
        assert tasks[disease_name]["test_genes"] == split["test_genes"]


def test_comparison_pool_uses_only_inner_training_information(small_graph):
    data = prepare_graph(small_graph)

    pool = comparison_gene_pool(data, seed_genes=[1], positive_genes=[2])
    pool_genes = {data.nodelist[index] for index in pool}

    assert pool_genes == {3, 4, 5, 6}
    # Gene 4 can be an outer-test positive, but the pool constructor has no
    # access to that information and therefore treats it like any other node.
    assert 4 in pool_genes
