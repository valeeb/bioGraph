"""Disease task construction and comparison-pool utilities."""

from typing import Mapping, Sequence

import numpy as np

from bioGraph.data.splitting import split_known_genes
from bioGraph.gcn_prioritization.data import GraphData


def validate_outer_split(
    split: Mapping[str, Sequence[int]], known_genes: Sequence[int]
) -> None:
    """Require a nonempty, disjoint partition of the known graph genes."""

    if set(split) != {"train_genes", "test_genes"}:
        raise ValueError("outer_split must contain exactly train_genes and test_genes.")
    train = set(split["train_genes"])
    test = set(split["test_genes"])
    if not train or not test:
        raise ValueError("outer_split train and test subsets must both be nonempty.")
    if train & test:
        raise ValueError("outer_split train and test subsets must be disjoint.")
    if train | test != set(known_genes):
        raise ValueError("outer_split must partition exactly the known genes in the graph.")


def build_disease_tasks(
    data: GraphData,
    diseases: Mapping[str, Sequence[int]],
    train_fraction: float,
    seed: int,
    outer_splits: Mapping[str, Mapping[str, Sequence[int]]] | None,
) -> dict[str, dict]:
    """Build tasks without exposing held-out disease associations to training."""

    graph_genes = set(data.nodelist)
    tasks: dict[str, dict] = {}
    for disease_name in sorted(diseases):
        supplied_genes = set(diseases[disease_name])
        known_genes = sorted(supplied_genes & graph_genes)
        split = (
            outer_splits[disease_name]
            if outer_splits is not None and disease_name in outer_splits
            else split_known_genes(known_genes, train_fraction, random_state=seed)
        )
        validate_outer_split(split, known_genes)

        tasks[disease_name] = {
            "train_genes": list(split["train_genes"]),
            "test_genes": list(split["test_genes"]),
            "known_in_graph": len(known_genes),
            "known_not_in_graph": len(supplied_genes - graph_genes),
        }
    return tasks


def comparison_gene_pool(
    data: GraphData,
    seed_genes: Sequence[int],
    positive_genes: Sequence[int],
) -> np.ndarray:
    """Return candidate indices using only the current training sample.

    Genes outside the inner seed and positive sets are indistinguishable here.
    In particular, this function receives no outer-test set, so held-out genes
    remain eligible as unlabelled comparison genes.
    """

    excluded = {int(gene) for gene in seed_genes}
    excluded.update(int(gene) for gene in positive_genes)
    return np.asarray(
        [data.node_to_index[gene] for gene in data.nodelist if gene not in excluded],
        dtype=np.int64,
    )
