from typing import Iterable, List, Sequence

import numpy as np
from scipy import sparse


def disease_relevance_weighted_adjacency(
    graph, disease_sets, beta: float, nodelist=None
):
    """Return a sparse adjacency weighted by shared disease memberships.

    An edge's relevance score is the number of supplied disease sets containing
    both endpoints, and its weight is ``score ** beta``. Callers must supply only
    the genes visible for the target disease (for example, the current outer
    training split); other disease sets may be complete. At ``beta=0``, every
    existing edge has weight one, including zero-score edges via ``0**0 == 1``.
    """

    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative.")
    if nodelist is None:
        nodelist = list(graph.nodes())
    if len(nodelist) != graph.number_of_nodes() or set(nodelist) != set(graph):
        raise ValueError("nodelist must contain every graph node exactly once.")

    graph_nodes = set(graph)
    memberships = {}
    for disease_name, genes in disease_sets.items():
        for gene in set(genes) & graph_nodes:
            memberships.setdefault(gene, set()).add(disease_name)

    node_to_index = {node: index for index, node in enumerate(nodelist)}
    rows, columns, weights = [], [], []
    for gene_a, gene_b in graph.edges():
        score = len(
            memberships.get(gene_a, set())
            & memberships.get(gene_b, set())
        )
        weight = float(score ** beta)
        if weight == 0.0:
            continue
        index_a, index_b = node_to_index[gene_a], node_to_index[gene_b]
        rows.extend((index_a, index_b))
        columns.extend((index_b, index_a))
        weights.extend((weight, weight))

    return sparse.csr_matrix(
        (weights, (rows, columns)),
        shape=(len(nodelist), len(nodelist)),
        dtype=float,
    )


def scores_to_ranking(
    scores: Sequence[float],
    nodelist: Sequence,
    graph,
    seed_genes: Iterable,
    include_seed_genes: bool = False,
) -> List[dict]:
    """Convert a score vector into sorted ranking rows.

    Each row has keys: gene_id, symbol, score.
    Scores are sorted descending, then by gene_id for deterministic output.
    """
    seed_set = set(seed_genes)
    ranking = []

    for node, score in zip(nodelist, scores):
        if not include_seed_genes and node in seed_set:
            continue
        ranking.append(
            {
                "gene_id": node,
                "symbol": graph.nodes[node].get("symbol", ""),
                "score": float(score),
            }
        )

    ranking.sort(key=lambda row: (-row["score"], row["gene_id"]))
    return ranking
