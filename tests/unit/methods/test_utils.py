import networkx as nx
import numpy as np

from bioGraph.methods.utils import (
    disease_edge_counts_for_split,
    disease_relevance_weighted_adjacency,
    precompute_disease_edge_counts,
    scores_to_ranking,
)


def test_precomputed_counts_replace_target_with_training_memberships():
    graph = nx.path_graph([0, 1, 2, 3])
    diseases = {"target": [0, 1, 2], "other": [1, 2, 3]}
    counts = precompute_disease_edge_counts(graph, diseases)

    adjusted = disease_edge_counts_for_split(
        graph, counts, diseases["target"], [0, 1]
    )

    np.testing.assert_array_equal(
        adjusted.toarray(),
        np.asarray([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]),
    )


def test_scores_to_ranking_sorts_candidates_and_excludes_seeds(small_graph):
    ranking = scores_to_ranking(
        [0.1, 0.9, 0.5, 0.5, 0.2, 0.0], [1, 2, 3, 4, 5, 6], small_graph, seed_genes={2},
    )

    assert [row["gene_id"] for row in ranking] == [3, 4, 5, 1, 6]
    assert ranking[0] == {"gene_id": 3, "symbol": "GENE3", "score": 0.5}


def test_disease_relevance_weighted_adjacency_uses_visible_memberships():
    graph = nx.path_graph([0, 1, 2, 3])
    nodelist = [0, 1, 2, 3]
    # The target disease exposes only its current training gene 0. The complete
    # other disease makes edges 0-1 and 1-2 score one; edge 2-3 scores zero.
    # The +1 baseline keeps the zero-score edge and gives it weight one.
    visible_diseases = {"target": [0], "other": [0, 1, 2]}

    weighted = disease_relevance_weighted_adjacency(
        graph, visible_diseases, beta=1.0, nodelist=nodelist
    )

    np.testing.assert_array_equal(
        weighted.toarray(),
        np.asarray(
            [
                [0, 2, 0, 0],
                [2, 0, 2, 0],
                [0, 2, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        ),
    )


def test_disease_relevance_weighted_adjacency_beta_zero_is_unweighted():
    graph = nx.path_graph([0, 1, 2, 3])
    weighted = disease_relevance_weighted_adjacency(
        graph, {"disease": [0, 1]}, beta=0.0
    )

    np.testing.assert_array_equal(weighted.toarray(), nx.to_numpy_array(graph))


def test_disease_relevance_weighted_adjacency_counts_shared_diseases():
    graph = nx.Graph()
    graph.add_edge(0, 1)
    weighted = disease_relevance_weighted_adjacency(
        graph,
        {"disease_a": [0, 1], "disease_b": [0, 1]},
        beta=2.0,
        nodelist=[0, 1],
    )

    assert weighted[0, 1] == 9.0
    assert weighted[1, 0] == 9.0
