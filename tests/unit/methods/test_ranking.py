import networkx as nx
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from bioGraph.methods.ranking import (
    diamond_score,
    dk_score,
    neighbourhood_score,
    normalize_adjacency,
    qa_score,
    rwr_score,
)


@pytest.fixture
def graph_matrices(small_graph):
    nodelist = sorted(small_graph.nodes)
    dense = nx.to_numpy_array(small_graph, nodelist=nodelist)
    return nodelist, dense, csr_matrix(dense)


def test_neighbourhood_score_counts_seed_neighbours(small_graph, graph_matrices):
    nodelist, dense, _ = graph_matrices

    scores = neighbourhood_score(
        small_graph, [1], dense, nodelist=nodelist, weighted="absolute"
    )

    np.testing.assert_array_equal(scores, [0, 1, 1, 0, 0, 0])


def test_normalize_adjacency_preserves_shape_and_symmetry(small_graph, graph_matrices):
    nodelist, _, sparse = graph_matrices

    normalized = normalize_adjacency(small_graph, sparse, nodelist=nodelist)

    assert normalized.shape == (6, 6)
    np.testing.assert_allclose(normalized.toarray(), normalized.toarray().T)


def test_diffusion_kernel_accepts_precomputed_transition_matrix(
    small_graph, graph_matrices
):
    nodelist, _, _ = graph_matrices
    scores = dk_score(small_graph, [1], P=np.eye(6), nodelist=nodelist)

    np.testing.assert_array_equal(scores, [1, 0, 0, 0, 0, 0])


def test_quantum_walk_with_zero_hamiltonian_stays_at_seed(small_graph, graph_matrices):
    nodelist, _, _ = graph_matrices
    scores = qa_score(small_graph, [1], t=1.0, H=csr_matrix((6, 6)), nodelist=nodelist)

    np.testing.assert_array_equal(scores, [1, 0, 0, 0, 0, 0])


def test_random_walk_with_restart_returns_finite_nonseed_scores(
    small_graph, graph_matrices
):
    nodelist, _, sparse = graph_matrices
    normalized = normalize_adjacency(small_graph, sparse, nodelist=nodelist)

    scores = rwr_score(small_graph, [1], normalized, nodelist=nodelist)

    assert scores.shape == (6,)
    assert scores[0] == 0.0
    assert np.all(np.isfinite(scores))
    assert np.any(scores[1:] > 0)


def test_diamond_assigns_highest_score_to_first_selected_candidate(
    small_graph, graph_matrices
):
    nodelist, dense, _ = graph_matrices
    scores = diamond_score(small_graph, [1], dense, number_to_rank=1, nodelist=nodelist)

    assert np.count_nonzero(scores) == 1
    assert scores.max() == 1.0
