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


def test_seed_input_can_be_resolved_from_disease_name(small_graph, graph_matrices):
    nodelist, dense, _ = graph_matrices
    direct = neighbourhood_score(
        small_graph, [1], dense, nodelist=nodelist, weighted="absolute"
    )
    named = neighbourhood_score(
        small_graph,
        "disease",
        dense,
        diseases_dict={"disease": [1]},
        nodelist=nodelist,
        weighted="absolute",
    )

    np.testing.assert_array_equal(named, direct)


@pytest.mark.parametrize(
    ("seeds", "diseases", "message"),
    [
        ("disease", None, "diseases_dict must be provided"),
        ("missing", {"disease": [1]}, "not found"),
        ([], None, "Seed list is empty"),
        ([99], None, "None of the seed nodes"),
    ],
)
def test_scoring_rejects_invalid_seed_inputs(
    small_graph, graph_matrices, seeds, diseases, message
):
    nodelist, dense, _ = graph_matrices
    with pytest.raises(ValueError, match=message):
        neighbourhood_score(
            small_graph, seeds, dense, diseases_dict=diseases, nodelist=nodelist
        )


def test_scoring_rejects_nodelist_that_omits_resolved_seeds(
    small_graph, graph_matrices
):
    _, dense, _ = graph_matrices
    with pytest.raises(ValueError, match="provided nodelist"):
        neighbourhood_score(small_graph, [1], dense[1:, 1:], nodelist=[2, 3, 4, 5, 6])


def test_neighbourhood_dense_sparse_and_both_modes_agree(
    small_graph, graph_matrices
):
    nodelist, dense, sparse = graph_matrices
    dense_relative = neighbourhood_score(
        small_graph, [1], dense, nodelist=nodelist, weighted="relative"
    )
    sparse_relative, sparse_absolute = neighbourhood_score(
        small_graph, [1], sparse, nodelist=nodelist, weighted="both"
    )

    np.testing.assert_allclose(sparse_relative, dense_relative)
    np.testing.assert_array_equal(sparse_absolute, [0, 1, 1, 0, 0, 0])


def test_neighbourhood_rejects_unknown_weighting(small_graph, graph_matrices):
    nodelist, dense, _ = graph_matrices
    with pytest.raises(ValueError, match="Invalid weighted value"):
        neighbourhood_score(small_graph, [1], dense, nodelist=nodelist, weighted="bad")


def test_diffusion_kernel_computes_from_laplacian(small_graph, graph_matrices):
    nodelist, _, sparse = graph_matrices
    laplacian = csr_matrix(nx.laplacian_matrix(small_graph, nodelist=nodelist))

    scores = dk_score(small_graph, [1], t=0.0, L=laplacian, nodelist=nodelist)

    np.testing.assert_array_equal(scores, [1, 0, 0, 0, 0, 0])


def test_diffusion_kernel_requires_laplacian_and_time(small_graph):
    with pytest.raises(ValueError, match="L and t must be provided"):
        dk_score(small_graph, [1])


@pytest.mark.parametrize("method", ["qa", "dk", "neighbourhood", "normalize", "rwr", "diamond"])
def test_methods_reject_matrices_with_wrong_shape(small_graph, method):
    nodelist = list(small_graph)
    bad = csr_matrix((5, 5))
    calls = {
        "qa": lambda: qa_score(small_graph, [1], 1.0, bad, nodelist=nodelist),
        "dk": lambda: dk_score(small_graph, [1], P=np.zeros((5, 5)), nodelist=nodelist),
        "neighbourhood": lambda: neighbourhood_score(small_graph, [1], bad, nodelist=nodelist),
        "normalize": lambda: normalize_adjacency(small_graph, bad, nodelist=nodelist),
        "rwr": lambda: rwr_score(small_graph, [1], bad, nodelist=nodelist),
        "diamond": lambda: diamond_score(small_graph, [1], bad, nodelist=nodelist),
    }

    with pytest.raises(ValueError, match="shape"):
        calls[method]()


def test_normalize_adjacency_handles_isolated_node():
    graph = nx.Graph([(1, 2)])
    graph.add_node(3)
    adjacency = csr_matrix(nx.to_numpy_array(graph, nodelist=[1, 2, 3]))

    normalized = normalize_adjacency(graph, adjacency, nodelist=[1, 2, 3])

    np.testing.assert_array_equal(normalized.toarray()[2], [0, 0, 0])


def test_quantum_walk_accepts_seed_diagonal(small_graph, graph_matrices):
    nodelist, _, sparse = graph_matrices
    scores = qa_score(
        small_graph, [1], t=0.0, H=sparse, diag=2.0, nodelist=nodelist
    )

    np.testing.assert_array_equal(scores, [1, 0, 0, 0, 0, 0])


def test_diamond_sparse_path_and_exhausted_candidates():
    graph = nx.Graph([(1, 2), (2, 3)])
    graph.add_node(4)
    nodelist = [1, 2, 3, 4]
    adjacency = csr_matrix(nx.to_numpy_array(graph, nodelist=nodelist))

    scores = diamond_score(
        graph, [1], adjacency, number_to_rank=10, nodelist=nodelist
    )

    assert np.count_nonzero(scores) == 2
    assert scores[3] == 0
