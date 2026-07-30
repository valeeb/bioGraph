import pickle

import networkx as nx
import numpy as np
from scipy import sparse

from bioGraph.data.splitting import split_known_genes
from bioGraph.sim import (precompute_basis_matrices, run_benchmark_simulation,
                          validate_benchmark_results)


def test_precompute_basis_matrices_uses_one_node_order():
    graph = nx.Graph()
    graph.add_edges_from([(20, 10), (10, 30)])

    basis = precompute_basis_matrices(graph)

    assert basis.nodelist == [20, 10, 30]
    assert sparse.isspmatrix_csr(basis.adjacency)
    assert sparse.isspmatrix_csr(basis.laplacian)
    assert sparse.isspmatrix_csr(basis.normalized_adjacency)
    np.testing.assert_array_equal(
        basis.adjacency.toarray(),
        nx.to_numpy_array(graph, nodelist=basis.nodelist),
    )


def test_benchmark_simulation_uses_each_runs_own_split(tmp_path):
    graph = nx.path_graph(range(8))
    nx.set_node_attributes(graph, {node: f"G{node}" for node in graph}, "symbol")
    diseases = {"toy disease": list(graph)}
    output_path = tmp_path / "benchmark.pkl"

    results = run_benchmark_simulation(
        graph,
        diseases,
        output_path,
        disease_set=["toy disease"],
        method_set=["aNBR", "DK", "QA+"],
        num_runs=3,
        qa_t=0.2,
        dk_t=0.2,
    )

    assert output_path.is_file()
    with output_path.open("rb") as handle:
        saved = pickle.load(handle)
    assert saved["config"] == results["config"]
    assert len(saved["runs"]) == 3

    for row in saved["runs"]:
        subsets = [set(row[name]) for name in ("train_genes", "test_genes")]
        assert subsets[0].isdisjoint(subsets[1])
        assert set().union(*subsets) == set(graph)
        assert set(row["scores"]) == {"aNBR", "DK", "QA+"}
        assert all(values.shape == (len(graph),) for values in row["scores"].values())

    qa_vectors = [row["scores"]["QA+"] for row in saved["runs"]]
    assert any(not np.array_equal(qa_vectors[0], values) for values in qa_vectors[1:])
    expected = split_known_genes(graph.nodes, random_state=0)
    assert saved["runs"][0]["train_genes"] == expected["train_genes"]
    assert saved["runs"][0]["test_genes"] == expected["test_genes"]


def test_benchmark_simulation_supports_sparse_diamond(tmp_path):
    graph = nx.cycle_graph(range(6))
    diseases = {"toy disease": list(graph)}

    results = run_benchmark_simulation(
        graph,
        diseases,
        tmp_path / "diamond.pkl",
        disease_set=["toy disease"],
        method_set=["DIAMOND"],
        num_runs=1,
        diamond_number_to_rank=3,
    )

    scores = results["runs"][0]["scores"]["DIAMOND"]
    assert scores.shape == (len(graph),)
    assert np.count_nonzero(scores) <= 3


def test_validation_rejects_old_unversioned_artifact():
    with np.testing.assert_raises_regex(ValueError, "Regenerate"):
        validate_benchmark_results({"config": {}, "runs": []})


def test_qa_score_does_not_mutate_adjacency(small_graph):
    from bioGraph.methods.ranking import qa_score

    nodelist = list(small_graph.nodes())
    adjacency = sparse.csr_matrix(
        nx.adjacency_matrix(small_graph, nodelist=nodelist), dtype=float
    )
    original = adjacency.copy()

    qa_score(small_graph, [1], t=0.1, H=adjacency, diag=5, nodelist=nodelist)

    assert (adjacency != original).nnz == 0
