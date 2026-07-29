import pickle

import networkx as nx
import numpy as np
from scipy import sparse

from bioGraph.sim import run_benchmark_simulation, validate_benchmark_results


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
        assert set(row["train_genes"]).isdisjoint(row["test_genes"])
        assert set(row["train_genes"]) | set(row["test_genes"]) == set(graph)
        assert set(row["scores"]) == {"aNBR", "DK", "QA+"}
        assert all(values.shape == (len(graph),) for values in row["scores"].values())

    qa_vectors = [row["scores"]["QA+"] for row in saved["runs"]]
    assert any(not np.array_equal(qa_vectors[0], values) for values in qa_vectors[1:])


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
