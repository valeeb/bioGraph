"""Reproducible simulations for disease-gene prioritization benchmarks."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple, Sequence
from tqdm import tqdm
import networkx as nx
import numpy as np
from scipy import sparse

from bioGraph.data.splitting import split_known_genes
from bioGraph.methods.ranking import (diamond_score, dk_score,
                                      neighbourhood_score, normalize_adjacency,
                                      qa_score, rwr_score)

DEFAULT_METHODS = ("aNBR", "rNBR", "RWR", "DK", "QA+", "QA-", "DIAMOND")
ARTIFACT_SCHEMA_VERSION = 5


class BasisMatrices(NamedTuple):
    """Graph node order and sparse matrices shared by benchmark methods."""

    nodelist: list
    adjacency: sparse.csr_matrix
    laplacian: sparse.csr_matrix
    normalized_adjacency: sparse.csr_matrix


def precompute_basis_matrices(graph: nx.Graph) -> BasisMatrices:
    """Build the sparse matrices used by the benchmark in one node order."""

    nodelist = list(graph.nodes())
    adjacency = sparse.csr_matrix(
        nx.adjacency_matrix(graph, nodelist=nodelist), dtype=float
    )
    laplacian = sparse.csr_matrix(
        nx.laplacian_matrix(graph, nodelist=nodelist), dtype=float
    )
    normalized_adjacency = normalize_adjacency(
        graph, adjacency, nodelist=nodelist
    )
    return BasisMatrices(
        nodelist=nodelist,
        adjacency=adjacency,
        laplacian=laplacian,
        normalized_adjacency=normalized_adjacency,
    )


def run_benchmark_simulation(
    graph: nx.Graph,
    diseases: Mapping[str, Sequence[int]],
    output_path: str | Path,
    *,
    disease_set: Iterable[str],
    method_set: Iterable[str] = DEFAULT_METHODS,
    num_runs: int = 30,
    split_fraction: float = 0.75,
    base_seed: int = 0,
    rwr_return_prob: float = 0.4,
    qa_t: float = 0.45,
    qa_diag: float = 5,
    dk_t: float = 0.3,
    diamond_alpha: float = 9,
    diamond_number_to_rank: int = 300,
) -> dict:
    """Run all requested methods on reproducible disease-specific splits.

    Every deterministic method receives the 75% training union of seed and
    development genes. Test genes remain reserved for evaluation. The complete
    result dictionary is serialized to ``output_path`` and returned.
    """

    selected_diseases = list(disease_set)
    selected_methods = list(method_set)
    if not selected_diseases:
        raise ValueError("disease_set must not be empty.")
    unknown_diseases = sorted(set(selected_diseases) - set(diseases))
    if unknown_diseases:
        raise ValueError(f"Unknown diseases: {', '.join(unknown_diseases)}")
    unknown_methods = sorted(set(selected_methods) - set(DEFAULT_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {', '.join(unknown_methods)}")
    if not selected_methods:
        raise ValueError("method_set must not be empty.")
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1.")

    basis = precompute_basis_matrices(graph)
    nodelist = basis.nodelist

    hyperparameters = {
        "rwr_return_prob": rwr_return_prob,
        "qa_t": qa_t,
        "qa_diag": qa_diag,
        "dk_t": dk_t,
        "diamond_alpha": diamond_alpha,
        "diamond_number_to_rank": diamond_number_to_rank,
    }
    results = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config": {
            "disease_set": selected_diseases,
            "method_set": selected_methods,
            "num_runs": num_runs,
            "split_fraction": split_fraction,
            "base_seed": base_seed,
            "hyperparameters": hyperparameters,
        },
        "nodelist": nodelist,
        "runs": [],
    }

    for disease_name in tqdm(selected_diseases, desc="Diseases"):
        for run_index in tqdm(range(num_runs), desc="Runs", leave=False):
            split_seed = base_seed + run_index
            known = sorted(set(diseases[disease_name]) & set(graph))
            split = split_known_genes(
                known,
                train_fraction=split_fraction,
                random_state=split_seed,
            )
            scores = _score_methods(
                graph,
                split["train_genes"],
                selected_methods,
                nodelist=nodelist,
                adjacency=basis.adjacency,
                laplacian=basis.laplacian,
                normalized_adjacency=basis.normalized_adjacency,
                hyperparameters=hyperparameters,
            )
            _validate_run(graph, split, scores)
            results["runs"].append(
                {
                    "disease": disease_name,
                    "seed": split_seed,
                    "train_genes": split["train_genes"],
                    "test_genes": split["test_genes"],
                    "scores": scores,
                }
            )

    validate_benchmark_results(results)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return results


def validate_benchmark_results(results: Mapping) -> None:
    """Validate an artifact created by :func:`run_benchmark_simulation`."""

    if results.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "This benchmark artifact predates the validated simulation format. "
            "Regenerate it with run_benchmark_simulation()."
        )
    config = results.get("config", {})
    runs = results.get("runs", [])
    nodelist = results.get("nodelist", [])
    expected_runs = (
        len(config.get("disease_set", [])) * config.get("num_runs", 0)
    )
    if len(runs) != expected_runs:
        raise ValueError(f"Artifact has {len(runs)} runs; expected {expected_runs}.")
    expected_methods = set(config.get("method_set", []))
    expected_size = len(nodelist)
    for row in runs:
        train_genes = set(row["train_genes"])
        test_genes = set(row["test_genes"])
        if train_genes & test_genes:
            raise ValueError("Artifact contains overlapping train/test genes.")
        if set(row["scores"]) != expected_methods:
            raise ValueError("Artifact score methods do not match its configuration.")
        for method_name, values in row["scores"].items():
            values = np.asarray(values)
            if values.shape != (expected_size,):
                raise ValueError(
                    f"{method_name} returned shape {values.shape}; "
                    f"expected {(expected_size,)}."
                )


def _score_methods(
    graph,
    training_genes,
    method_set,
    *,
    nodelist,
    adjacency,
    laplacian,
    normalized_adjacency,
    hyperparameters,
):
    """Calculate only the requested score vectors for one split."""

    scores = {}
    if "QA+" in method_set:
        scores["QA+"] = qa_score(
            graph,
            training_genes,
            t=hyperparameters["qa_t"],
            H=adjacency,
            diag=hyperparameters["qa_diag"],
            nodelist=nodelist,
        )
    if "QA-" in method_set:
        scores["QA-"] = qa_score(
            graph,
            training_genes,
            t=hyperparameters["qa_t"],
            H=-adjacency,
            diag=hyperparameters["qa_diag"],
            nodelist=nodelist,
        )
    if "DK" in method_set:
        scores["DK"] = dk_score(
            graph,
            training_genes,
            t=hyperparameters["dk_t"],
            L=laplacian,
            nodelist=nodelist,
        )
    if "RWR" in method_set:
        scores["RWR"] = rwr_score(
            graph,
            training_genes,
            normalized_adjacency=normalized_adjacency,
            return_prob=hyperparameters["rwr_return_prob"],
            nodelist=nodelist,
        )
    if "aNBR" in method_set or "rNBR" in method_set:
        relative, absolute = neighbourhood_score(
            graph, training_genes, A=adjacency, nodelist=nodelist, weighted="both",
        )
        if "aNBR" in method_set:
            scores["aNBR"] = absolute
        if "rNBR" in method_set:
            scores["rNBR"] = relative
    if "DIAMOND" in method_set:
        scores["DIAMOND"] = diamond_score(
            graph,
            training_genes,
            A=adjacency,
            alpha=hyperparameters["diamond_alpha"],
            number_to_rank=hyperparameters["diamond_number_to_rank"],
            nodelist=nodelist,
        )
    return {name: np.asarray(scores[name], dtype=float) for name in method_set}


def _validate_run(graph, split, scores):
    """Reject inconsistent splits or malformed method outputs before saving."""

    train_genes = set(split["train_genes"])
    test_genes = set(split["test_genes"])
    if train_genes & test_genes:
        raise ValueError("Train and test genes overlap.")
    expected_size = graph.number_of_nodes()
    for method_name, values in scores.items():
        if values.shape != (expected_size,):
            raise ValueError(
                f"{method_name} returned shape {values.shape}; "
                f"expected {(expected_size,)}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{method_name} returned non-finite scores.")
