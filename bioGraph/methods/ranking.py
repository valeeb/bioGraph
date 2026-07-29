"""Network-based disease-gene prioritization methods."""

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Union

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import expm_multiply
from scipy.stats import hypergeom

logger = logging.getLogger(__name__)


def _resolve_seed_nodes(
    G: nx.Graph,
    seed_input: Union[str, Sequence],
    diseases_dict: Optional[Dict[str, Iterable]] = None,
) -> List:
    """Resolve seed input to graph node labels.

    seed_input can be either:
    - a disease name (str), resolved through diseases_dict
    - a sequence of node IDs directly
    """
    if isinstance(seed_input, str):
        if diseases_dict is None:
            raise ValueError(
                "diseases_dict must be provided when seed_input is a disease name."
            )
        if seed_input not in diseases_dict:
            raise ValueError(f"Disease '{seed_input}' not found.")
        seed_nodes = list(diseases_dict[seed_input])
    else:
        seed_nodes = list(seed_input)

    if len(seed_nodes) == 0:
        raise ValueError("Seed list is empty.")

    node_set = set(G.nodes())
    filtered = [node for node in seed_nodes if node in node_set]
    if len(filtered) == 0:
        raise ValueError("None of the seed nodes are present in the graph.")
    return filtered


def _build_node_index(nodelist: Sequence) -> Dict:
    """Build node -> index mapping for a fixed graph order."""
    return {node: idx for idx, node in enumerate(nodelist)}


def _seed_indices_from_input(
    G: nx.Graph,
    seed_input: Union[str, Sequence],
    nodelist: Sequence,
    diseases_dict: Optional[Dict[str, Iterable]] = None,
) -> np.ndarray:
    """Resolve seed input and convert node labels to positional indices."""
    seed_nodes = _resolve_seed_nodes(
        G, seed_input=seed_input, diseases_dict=diseases_dict
    )
    node_to_idx = _build_node_index(nodelist)
    seed_indices = [node_to_idx[node] for node in seed_nodes if node in node_to_idx]
    if len(seed_indices) == 0:
        raise ValueError(
            "None of the resolved seed nodes are present in the provided nodelist."
        )
    return np.asarray(seed_indices, dtype=int)


def _seed_mask(seed_indices: np.ndarray, n_nodes: int) -> np.ndarray:
    """Build a dense 0/1 seed mask from seed indices."""
    mask = np.zeros(n_nodes, dtype=float)
    mask[seed_indices] = 1.0
    return mask


def qa_score(
    G: nx.Graph,
    seed_list: Union[str, Sequence],
    t: float,
    H: csr_matrix,
    diag: Union[float, None] = None,
    diseases_dict: Optional[Dict[str, Iterable]] = None,
    nodelist: Optional[Sequence] = None,
) -> np.ndarray:
    """Calculate quantum walk scores.

    Args:
    ----
    G: Graph that the walk occures on.
    seed_list: Disease name or list of seed nodes.
    t: Time for which the walk lasts.
    H: Matrix to use as Hamiltonian.
    diag: How to set diagonals of the Hamiltonian.
    diseases_dict: Disease-to-genes mapping (required if seed_input is str).
    nodelist: Graph node order used by H. Defaults to list(G.nodes()).

    Returns:
    -------
    Array containing scores for each node in G.

    """
    if nodelist is None:
        nodelist = list(G.nodes())

    n = len(nodelist)
    seed_indices = _seed_indices_from_input(
        G, seed_input=seed_list, nodelist=nodelist, diseases_dict=diseases_dict,
    )
    n_seeds = len(seed_indices)

    if H.shape[0] != n or H.shape[1] != n:
        raise ValueError(f"H shape {H.shape} does not match nodelist length {n}.")

    hamiltonian = H
    if isinstance(diag, (float, int)):
        diagonal = csr_matrix(
            ([diag] * n_seeds, (seed_indices, seed_indices)), shape=(n, n)
        )
        hamiltonian = H + diagonal

    Z = np.zeros((n, n_seeds), dtype=int)
    Z[seed_indices, np.arange(n_seeds)] = 1
    res = expm_multiply(-1j * t * hamiltonian, Z)
    res = np.abs(res) ** 2
    return res.sum(axis=1)


def dk_score(
    G: nx.Graph,
    seed_list: Union[str, Sequence],
    t: Optional[float] = None,
    L: Optional[csr_matrix] = None,
    P: Optional[np.ndarray] = None,
    diseases_dict: Optional[Dict[str, Iterable]] = None,
    nodelist: Optional[Sequence] = None,
) -> np.ndarray:
    """Score nodes based on classical walk.

    If the probability transition matrix P is pre-computed, use it.
    Otherwise, the Laplacian L will be used to perform the exponential matrix action
    on the seeds.

    Args:
    ----
    G: Graph upon which to walk.
    seed_list: Disease name or list of seed nodes.
    t: Time for which the walk lasts.
    L: Laplacian of G.
    P: Probability transition matrix, if pre-computed.
    diseases_dict: Disease-to-genes mapping (required if seed_input is str).
    nodelist: Graph node order used by L and P. Defaults to list(G.nodes()).

    Returns:
    -------
    Array containing scores for each node in G.
    """
    if nodelist is None:
        nodelist = list(G.nodes())
    n = len(nodelist)
    seed_indices = _seed_indices_from_input(
        G, seed_input=seed_list, nodelist=nodelist, diseases_dict=diseases_dict,
    )

    if P is None:
        if L is None or t is None:
            e = "L and t must be provided if P is None."
            raise ValueError(e)
        if L.shape[0] != n or L.shape[1] != n:
            raise ValueError(f"L shape {L.shape} does not match nodelist length {n}.")
        train_seed_mask = _seed_mask(seed_indices, n)
        return expm_multiply(-t * L, train_seed_mask)

    if P.shape[0] != n or P.shape[1] != n:
        raise ValueError(f"P shape {P.shape} does not match nodelist length {n}.")

    # Aggregate seed influence from selected seed columns.
    return np.asarray(P[:, seed_indices]).mean(axis=1)


def neighbourhood_score(
    G: nx.Graph,
    seed_list: Union[str, Sequence],
    A: Union[np.ndarray, csr_matrix],
    diseases_dict: Optional[Dict[str, Iterable]] = None,
    nodelist: Optional[Sequence] = None,
    weighted: str = "relative",
) -> np.ndarray:
    """Calculate node scores using (un)weighted neighbours.

    Args:
    ----
    G: Graph to use.
    seed_list: Disease name or list of seed nodes.
    A: Dense adjacency of G.
    diseases_dict: Disease-to-genes mapping (required if seed_input is str).
    nodelist: Graph node order used by A. Defaults to list(G.nodes()).
    weighted: ``relative`` normalizes by degree, ``absolute`` counts seed
        neighbours, and ``both`` returns both score vectors.

    Returns:
    -------
    Array containing scores for each node in G.


    """
    if nodelist is None:
        nodelist = list(G.nodes())
    n = len(nodelist)
    seed_indices = _seed_indices_from_input(
        G, seed_input=seed_list, nodelist=nodelist, diseases_dict=diseases_dict,
    )

    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError(f"A shape {A.shape} does not match nodelist length {n}.")

    train_seed_mask = _seed_mask(seed_indices, n)
    if issparse(A):
        # Sparse path: sum seed columns and row degrees without densifying A.
        num_seed_neighbours = np.asarray(A[:, seed_indices].sum(axis=1)).ravel()
        degrees = np.asarray(A.sum(axis=1)).ravel()
    else:
        num_seed_neighbours = np.dot(A, train_seed_mask)
        degrees = np.sum(A, axis=1)
    if weighted == "relative":
        scores = num_seed_neighbours / (degrees + 1e-50)
    elif weighted == "absolute":
        scores = num_seed_neighbours
    elif weighted == "both":
        relative_scores = num_seed_neighbours / (degrees + 1e-50)
        absolute_scores = num_seed_neighbours
        return (
            relative_scores * (1 - train_seed_mask),
            absolute_scores * (1 - train_seed_mask),
        )
    else:
        raise ValueError(
            f"Invalid weighted value {weighted!r}; expected relative, "
            "absolute, or both."
        )

    return scores * (1 - train_seed_mask)


def normalize_adjacency(
    G: nx.Graph, A: csr_matrix, nodelist: Optional[Sequence] = None,
) -> csr_matrix:
    """Compute normalized adjacnecy matrix for RWR.

    Args:
    ----
    G: Graph to use.
    A: Sparse representation of the adjacency matrix.
    nodelist: Graph node order used by A. Defaults to list(G.nodes()).

    Returns:
    -------
    The normalized adjacency matrix, defined by D^(-1/2)AD^(-1/2).

    """
    if nodelist is None:
        nodelist = list(G.nodes())
    n = len(nodelist)

    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError(f"A shape {A.shape} does not match nodelist length {n}.")

    degrees = np.array([G.degree(node) for node in nodelist], dtype=float)
    with np.errstate(divide="ignore"):  # Handle division by zero for isolated nodes
        D_inv_sqrt = np.power(degrees, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0  # Replace inf with 0 for isolated nodes
    D_inv_sqrt_matrix = diags(D_inv_sqrt)
    return D_inv_sqrt_matrix @ A @ D_inv_sqrt_matrix


def rwr_score(
    G: nx.Graph,
    seed_list: Union[str, Sequence],
    normalized_adjacency: csr_matrix,
    return_prob: float = 0.75,
    diseases_dict: Optional[Dict[str, Iterable]] = None,
    nodelist: Optional[Sequence] = None,
) -> np.ndarray:
    """Score nodes based on random walk with restart.

    Modified from
    https://github.com/mims-harvard/pathways/blob/master/prediction/randomWalk.py

    Args:
    ----
    G: Graph to use.
    seed_list: Disease name or list of seed nodes.
    return_prob: Probability of return to seed nodes.
    normalized_adjacency: D^{-1/2} A D^{-1/2}.
    diseases_dict: Disease-to-genes mapping (required if seed_input is str).
    nodelist: Graph node order used by normalized_adjacency. Defaults to graph
        node order.

    Returns:
    -------
    Array containing scores for each node in G.

    """
    if nodelist is None:
        nodelist = list(G.nodes())
    n = len(nodelist)
    seed_indices = _seed_indices_from_input(
        G, seed_input=seed_list, nodelist=nodelist, diseases_dict=diseases_dict,
    )

    if normalized_adjacency.shape[0] != n or normalized_adjacency.shape[1] != n:
        raise ValueError(
            "normalized_adjacency shape "
            f"{normalized_adjacency.shape} does not match nodelist length {n}."
        )

    train_seed_mask = _seed_mask(seed_indices, n)
    assoc_gene_vector = train_seed_mask
    ratio = return_prob
    convergence_metric = 1
    p0 = assoc_gene_vector / np.sum(assoc_gene_vector)
    old_vector = p0
    while convergence_metric > 1e-6:
        new_vector = (1 - ratio) * normalized_adjacency.dot(old_vector) + ratio * p0
        convergence_metric = np.linalg.norm(new_vector - old_vector)
        old_vector = np.copy(new_vector)
    return old_vector * (1 - assoc_gene_vector)


def _compare_to_existing(
    processed_seed_conns: List[float],
    processed_total_conns: List[float],
    seed_conns: float,
    total_conns: float,
) -> bool:
    for a, b in zip(processed_seed_conns, processed_total_conns):
        if a >= seed_conns and b <= total_conns:
            return True
    return False


def diamond_score(
    G: nx.Graph,
    seed_list: Union[str, Sequence],
    A: Union[np.ndarray, csr_matrix],
    alpha: float = 5,
    number_to_rank: int = 100,
    diseases_dict: Optional[Dict[str, Iterable]] = None,
    nodelist: Optional[Sequence] = None,
) -> np.ndarray:
    """Score nodes based on the diamond algorithm.

    Modified from https://github.com/markgolds/qdgp

    Args:
    ----
    G: Graph to use.
    seed_list: Disease name or list of seed nodes.
    A: Dense or sparse adjacency of G.
    alpha: diamond parameter.
    number_to_rank: Score only this many nodes.
    diseases_dict: Disease-to-genes mapping (required if seed_input is str).
    nodelist: Graph node order used by A. Defaults to list(G.nodes()).

    Returns:
    -------
    Scores for the top `number_to_rank` nodes, according to the diamond algorithm.

    """
    if nodelist is None:
        nodelist = list(G.nodes())
    n = len(nodelist)
    seed_indices = _seed_indices_from_input(
        G, seed_input=seed_list, nodelist=nodelist, diseases_dict=diseases_dict,
    )

    if A.shape[0] != n or A.shape[1] != n:
        raise ValueError(f"A shape {A.shape} does not match nodelist length {n}.")

    train_seed_mask = _seed_mask(seed_indices, n)
    assoc_gene_vector = train_seed_mask
    num_genes = assoc_gene_vector.shape[0]
    edges_per_gene = np.asarray(A.sum(axis=0)).ravel()
    scores = np.zeros(assoc_gene_vector.shape)
    seeds = np.copy(assoc_gene_vector)
    connections_to_seeds = np.asarray(
        A[:, np.nonzero(assoc_gene_vector)[0]].sum(axis=1)
    ).ravel()
    num_gene_edges = edges_per_gene + (alpha - 1) * connections_to_seeds
    N = num_genes + np.sum(assoc_gene_vector) * (alpha - 1)
    connections_to_seeds = connections_to_seeds * (alpha)
    num_seeds = alpha * np.sum(assoc_gene_vector)
    for index in range(1, number_to_rank + 1):
        potential_cand = np.nonzero(connections_to_seeds * (1 - seeds) >= 1)[0]
        num_candidates = potential_cand.shape[0]
        if num_candidates == 0:
            break
        best_cand = -1
        best_conn = 1
        processed_seed_conns: List[float] = []
        processed_total_conns: List[float] = []
        sf_cache: Dict = {}
        for i in range(num_candidates):
            cand_index = potential_cand[i]
            cand_seed_conns = float(connections_to_seeds[cand_index])
            cand_total_conns = float(num_gene_edges[cand_index])
            if _compare_to_existing(
                processed_seed_conns,
                processed_total_conns,
                cand_seed_conns,
                cand_total_conns,
            ):
                continue
            sf_key = (cand_seed_conns, cand_total_conns, num_seeds)
            conn = sf_cache.get(sf_key)
            if conn is None:
                conn = hypergeom.sf(
                    cand_seed_conns - 1, N, num_seeds, cand_total_conns,
                )
                sf_cache[sf_key] = conn
            processed_seed_conns.append(cand_seed_conns)
            processed_total_conns.append(cand_total_conns)
            if conn < best_conn:
                best_conn = conn
                best_cand = cand_index
        candidate_connections = A[:, best_cand]
        if issparse(candidate_connections):
            candidate_connections = candidate_connections.toarray().ravel()
        else:
            candidate_connections = np.asarray(candidate_connections).ravel()
        connections_to_seeds += candidate_connections
        seeds[best_cand] = 1
        scores[best_cand] = 1.0 / index
        num_seeds += 1
    return scores
