"""Training workflows for single- and multi-disease GCN prioritization."""

from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn.functional
from torch.optim import Adam

from bioGraph.data.splitting import split_known_genes, split_training_genes
from bioGraph.evaluation.metrics import compute_ranking_metrics
from bioGraph.gcn_prioritization.model import (
    GCN,
    GraphData,
    make_features,
    prepare_graph,
)
from bioGraph.methods.utils import scores_to_ranking


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducible splits and initialization."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    # Avoid an MKL/OpenMP sparse-operation crash in the qdgp environment.
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ranking(
    scores: torch.Tensor, data: GraphData, excluded: Sequence[int]
) -> list[dict]:
    return scores_to_ranking(
        scores.detach().cpu().numpy(), data.nodelist, data.graph, excluded
    )


def _validate_outer_split(split: Mapping[str, Sequence[int]], known: Sequence[int]) -> None:
    if set(split) != {"train_genes", "test_genes"}:
        raise ValueError("outer_split must contain exactly train_genes and test_genes.")
    train_set, test_set = set(split["train_genes"]), set(split["test_genes"])
    if not train_set or not test_set:
        raise ValueError("outer_split train and test subsets must both be nonempty.")
    if train_set & test_set:
        raise ValueError("outer_split train and test subsets must be disjoint.")
    if train_set | test_set != set(known):
        raise ValueError("outer_split must partition exactly the known genes in the graph.")


def predict_from_seed_genes(
    model: GCN, graph_data: GraphData, seed_genes: Sequence[int],
) -> list[dict]:
    """Use a trained shared GCN to rank candidates for an input seed set."""

    graph_seed_genes = [
        int(gene) for gene in seed_genes if int(gene) in graph_data.node_to_index
    ]
    if not graph_seed_genes:
        raise ValueError("None of the supplied seed genes are present in the graph.")

    device = next(model.parameters()).device
    features = make_features(graph_data, graph_seed_genes).to(device)
    adjacency = graph_data.adjacency.to(device)
    model.eval()
    with torch.no_grad():
        scores = model(features, adjacency)
    return _ranking(scores, graph_data, graph_seed_genes)


def train_single_disease(
    graph: nx.Graph,
    disease_genes: Sequence[int],
    *,
    hidden_dim: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    negative_ratio: int = 5,
    train_fraction: float = 0.75,
    inner_seed_fraction: float = 2.0 / 3.0,
    seed: int = 42,
    graph_data: GraphData | None = None,
    outer_split: Mapping[str, Sequence[int]] | None = None,
) -> dict:
    """Train on random inner splits and evaluate on one shared outer split."""

    set_seed(seed)
    # Reusing graph_data avoids rebuilding the same full sparse adjacency when
    # fitting many disease-specific models.
    data = prepare_graph(graph) if graph_data is None else graph_data
    if data.graph is not graph:
        raise ValueError("graph_data must have been prepared from the supplied graph.")
    graph_genes = set(data.nodelist)
    known = sorted(set(disease_genes) & graph_genes)
    split = outer_split or split_known_genes(known, train_fraction, random_state=seed)
    _validate_outer_split(split, known)
    train_genes = list(split["train_genes"])
    test_genes = list(split["test_genes"])

    known_set = set(known)
    negative_pool = np.asarray(
        [gene for gene in data.nodelist if gene not in known_set], dtype=np.int64
    )
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    rng = np.random.default_rng(seed + 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adjacency = data.adjacency.to(device)
    model = GCN(in_channels=2, hidden_dim=hidden_dim, dropout=0.2).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    losses: list[float] = []
    inner_splits: list[dict[str, list]] = []
    for _ in range(epochs):
        inner_split = split_training_genes(
            train_genes, inner_seed_fraction, random_state=rng
        )
        inner_splits.append(inner_split)
        x = make_features(data, inner_split["seed_genes"]).to(device)
        positive_indices = torch.tensor(
            [data.node_to_index[int(gene)] for gene in inner_split["label_genes"]],
            dtype=torch.long,
            device=device,
        )
        n_negatives = min(
            len(negative_pool), negative_ratio * len(positive_indices)
        )
        sampled = rng.choice(negative_pool, size=n_negatives, replace=False)
        negative_indices = torch.tensor(
            [data.node_to_index[int(gene)] for gene in sampled],
            dtype=torch.long,
            device=device,
        )
        model.train()
        optimizer.zero_grad()
        logits = model(x, adjacency)
        # Pair each sampled negative with a positive target. Repeating positives
        # supports negative_ratio > 1 without changing the requested loss.
        paired_positive_indices = positive_indices.repeat_interleave(negative_ratio)[
            : len(negative_indices)
        ]
        positive_score = logits[paired_positive_indices]
        negative_score = logits[negative_indices]
        loss = torch.nn.functional.softplus(-(positive_score - negative_score)).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    x = make_features(data, train_genes).to(device)
    with torch.no_grad():
        scores = model(x, adjacency)
    ranking = _ranking(scores, data, train_genes)
    return {
        "model": model,
        "graph_data": data,
        "scores": scores.cpu(),
        "ranking": ranking,
        "train_genes": train_genes,
        "test_genes": test_genes,
        "inner_splits": inner_splits,
        "visible_seed_genes": train_genes,
        "losses": losses,
        "known_in_graph": len(known),
        "known_not_in_graph": len(set(disease_genes) - graph_genes),
        "device": str(device),
    }


def train_all_diseases(
    graph: nx.Graph,
    diseases: Mapping[str, Sequence[int]],
    *,
    k_values: Sequence[int] = (25, 100, 300),
    hidden_dim: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    negative_ratio: int = 5,
    train_fraction: float = 0.75,
    inner_seed_fraction: float = 2.0 / 3.0,
    seed: int = 42,
    verbose: bool = True,
    keep_details: bool = False,
    task_batch_size: int = 8,
    outer_splits: Mapping[str, Mapping[str, Sequence[int]]] | None = None,
) -> dict:
    """Train one shared GCN across all disease-conditioned ranking tasks.

    A task is defined by a disease-specific visible-seed indicator and positive
    targets. Disease tasks are batched, but every task uses the same graph and
    updates the same model parameters.
    """

    if not diseases:
        raise ValueError("diseases must not be empty.")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1.")
    if task_batch_size < 1:
        raise ValueError("task_batch_size must be at least 1.")

    # Configure PyTorch before creating/using sparse tensors. In the qdgp
    # environment, changing MKL thread count after sparse work can segfault.
    set_seed(seed)
    data = prepare_graph(graph)
    graph_gene_set = set(data.nodelist)
    rng = np.random.default_rng(seed + 1)

    # Construct every outer task once. Inner seed/label samples are regenerated
    # during training; outer test genes never enter an input, loss, or pool.
    tasks: dict[str, dict] = {}
    for disease_name in sorted(diseases):
        supplied_genes = set(diseases[disease_name])
        known = sorted(supplied_genes & graph_gene_set)
        split = (
            outer_splits[disease_name]
            if outer_splits is not None and disease_name in outer_splits
            else split_known_genes(known, train_fraction, random_state=seed)
        )
        _validate_outer_split(split, known)
        train_genes, test = split["train_genes"], split["test_genes"]
        known_set = set(known)
        negative_pool = np.asarray(
            [data.node_to_index[g] for g in data.nodelist if g not in known_set],
            dtype=np.int64,
        )
        tasks[disease_name] = {
            "negative_pool": negative_pool,
            "train_genes": list(train_genes),
            "test_genes": list(test),
            "known_in_graph": len(known),
            "known_not_in_graph": len(supplied_genes - graph_gene_set),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adjacency = data.adjacency.to(device)
    model = GCN(in_channels=2, hidden_dim=hidden_dim, dropout=0.2).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    disease_names = list(tasks)
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        shuffled_names = rng.permutation(disease_names).tolist()
        batch_losses: list[float] = []
        model.train()

        for start in range(0, len(shuffled_names), task_batch_size):
            batch_names = shuffled_names[start : start + task_batch_size]
            inner_splits = {
                name: split_training_genes(
                    tasks[name]["train_genes"], inner_seed_fraction, random_state=rng
                )
                for name in batch_names
            }
            # [nodes, disease tasks, 2 features]. The sparse graph multiply is
            # shared across the task batch; only the seed channel differs.
            features = torch.stack(
                [make_features(data, inner_splits[name]["seed_genes"])
                 for name in batch_names],
                dim=1,
            ).to(device)

            optimizer.zero_grad()
            logits = model(features, adjacency)  # [nodes, disease tasks]
            task_losses = []
            for task_column, disease_name in enumerate(batch_names):
                task = tasks[disease_name]
                positive_indices = np.asarray(
                    [data.node_to_index[int(g)]
                     for g in inner_splits[disease_name]["label_genes"]],
                    dtype=np.int64,
                )
                n_negatives = min(
                    len(task["negative_pool"]), negative_ratio * len(positive_indices),
                )
                negative_indices = rng.choice(
                    task["negative_pool"], size=n_negatives, replace=False
                )
                paired_positive_indices = np.repeat(positive_indices, negative_ratio)[
                    :n_negatives
                ]
                positive_score = logits[
                    torch.as_tensor(paired_positive_indices, device=device),
                    task_column,
                ]
                negative_score = logits[
                    torch.as_tensor(negative_indices, device=device), task_column
                ]
                task_losses.append(
                    torch.nn.functional.softplus(
                        -(positive_score - negative_score)
                    ).mean()
                )

            loss = torch.stack(task_losses).mean()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(epoch_loss)
        if verbose:
            print(f"Epoch {epoch + 1:>3}/{epochs}: pairwise loss={epoch_loss:.4f}")

    # Evaluate every disease by changing its seed-indicator input while keeping
    # the single trained model fixed.
    model.eval()
    disease_results: dict[str, dict] = {}
    with torch.no_grad():
        for start in range(0, len(disease_names), task_batch_size):
            batch_names = disease_names[start : start + task_batch_size]
            features = torch.stack(
                [make_features(data, tasks[name]["train_genes"])
                 for name in batch_names],
                dim=1,
            ).to(device)
            batch_scores = model(features, adjacency).cpu()

            for task_column, disease_name in enumerate(batch_names):
                task = tasks[disease_name]
                scores = batch_scores[:, task_column]
                ranking = _ranking(scores, data, task["train_genes"])
                metrics: dict[str, float] = {}
                for k in k_values:
                    combined_metrics = compute_ranking_metrics(
                        ranking, task["test_genes"], k
                    )
                    metrics[f"recall@{k}"] = combined_metrics["recall"]
                    metrics[f"ap@{k}"] = combined_metrics["average_precision"]
                result = {
                    key: value
                    for key, value in task.items()
                    if key != "negative_pool"
                }
                result["metrics"] = metrics
                if keep_details:
                    result["scores"] = scores
                    result["ranking"] = ranking
                disease_results[disease_name] = result

    return {
        "model": model,
        "graph_data": data,
        "disease_results": disease_results,
        "losses": epoch_losses,
        "device": str(device),
    }
