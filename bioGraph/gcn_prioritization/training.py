"""Training workflows for single- and multi-disease GCN prioritization."""

from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import torch
import torch.nn.functional
from torch.optim import Adam

from bioGraph.data.splitting import split_disease_genes_three_way
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
    seed_fraction: float = 0.5,
    training_target_fraction: float = 0.25,
    seed: int = 42,
    graph_data: GraphData | None = None,
) -> dict:
    """Train with visible seeds/targets and evaluate only on held-out genes."""

    set_seed(seed)
    # Reusing graph_data avoids rebuilding the same full sparse adjacency when
    # fitting many disease-specific models.
    data = prepare_graph(graph) if graph_data is None else graph_data
    if data.graph is not graph:
        raise ValueError("graph_data must have been prepared from the supplied graph.")
    graph_genes = set(data.nodelist)
    known = sorted(set(disease_genes) & graph_genes)
    (
        visible_seed_genes,
        positive_training_targets,
        test_genes,
    ) = split_disease_genes_three_way(
        known, seed_fraction, training_target_fraction, random_state=seed + 1,
    )
    x = make_features(data, visible_seed_genes)

    known_set = set(known)
    negative_pool = np.asarray(
        [gene for gene in data.nodelist if gene not in known_set], dtype=np.int64
    )
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    n_negatives = min(
        len(negative_pool), negative_ratio * len(positive_training_targets)
    )
    rng = np.random.default_rng(seed + 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, adjacency = x.to(device), data.adjacency.to(device)
    model = GCN(in_channels=2, hidden_dim=hidden_dim, dropout=0.2).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positive_indices = torch.tensor(
        [data.node_to_index[int(gene)] for gene in positive_training_targets],
        dtype=torch.long,
        device=device,
    )

    losses: list[float] = []
    for _ in range(epochs):
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
    with torch.no_grad():
        scores = model(x, adjacency)
    # Only visible seeds are excluded. Training targets remain candidates, while
    # evaluation relevance is restricted strictly to held-out test genes.
    ranking = _ranking(scores, data, visible_seed_genes)
    return {
        "model": model,
        "graph_data": data,
        "scores": scores.cpu(),
        "ranking": ranking,
        "visible_seed_genes": visible_seed_genes.tolist(),
        "positive_training_targets": positive_training_targets.tolist(),
        "test_genes": test_genes.tolist(),
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
    seed_fraction: float = 0.5,
    training_target_fraction: float = 0.25,
    seed: int = 42,
    verbose: bool = True,
    keep_details: bool = False,
    task_batch_size: int = 8,
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

    # Construct every disease task once. Test genes are retained only for the
    # final metric calculation and never enter a loss or negative pool.
    tasks: dict[str, dict] = {}
    for disease_index, disease_name in enumerate(sorted(diseases)):
        supplied_genes = set(diseases[disease_name])
        known = sorted(supplied_genes & graph_gene_set)
        visible, targets, test = split_disease_genes_three_way(
            known,
            seed_fraction,
            training_target_fraction,
            random_state=seed + disease_index + 2,
        )
        known_set = set(known)
        negative_pool = np.asarray(
            [data.node_to_index[g] for g in data.nodelist if g not in known_set],
            dtype=np.int64,
        )
        tasks[disease_name] = {
            "features": make_features(data, visible),
            "positive_indices": np.asarray(
                [data.node_to_index[int(g)] for g in targets], dtype=np.int64
            ),
            "negative_pool": negative_pool,
            "visible_seed_genes": visible.tolist(),
            "positive_training_targets": targets.tolist(),
            "test_genes": test.tolist(),
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
            # [nodes, disease tasks, 2 features]. The sparse graph multiply is
            # shared across the task batch; only the seed channel differs.
            features = torch.stack(
                [tasks[name]["features"] for name in batch_names], dim=1
            ).to(device)

            optimizer.zero_grad()
            logits = model(features, adjacency)  # [nodes, disease tasks]
            task_losses = []
            for task_column, disease_name in enumerate(batch_names):
                task = tasks[disease_name]
                positive_indices = task["positive_indices"]
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
                [tasks[name]["features"] for name in batch_names], dim=1
            ).to(device)
            batch_scores = model(features, adjacency).cpu()

            for task_column, disease_name in enumerate(batch_names):
                task = tasks[disease_name]
                scores = batch_scores[:, task_column]
                ranking = _ranking(scores, data, task["visible_seed_genes"])
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
                    if key not in {"features", "positive_indices", "negative_pool"}
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
