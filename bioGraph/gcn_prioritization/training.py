"""Training workflows for single- and multi-disease GCN prioritization."""

from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import torch
from torch.optim import Adam

from bioGraph.data.splitting import split_known_genes, split_training_genes
from bioGraph.evaluation.metrics import compute_ranking_metrics
from bioGraph.gcn_prioritization.data import GraphData, make_features, prepare_graph
from bioGraph.gcn_prioritization.inference import (
    predict_from_seed_genes,
    ranking_from_scores,
)
from bioGraph.gcn_prioritization.model import DiseaseConditionedGCN, GCN
from bioGraph.gcn_prioritization.objectives import pairwise_ranking_loss
from bioGraph.gcn_prioritization.tasks import build_disease_tasks, validate_outer_split


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducible splits and initialization."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    # Avoid an MKL/OpenMP sparse-operation crash in the qdgp environment.
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    """Train one disease model and evaluate it on genes hidden from training.

    The outer split estimates generalization: ``test_genes`` are never exposed
    to the model. Within each epoch, the outer training genes are split again.
    One part becomes the visible input seeds and the other the positive targets.
    """

    set_seed(seed)
    # Reusing graph_data avoids rebuilding the same full sparse adjacency when
    # fitting many disease-specific models.
    data = prepare_graph(graph) if graph_data is None else graph_data
    if data.graph is not graph:
        raise ValueError("graph_data must have been prepared from the supplied graph.")
    graph_genes = set(data.nodelist)
    known = sorted(set(disease_genes) & graph_genes)
    split = outer_split or split_known_genes(known, train_fraction, random_state=seed)
    validate_outer_split(split, known)
    train_genes = list(split["train_genes"])
    test_genes = list(split["test_genes"])

    # Genes with no known association to this disease act as candidate
    # negatives. Because reliable biological negatives are rarely available,
    # these are more precisely "unlabelled" examples treated as negatives.
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
        # Re-sample the learning problem every epoch. The model sees some known
        # disease genes as an input field on the graph (seed_genes) and learns
        # to assign high scores to the remaining known genes (label_genes).
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
        # Sampling a small negative subset avoids comparing every positive with
        # every node, which would be expensive and dominated by easy negatives.
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
        # Pairwise ranking loss. For each pair, softplus(-(s_pos - s_neg))
        # decreases smoothly when the positive score exceeds the negative one.
        # Thus the absolute score scale is irrelevant; only ordering matters.
        loss = pairwise_ranking_loss(positive_score, negative_score)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    model.eval()
    # Final inference uses every outer-training gene as visible evidence. The
    # held-out test genes remain invisible and are used only to score the ranking.
    x = make_features(data, train_genes).to(device)
    with torch.no_grad():
        scores = model(x, adjacency)
    ranking = ranking_from_scores(scores, data, train_genes)
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
    disease_embedding_dim: int = 16,
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

    A sample consists of the shared graph, a disease-specific seed indicator,
    and a disease ID. Disease tasks are batched, but every task updates the same
    encoder, scorer, and disease-embedding table jointly from initialization.

    In physical terms, the graph propagation rule is shared, while each disease
    supplies a different boundary condition through its seed-indicator feature.
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
    rng = np.random.default_rng(seed + 1)

    # Construct the fixed outer train/test partition for every disease. Inner
    # seed/label samples are regenerated during training; outer test genes never
    # enter an input, loss, or negative pool. This is the key leakage boundary.
    tasks = build_disease_tasks(
        data, diseases, train_fraction, seed, outer_splits
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adjacency = data.adjacency.to(device)
    disease_names = list(tasks)
    disease_to_id = {name: index for index, name in enumerate(disease_names)}
    model = DiseaseConditionedGCN(
        len(disease_names),
        in_channels=2,
        hidden_dim=hidden_dim,
        disease_embedding_dim=disease_embedding_dim,
        dropout=0.2,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_losses: list[float] = []

    for epoch in range(epochs):
        # Shuffling changes which diseases share a mini-batch and prevents a
        # fixed batch ordering from systematically biasing the updates.
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
            # shared across the task batch; only the seed channel differs. This
            # is faster than running an otherwise identical GCN disease by disease.
            features = torch.stack(
                [make_features(data, inner_splits[name]["seed_genes"])
                 for name in batch_names],
                dim=1,
            ).to(device)

            optimizer.zero_grad()
            disease_ids = torch.tensor(
                [disease_to_id[name] for name in batch_names],
                dtype=torch.long,
                device=device,
            )
            logits = model(features, adjacency, disease_ids)
            task_losses = []
            for task_column, disease_name in enumerate(batch_names):
                # Each column is an independent ranking problem. It has its own
                # positive/negative pairs, but all columns differentiate through
                # and update the same GCN parameters.
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
                    pairwise_ranking_loss(positive_score, negative_score)
                )

            # Give diseases equal weight within this batch, rather than allowing
            # diseases with more known genes to dominate the parameter update.
            loss = torch.stack(task_losses).mean()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(epoch_loss)
        if verbose:
            print(f"Epoch {epoch + 1:>3}/{epochs}: pairwise loss={epoch_loss:.4f}")

    # Evaluate every disease by changing only its seed-indicator input while
    # keeping the trained propagation rule fixed. No parameters change here.
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
            disease_ids = torch.tensor(
                [disease_to_id[name] for name in batch_names],
                dtype=torch.long,
                device=device,
            )
            batch_scores = model(features, adjacency, disease_ids).cpu()

            for task_column, disease_name in enumerate(batch_names):
                task = tasks[disease_name]
                scores = batch_scores[:, task_column]
                ranking = ranking_from_scores(scores, data, task["train_genes"])
                metrics: dict[str, float] = {}
                for k in k_values:
                    # Metrics ask how many held-out associations occur near the
                    # top of the candidate list, where experimental follow-up is
                    # realistically possible.
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
        "disease_to_id": disease_to_id,
        "graph_data": data,
        "disease_results": disease_results,
        "losses": epoch_losses,
        "device": str(device),
    }
