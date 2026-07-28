"""Minimal single-disease GCN on the project's real PPI graph.

Run from the repository root with::

    python -m bioGraph.gcn_prioritization.main --disease-name "breast neoplasms"

The example deliberately uses only PyTorch (not torch-geometric).  This keeps the
GCN small and makes the exact graph convolution visible: A_norm @ X @ W.
"""

from __future__ import annotations

import argparse
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import networkx as nx
import numpy as np

try:
    import torch
    from torch import nn
    from torch.optim import Adam
except ImportError as exc:  # Give notebook users a useful error instead of a traceback later.
    raise ImportError(
        "This example requires PyTorch. Install it in the notebook kernel with "
        "`%pip install torch`, restart the kernel, and run the cells again."
    ) from exc

from bioGraph.performance_metric import average_precision_at_k, recall_at_k
from bioGraph.utils import scores_to_ranking


@dataclass
class GraphData:
    """The fixed node order and tensors used by the GCN."""

    graph: nx.Graph
    nodelist: list[int]
    node_to_index: dict[int, int]
    edge_index: torch.Tensor
    adjacency: torch.Tensor
    base_features: torch.Tensor


def set_seed(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducible splits and initialization."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    # This small sparse workload is faster and more stable with one CPU thread.
    # It also avoids an MKL/OpenMP crash in the qdgp notebook environment.
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ppi_graph(path: str | Path) -> nx.Graph:
    """Load the four-column Entrez/symbol PPI file as an undirected graph."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PPI file not found: {path}")

    graph = nx.Graph()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = raw_line.rstrip().split("\t")
            if len(parts) != 4 or not parts[0].isdigit() or not parts[2].isdigit():
                continue
            gene_a, symbol_a, gene_b, symbol_b = parts
            gene_a, gene_b = int(gene_a), int(gene_b)
            graph.add_node(gene_a, symbol=symbol_a)
            graph.add_node(gene_b, symbol=symbol_b)
            if gene_a != gene_b:
                graph.add_edge(gene_a, gene_b)

    if graph.number_of_nodes() == 0:
        raise ValueError(f"No valid interactions were read from {path}")
    return graph


def load_disease_genes(path: str | Path) -> dict[str, list[int]]:
    """Load the disease-to-Entrez-gene mapping used by the notebooks."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Disease file not found: {path}")

    diseases: dict[str, list[int]] = {}
    record = re.compile(r"^(.*\S)\s+([/\d]+)$")
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)  # header
        for raw_line in handle:
            match = record.match(raw_line.strip())
            if match:
                diseases[match.group(1).strip()] = [
                    int(gene) for gene in match.group(2).split("/") if gene
                ]
    if not diseases:
        raise ValueError(f"No disease records were read from {path}")
    return diseases


def split_positives(
    positives: Sequence[int],
    train_fraction: float,
    val_fraction_within_train: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split disease genes into loss/early-stopping/final-test subsets."""

    positives = np.asarray(sorted(set(positives)), dtype=np.int64)
    if len(positives) < 3:
        raise ValueError("At least three disease genes in the PPI are needed.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0.0 < val_fraction_within_train < 1.0:
        raise ValueError("val_fraction_within_train must be between 0 and 1.")

    shuffled = np.random.default_rng(seed).permutation(positives)
    n_train_and_val = min(max(2, round(train_fraction * len(shuffled))), len(shuffled) - 1)
    train_and_val, test = shuffled[:n_train_and_val], shuffled[n_train_and_val:]
    n_val = min(max(1, round(val_fraction_within_train * len(train_and_val))), len(train_and_val) - 1)
    return (
        np.sort(train_and_val[n_val:]),
        np.sort(train_and_val[:n_val]),
        np.sort(test),
    )


def prepare_graph(graph: nx.Graph) -> GraphData:
    """Create normalized sparse adjacency and simple topology-only features."""

    nodelist = sorted(graph.nodes())
    node_to_index = {node: index for index, node in enumerate(nodelist)}
    edges = [(node_to_index[a], node_to_index[b]) for a, b in graph.edges()]
    src = [a for a, b in edges] + [b for a, b in edges]
    dst = [b for a, b in edges] + [a for a, b in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # A_hat = A + I, followed by symmetric D^-1/2 normalization.
    n = len(nodelist)
    loop = torch.arange(n, dtype=torch.long)
    row = torch.cat((edge_index[0], loop))
    col = torch.cat((edge_index[1], loop))
    degree = torch.bincount(row, minlength=n).float()
    values = degree[row].rsqrt() * degree[col].rsqrt()
    adjacency = torch.sparse_coo_tensor(
        torch.stack((row, col)), values, (n, n)
    ).coalesce()

    # No omics covariates are supplied, so use graph topology plus a constant.
    log_degree = torch.log1p(degree - 1.0)
    standardized_degree = (log_degree - log_degree.mean()) / log_degree.std().clamp_min(1e-8)
    base_features = torch.stack((torch.ones(n), standardized_degree), dim=1)
    return GraphData(graph, nodelist, node_to_index, edge_index, adjacency, base_features)


def make_features(data: GraphData, train_genes: Sequence[int]) -> torch.Tensor:
    """Append a seed indicator; message passing spreads it through the PPI."""

    seed_feature = torch.zeros(len(data.nodelist), dtype=torch.float32)
    seed_feature[[data.node_to_index[int(gene)] for gene in train_genes]] = 1.0
    return torch.cat((data.base_features, seed_feature[:, None]), dim=1)


class GCN(nn.Module):
    """Two graph-convolution layers producing one disease score per gene."""

    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = torch.sparse.mm(adjacency, x)
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = torch.sparse.mm(adjacency, x)
        return self.linear2(x).squeeze(1)


def _ranking(scores: torch.Tensor, data: GraphData, excluded: Sequence[int]) -> list[dict]:
    return scores_to_ranking(
        scores.detach().cpu().numpy(), data.nodelist, data.graph, excluded
    )


def train_single_disease(
    graph: nx.Graph,
    disease_genes: Sequence[int],
    *,
    hidden_dim: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    negative_ratio: int = 5,
    train_fraction: float = 0.8,
    val_fraction_within_train: float = 0.2,
    patience: int = 20,
    seed: int = 42,
) -> dict:
    """Train and evaluate one disease; return model, ranking, and diagnostics."""

    set_seed(seed)
    data = prepare_graph(graph)
    graph_genes = set(data.nodelist)
    known = sorted(set(disease_genes) & graph_genes)
    train_genes, val_genes, test_genes = split_positives(
        known, train_fraction, val_fraction_within_train, seed + 1
    )
    x = make_features(data, train_genes)

    # Unknown genes are treated as negatives for this minimal supervised example.
    # They may contain undiscovered positives, a limitation of disease-gene data.
    known_set = set(known)
    negative_pool = np.asarray([g for g in data.nodelist if g not in known_set], dtype=np.int64)
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be at least 1.")
    n_negatives = min(len(negative_pool), negative_ratio * len(train_genes))
    rng = np.random.default_rng(seed + 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, adjacency = x.to(device), data.adjacency.to(device)
    model = GCN(x.shape[1], hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positive_indices = torch.tensor(
        [data.node_to_index[int(g)] for g in train_genes], dtype=torch.long, device=device
    )

    best_state, best_val_ap, stale = None, -1.0, 0
    losses: list[float] = []
    for epoch in range(epochs):
        sampled = rng.choice(negative_pool, size=n_negatives, replace=False)
        negative_indices = torch.tensor(
            [data.node_to_index[int(g)] for g in sampled], dtype=torch.long, device=device
        )
        supervised_indices = torch.cat((positive_indices, negative_indices))
        labels = torch.cat(
            (torch.ones(len(positive_indices), device=device), torch.zeros(len(negative_indices), device=device))
        )

        model.train()
        optimizer.zero_grad()
        logits = model(x, adjacency)
        # Balance classes despite sampling several negatives per positive.
        positive_weight = torch.tensor(
            [len(negative_indices) / len(positive_indices)], dtype=torch.float32, device=device
        )
        loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight)(logits[supervised_indices], labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_scores = model(x, adjacency)
        val_ranking = _ranking(val_scores, data, train_genes)
        val_ap = average_precision_at_k(val_ranking, val_genes)
        if val_ap > best_val_ap + 1e-12:
            best_val_ap = val_ap
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x, adjacency))
    ranking = _ranking(scores, data, np.concatenate((train_genes, val_genes)))
    return {
        "model": model,
        "graph_data": data,
        "scores": scores.cpu(),
        "ranking": ranking,
        "train_genes": train_genes.tolist(),
        "val_genes": val_genes.tolist(),
        "test_genes": test_genes.tolist(),
        "losses": losses,
        "best_val_ap": best_val_ap,
        "known_in_graph": len(known),
        "known_not_in_graph": len(set(disease_genes) - graph_genes),
        "device": str(device),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppi-path", type=Path, default=root / "PPI202207.txt")
    parser.add_argument("--disease-path", type=Path, default=root / "pcbi.1004120.s004.txt")
    parser.add_argument("--disease-name", default="breast neoplasms")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[25, 100, 300])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_ppi_graph(args.ppi_path)
    diseases = load_disease_genes(args.disease_path)
    if args.disease_name not in diseases:
        choices = ", ".join(sorted(diseases))
        raise ValueError(f"Unknown disease {args.disease_name!r}. Available diseases: {choices}")

    result = train_single_disease(
        graph,
        diseases[args.disease_name],
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        negative_ratio=args.negative_ratio,
        patience=args.patience,
        seed=args.seed,
    )
    ranking, test_genes = result["ranking"], result["test_genes"]

    print("Single-disease GCN prioritization")
    print(f"Disease: {args.disease_name}")
    print(f"Graph: {graph.number_of_nodes():,} genes, {graph.number_of_edges():,} interactions")
    print(
        f"Known genes in/outside graph: {result['known_in_graph']}/{result['known_not_in_graph']} | "
        f"split: {len(result['train_genes'])} train, {len(result['val_genes'])} val, "
        f"{len(test_genes)} test"
    )
    print(
        f"Training: {len(result['losses'])} epochs on {result['device']} | "
        f"best validation AP={result['best_val_ap']:.4f}"
    )
    print("Hidden-test metrics:")
    print(f"  average precision = {average_precision_at_k(ranking, test_genes):.4f}")
    for k in args.k_values:
        print(f"  recall@{k:<3}        = {recall_at_k(ranking, test_genes, k):.4f}")
    print("Top 10 candidates:")
    for rank, row in enumerate(ranking[:10], 1):
        print(f"  {rank:2}. {row['gene_id']:>9}  {row['symbol']:<12} score={row['score']:.4f}")


if __name__ == "__main__":
    main()
