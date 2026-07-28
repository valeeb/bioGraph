"""GCN architecture and conversion of a NetworkX graph to PyTorch tensors."""

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import torch
from torch import nn


@dataclass
class GraphData:
    """The fixed node order and tensors used by the GCN."""

    graph: nx.Graph
    nodelist: list[int]
    node_to_index: dict[int, int]
    edge_index: torch.Tensor
    adjacency: torch.Tensor


def prepare_graph(graph: nx.Graph) -> GraphData:
    """Create the normalized sparse adjacency for the complete input graph."""

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

    return GraphData(graph, nodelist, node_to_index, edge_index, adjacency)


def make_features(data: GraphData, visible_seed_genes: Sequence[int]) -> torch.Tensor:
    """Create exactly two features: constant one and visible-seed indicator."""

    seed_feature = torch.zeros(len(data.nodelist), dtype=torch.float32)
    seed_feature[
        [data.node_to_index[int(gene)] for gene in visible_seed_genes]
    ] = 1.0
    return torch.stack((torch.ones(len(data.nodelist)), seed_feature), dim=1)


class GCN(nn.Module):
    """Four-step residual GCN producing one disease logit per node."""

    def __init__(self, in_channels: int = 2, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError("This disease-prioritization GCN requires exactly two inputs.")
        self.linear1 = nn.Linear(in_channels, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _propagate(adjacency: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Sparse propagation for [nodes, features] or [nodes, tasks, features]."""

        if x.ndim == 2:
            return torch.sparse.mm(adjacency, x)
        if x.ndim == 3:
            shape = x.shape
            propagated = torch.sparse.mm(adjacency, x.reshape(shape[0], -1))
            return propagated.reshape(shape)
        raise ValueError("x must have shape [nodes, features] or [nodes, tasks, features].")

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # Propagation 1: 2 -> hidden.
        hidden = self._propagate(adjacency, x)
        hidden = self.dropout(torch.relu(self.linear1(hidden)))

        # Propagations 2 and 3 retain hidden width, so residuals are valid.
        update = self._propagate(adjacency, hidden)
        update = self.dropout(torch.relu(self.linear2(update)))
        hidden = hidden + update

        update = self._propagate(adjacency, hidden)
        update = self.dropout(torch.relu(self.linear3(update)))
        hidden = hidden + update

        # Propagation 4 followed by one scalar logit per node.
        hidden = self._propagate(adjacency, hidden)
        return self.output(hidden).squeeze(-1)
