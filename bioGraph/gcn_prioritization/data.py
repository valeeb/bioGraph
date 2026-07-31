"""Graph tensors and node features used by the GCN models."""

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import torch


@dataclass
class GraphData:
    """A graph together with its fixed node order and normalized adjacency."""

    graph: nx.Graph
    nodelist: list[int]
    node_to_index: dict[int, int]
    edge_index: torch.Tensor
    adjacency: torch.Tensor


def prepare_graph(graph: nx.Graph) -> GraphData:
    """Convert a NetworkX graph to tensors, including self-loops."""

    nodelist = sorted(graph.nodes())
    node_to_index = {node: index for index, node in enumerate(nodelist)}
    indexed_edges = [(node_to_index[a], node_to_index[b]) for a, b in graph.edges()]
    source = [a for a, b in indexed_edges] + [b for a, b in indexed_edges]
    target = [b for a, b in indexed_edges] + [a for a, b in indexed_edges]
    edge_index = torch.tensor([source, target], dtype=torch.long)

    node_count = len(nodelist)
    self_loops = torch.arange(node_count, dtype=torch.long)
    row = torch.cat((edge_index[0], self_loops))
    column = torch.cat((edge_index[1], self_loops))
    degree = torch.bincount(row, minlength=node_count).float()
    values = degree[row].rsqrt() * degree[column].rsqrt()
    adjacency = torch.sparse_coo_tensor(
        torch.stack((row, column)), values, (node_count, node_count)
    ).coalesce()

    return GraphData(graph, nodelist, node_to_index, edge_index, adjacency)


def make_features(
    data: GraphData, visible_seed_genes: Sequence[int]
) -> torch.Tensor:
    """Return the constant and disease-specific seed features for every gene."""

    seed_indicator = torch.zeros(len(data.nodelist), dtype=torch.float32)
    seed_indices = [
        data.node_to_index[int(gene)] for gene in visible_seed_genes
    ]
    seed_indicator[seed_indices] = 1.0
    return torch.stack((torch.ones(len(data.nodelist)), seed_indicator), dim=1)
