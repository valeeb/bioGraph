from __future__ import annotations

from pathlib import Path

import networkx as nx
import re

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
