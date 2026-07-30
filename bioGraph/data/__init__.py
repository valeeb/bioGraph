"""Data loading and dataset-splitting utilities."""

from .loading import load_disease_genes, load_ppi_graph
from .splitting import (
    split_disease_genes,
    split_disease_genes_three_way,
    split_known_genes,
    split_training_genes,
)

__all__ = [
    "load_disease_genes",
    "load_ppi_graph",
    "split_known_genes",
    "split_training_genes",
    "split_disease_genes",
    "split_disease_genes_three_way",
]
