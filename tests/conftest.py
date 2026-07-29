"""Shared deterministic fixtures and test-suite configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import networkx as nx
import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a fresh deterministic generator for each test."""

    return np.random.default_rng(42)


@pytest.fixture
def small_graph() -> nx.Graph:
    """Return a connected synthetic graph with stable integer gene IDs."""

    graph = nx.Graph()
    graph.add_nodes_from(
        (gene_id, {"symbol": f"GENE{gene_id}"}) for gene_id in range(1, 7)
    )
    graph.add_edges_from(((1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6)))
    return graph


@pytest.fixture
def disease_genes() -> dict[str, list[int]]:
    """Return small disease-gene sets with known overlap."""

    return {"disease_a": [1, 2, 4], "disease_b": [3, 5, 6]}


@pytest.fixture
def ranking_factory() -> Callable[[Mapping[int, float]], list[dict]]:
    """Build ranking rows from gene-to-score mappings."""

    def build(scores: Mapping[int, float]) -> list[dict]:
        rows = [
            {"gene_id": gene_id, "symbol": f"GENE{gene_id}", "score": score}
            for gene_id, score in scores.items()
        ]
        return sorted(rows, key=lambda row: (-row["score"], row["gene_id"]))

    return build


def pytest_collection_modifyitems(items: Sequence[pytest.Item]) -> None:
    """Apply suite markers from the test directory containing each item."""

    marker_by_directory = {
        "unit": pytest.mark.unit,
        "equivalence": pytest.mark.equivalence,
        "integration": pytest.mark.integration,
    }
    for item in items:
        path_parts = Path(str(item.fspath)).parts
        for directory, marker in marker_by_directory.items():
            if directory in path_parts:
                item.add_marker(marker)
                break
