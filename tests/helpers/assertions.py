"""Assertions for comparing gene-prioritization outcomes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def assert_same_ranked_genes(
    actual: Sequence[Mapping], expected: Sequence[Mapping], *, top_k: int | None = None,
) -> None:
    """Assert that two rankings contain the same gene IDs in the same order."""

    cutoff = min(len(actual), len(expected)) if top_k is None else top_k
    actual_ids = [row["gene_id"] for row in actual[:cutoff]]
    expected_ids = [row["gene_id"] for row in expected[:cutoff]]
    assert actual_ids == expected_ids


def assert_same_top_genes(
    actual: Sequence[Mapping], expected: Sequence[Mapping], *, top_k: int,
) -> None:
    """Assert that two rankings select the same top-k genes, ignoring order."""

    actual_ids = {row["gene_id"] for row in actual[:top_k]}
    expected_ids = {row["gene_id"] for row in expected[:top_k]}
    assert actual_ids == expected_ids


def assert_metrics_close(
    actual: Mapping[str, float],
    expected: Mapping[str, float],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> None:
    """Assert equal metric names and numerically equivalent values."""

    assert actual.keys() == expected.keys()
    for metric_name in expected:
        np.testing.assert_allclose(
            actual[metric_name],
            expected[metric_name],
            rtol=rtol,
            atol=atol,
            err_msg=f"Metric differs: {metric_name}",
        )
