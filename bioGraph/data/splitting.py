from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_unique_array(known_genes: Iterable) -> np.ndarray:
    if hasattr(known_genes, "detach"):
        known_genes = known_genes.detach().cpu().numpy()
    # Materialize arbitrary iterables such as generators. Passing a generator
    # directly to np.asarray creates a scalar object array instead of consuming
    # its values.
    values = np.asarray(list(known_genes))
    values = values.reshape(-1)
    if values.size == 0:
        return values
    # Sorting makes a seeded split independent of list/set iteration order.
    return np.asarray(sorted(set(values.tolist())))


def _split_sizes(size: int, fractions: np.ndarray) -> np.ndarray:
    """Allocate integer sizes by largest remainder, then enforce nonempty parts."""

    exact = fractions * size
    counts = np.floor(exact).astype(int)
    remainder_order = np.argsort(-(exact - counts), kind="stable")
    for index in remainder_order[: size - int(counts.sum())]:
        counts[index] += 1

    for empty_index in np.flatnonzero(counts == 0):
        donors = np.flatnonzero(counts > 1)
        if not len(donors):
            raise ValueError("Cannot make every split nonempty with these genes.")
        donor = donors[np.argmax(counts[donors] - exact[donors])]
        counts[donor] -= 1
        counts[empty_index] += 1
    return counts


def split_known_genes(
    known_genes: Iterable,
    train_fraction: float = 0.75,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, list]:
    """Reproducibly create the shared outer train/test split.

    Integer sizes use the largest-remainder rule, and both subsets are kept
    nonempty whenever at least two unique genes are supplied.
    """

    genes = _as_unique_array(known_genes)
    if len(genes) < 2:
        raise ValueError(
            "At least two unique known genes are required for nonempty train "
            "and test subsets."
        )
    if not np.isfinite(train_fraction) or not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one.")

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    shuffled = rng.permutation(genes)
    n_train, _ = _split_sizes(
        len(shuffled), np.asarray([train_fraction, 1.0 - train_fraction])
    )
    split = {
        "train_genes": np.sort(shuffled[:n_train]).tolist(),
        "test_genes": np.sort(shuffled[n_train:]).tolist(),
    }
    assert set().union(*(set(values) for values in split.values())) == set(genes)
    assert set(split["train_genes"]).isdisjoint(split["test_genes"])
    return split


def split_training_genes(
    train_genes: Iterable,
    seed_fraction: float = 2.0 / 3.0,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, list]:
    """Create one random inner GCN sample from the outer training genes."""

    split = split_known_genes(train_genes, seed_fraction, random_state)
    return {
        "seed_genes": split["train_genes"],
        "label_genes": split["test_genes"],
    }


def split_disease_genes_three_way(
    positives, seed_fraction, training_target_fraction, random_state=None
):
    """Compatibility wrapper; new code should use :func:`split_known_genes`."""

    outer = split_known_genes(
        positives, seed_fraction + training_target_fraction, random_state
    )
    inner = split_training_genes(
        outer["train_genes"],
        seed_fraction / (seed_fraction + training_target_fraction),
        random_state,
    )
    return inner["seed_genes"], inner["label_genes"], outer["test_genes"]


def split_disease_genes(disease_name, split_fraction, diseases_dict, random_state=None):
    """Compatibility wrapper around the shared outer train/test split."""

    if disease_name not in diseases_dict:
        raise ValueError(f"Disease '{disease_name}' not found.")
    split = split_known_genes(
        diseases_dict[disease_name], split_fraction, random_state
    )
    return split["train_genes"], split["test_genes"]
