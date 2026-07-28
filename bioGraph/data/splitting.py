from __future__ import annotations

import numpy as np
from typing import Sequence


def split_disease_genes_three_way(
    positives: Sequence[int],
    seed_fraction: float,
    training_target_fraction: float,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split known genes into visible seeds, training targets, and held-out tests."""

    positives = np.asarray(sorted(set(positives)), dtype=np.int64)
    if len(positives) < 3:
        raise ValueError("At least three disease genes in the PPI are needed.")
    if not 0.0 < seed_fraction < 1.0:
        raise ValueError("seed_fraction must be between 0 and 1.")
    if not 0.0 < training_target_fraction < 1.0:
        raise ValueError("training_target_fraction must be between 0 and 1.")
    if seed_fraction + training_target_fraction >= 1.0:
        raise ValueError("seed and training-target fractions must sum to less than 1.")

    shuffled = np.random.default_rng(random_state).permutation(positives)
    n_seeds = min(max(1, round(seed_fraction * len(shuffled))), len(shuffled) - 2)
    n_targets = min(
        max(1, round(training_target_fraction * len(shuffled))),
        len(shuffled) - n_seeds - 1,
    )
    return (
        np.sort(shuffled[:n_seeds]),
        np.sort(shuffled[n_seeds : n_seeds + n_targets]),
        np.sort(shuffled[n_seeds + n_targets :]),
    )

def split_disease_genes(disease_name, split_fraction, diseases_dict, random_state=None):
    """
    Randomly split the genes of one disease into training and test sets.

    Parameters
    ----------
    disease_name : str
        Disease key in the diseases dictionary.
    split_fraction : float
        Fraction of genes to place in training set (between 0 and 1).
    diseases_dict : dict
        Disease-to-genes mapping.
    random_state : int, optional
        Seed for reproducible random split.

    Returns
    -------
    train_genes : list[int]
    test_genes : list[int]
    """

    if disease_name not in diseases_dict:
        raise ValueError(f"Disease '{disease_name}' not found.")

    if not (0 < split_fraction < 1):
        raise ValueError("split_fraction must be between 0 and 1 (exclusive).")

    genes = list(diseases_dict[disease_name])
    if len(genes) < 2:
        raise ValueError("Need at least 2 genes to create train/test split.")

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(genes)

    n_train = int(len(shuffled) * split_fraction)
    n_train = max(1, min(n_train, len(shuffled) - 1))

    train_genes = shuffled[:n_train].tolist()
    test_genes = shuffled[n_train:].tolist()

    return train_genes, test_genes
