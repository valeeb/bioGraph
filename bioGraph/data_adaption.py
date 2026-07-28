import numpy as np

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