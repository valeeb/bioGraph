"""Training objectives for disease-gene ranking."""

import torch


def pairwise_ranking_loss(
    positive_score: torch.Tensor, negative_score: torch.Tensor
) -> torch.Tensor:
    """Prefer each positive gene over its paired sampled negative."""

    return torch.nn.functional.softplus(
        -(positive_score - negative_score)
    ).mean()
