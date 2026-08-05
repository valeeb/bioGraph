"""Shared and disease-conditioned GCN model definitions."""

import torch
from torch import nn

# Compatibility re-exports: existing callers may still import these from model.
from bioGraph.gcn_prioritization.data import GraphData, make_features, prepare_graph


class GCNEncoder(nn.Module):
    """Four-layer graph encoder producing one representation per node."""

    def __init__(
        self, in_channels: int = 2, hidden_dim: int = 32, dropout: float = 0.2
    ) -> None:
        super().__init__()
        if in_channels != 2:
            raise ValueError(
                "This disease-prioritization GCN requires exactly two inputs."
            )
        self.linear1 = nn.Linear(in_channels, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
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
        raise ValueError(
            "x must have shape [nodes, features] or [nodes, tasks, features]."
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # Propagation 1: 2 -> hidden.
        hidden = self._propagate(adjacency, x)
        hidden = self.dropout(torch.relu(self.linear1(hidden)))

        # Propagations 2--4 retain hidden width, so residuals are valid.
        for linear in (self.linear2, self.linear3, self.linear4):
            update = self._propagate(adjacency, hidden)
            update = self.dropout(torch.relu(linear(update)))
            hidden = hidden + update

        return hidden


class GCN(nn.Module):
    """Backward-compatible single-task GCN built on the shared encoder."""

    def __init__(
        self, in_channels: int = 2, hidden_dim: int = 32, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_dim, dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(x, adjacency)).squeeze(-1)


class DiseaseConditionedGCN(nn.Module):
    """A shared GCN encoder with learned disease-specific embeddings."""

    def __init__(
        self,
        num_diseases: int,
        *,
        in_channels: int = 2,
        hidden_dim: int = 32,
        disease_embedding_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_diseases < 1:
            raise ValueError("num_diseases must be at least 1.")
        self.encoder = GCNEncoder(in_channels, hidden_dim, dropout)
        self.disease_embeddings = nn.Embedding(num_diseases, disease_embedding_dim)
        # A nonlinear interaction is essential here. A single linear projection
        # would add the same disease-dependent constant to every gene, which
        # cancels from (positive_score - negative_score) in the ranking loss.
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + disease_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.disease_projection = nn.Linear(
            disease_embedding_dim, hidden_dim, bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        disease_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score genes for one task or a batch of disease tasks.

        ``x`` is either ``[nodes, features]`` or
        ``[nodes, tasks, features]``. The corresponding disease IDs have shape
        ``[]``/``[1]`` or ``[tasks]``.
        """

        hidden = self.encoder(x, adjacency)
        disease_ids = disease_ids.to(device=hidden.device, dtype=torch.long)
        if x.ndim == 2:
            if disease_ids.numel() != 1:
                raise ValueError("A single graph sample requires one disease ID.")
            embedding = self.disease_embeddings(disease_ids.reshape(1))[0]
            embedding = embedding.expand(hidden.shape[0], -1)
        elif x.ndim == 3:
            disease_ids = disease_ids.reshape(-1)
            if disease_ids.numel() != x.shape[1]:
                raise ValueError("There must be one disease ID per task.")
            embedding = self.disease_embeddings(disease_ids)
            embedding = embedding.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        else:
            raise ValueError("x must have two or three dimensions.")
        combined_score = self.scorer(
            torch.cat((hidden, embedding), dim=-1)
        ).squeeze(-1)
        # This explicit gene/disease interaction also ensures the embedding is
        # not merely a gene-independent offset under the pairwise objective.
        interaction_score = (
            hidden * self.disease_projection(embedding)
        ).sum(dim=-1) / hidden.shape[-1] ** 0.5
        return combined_score + interaction_score
