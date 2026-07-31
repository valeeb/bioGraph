"""Inference helpers for converting model scores to candidate rankings."""

from typing import Sequence

import torch

from bioGraph.gcn_prioritization.data import GraphData, make_features
from bioGraph.gcn_prioritization.model import DiseaseConditionedGCN, GCN
from bioGraph.methods.utils import scores_to_ranking


def ranking_from_scores(
    scores: torch.Tensor, data: GraphData, excluded_genes: Sequence[int]
) -> list[dict]:
    """Convert one score per graph node into a sorted candidate list."""

    return scores_to_ranking(
        scores.detach().cpu().numpy(),
        data.nodelist,
        data.graph,
        excluded_genes,
    )


def predict_from_seed_genes(
    model: GCN | DiseaseConditionedGCN,
    graph_data: GraphData,
    seed_genes: Sequence[int],
    disease_id: int | None = None,
) -> list[dict]:
    """Rank graph genes from a visible seed set and optional disease ID."""

    graph_seed_genes = [
        int(gene) for gene in seed_genes if int(gene) in graph_data.node_to_index
    ]
    if not graph_seed_genes:
        raise ValueError("None of the supplied seed genes are present in the graph.")

    device = next(model.parameters()).device
    features = make_features(graph_data, graph_seed_genes).to(device)
    adjacency = graph_data.adjacency.to(device)
    model.eval()
    with torch.no_grad():
        if isinstance(model, DiseaseConditionedGCN):
            if disease_id is None:
                raise ValueError(
                    "disease_id is required for a disease-conditioned model."
                )
            disease = torch.tensor(disease_id, device=device)
            scores = model(features, adjacency, disease)
        else:
            scores = model(features, adjacency)
    return ranking_from_scores(scores, graph_data, graph_seed_genes)
