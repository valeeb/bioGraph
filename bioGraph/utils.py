from typing import Iterable, List, Sequence


def scores_to_ranking(
    scores: Sequence[float],
    nodelist: Sequence,
    graph,
    seed_genes: Iterable,
    include_seed_genes: bool = False,
) -> List[dict]:
    """Convert a score vector into sorted ranking rows.

    Each row has keys: gene_id, symbol, score.
    Scores are sorted descending, then by gene_id for deterministic output.
    """
    seed_set = set(seed_genes)
    ranking = []

    for node, score in zip(nodelist, scores):
        if not include_seed_genes and node in seed_set:
            continue
        ranking.append(
            {
                "gene_id": node,
                "symbol": graph.nodes[node].get("symbol", ""),
                "score": float(score),
            }
        )

    ranking.sort(key=lambda row: (-row["score"], row["gene_id"]))
    return ranking
