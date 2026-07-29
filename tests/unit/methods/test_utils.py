from bioGraph.methods.utils import scores_to_ranking


def test_scores_to_ranking_sorts_candidates_and_excludes_seeds(small_graph):
    ranking = scores_to_ranking(
        [0.1, 0.9, 0.5, 0.5, 0.2, 0.0], [1, 2, 3, 4, 5, 6], small_graph, seed_genes={2},
    )

    assert [row["gene_id"] for row in ranking] == [3, 4, 5, 1, 6]
    assert ranking[0] == {"gene_id": 3, "symbol": "GENE3", "score": 0.5}
