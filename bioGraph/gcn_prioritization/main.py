"""Command-line entry point for the minimal single-disease GCN example."""

import argparse
from pathlib import Path

from bioGraph.data.loading import load_disease_genes, load_ppi_graph
from bioGraph.evaluation.metrics import mean_reciprocal_rank_at_k, recall_at_k
from bioGraph.gcn_prioritization.training import train_single_disease


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ppi-path", type=Path, default=root / "data" / "raw" / "PPI202207.txt"
    )
    parser.add_argument(
        "--disease-path",
        type=Path,
        default=root / "data" / "raw" / "pcbi.1004120.s004.txt",
    )
    parser.add_argument("--disease-name", default="breast neoplasms")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=int, default=5)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--inner-seed-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[25, 100, 300])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_ppi_graph(args.ppi_path)
    diseases = load_disease_genes(args.disease_path)
    if args.disease_name not in diseases:
        choices = ", ".join(sorted(diseases))
        raise ValueError(
            f"Unknown disease {args.disease_name!r}. Available diseases: {choices}"
        )

    result = train_single_disease(
        graph,
        diseases[args.disease_name],
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        negative_ratio=args.negative_ratio,
        train_fraction=args.train_fraction,
        inner_seed_fraction=args.inner_seed_fraction,
        seed=args.seed,
    )
    ranking, test_genes = result["ranking"], result["test_genes"]

    print("Single-disease GCN prioritization")
    print(f"Disease: {args.disease_name}")
    print(
        f"Graph: {graph.number_of_nodes():,} genes, "
        f"{graph.number_of_edges():,} interactions"
    )
    print(
        f"Known genes in/outside graph: "
        f"{result['known_in_graph']}/{result['known_not_in_graph']} | "
        f"outer split: {len(result['train_genes'])} training genes, "
        f"{len(test_genes)} held-out tests"
    )
    print(
        f"Training: {len(result['losses'])} epochs on {result['device']} | "
        f"final pairwise loss={result['losses'][-1]:.4f}"
    )
    print("Hidden-test metrics:")
    for k in args.k_values:
        print(f"  recall@{k:<3}        = {recall_at_k(ranking, test_genes, k):.4f}")
        print(
            f"  MRR@{k:<3}           = "
            f"{mean_reciprocal_rank_at_k(ranking, test_genes, k):.4f}"
        )
    print("Top 10 candidates:")
    for rank, row in enumerate(ranking[:10], 1):
        print(
            f"  {rank:2}. {row['gene_id']:>9}  {row['symbol']:<12} "
            f"score={row['score']:.4f}"
        )


if __name__ == "__main__":
    main()
