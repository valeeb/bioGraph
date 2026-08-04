"""Create shared splits, run cluster shards, and collect benchmark results."""

from __future__ import annotations

import argparse
import os
import pickle
import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from bioGraph.data.loading import load_disease_genes, load_ppi_graph
from bioGraph.data.splitting import split_known_genes
from bioGraph.sim import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_METHODS,
    run_benchmark_simulation,
    validate_benchmark_results,
)

CLUSTER_SCHEMA_VERSION = 1
METHOD_GROUPS = ("classical", "gcn")


def _dump_pickle(value: object, path: str | Path) -> None:
    """Write a pickle atomically within its destination filesystem."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)


def _load_pickle(path: str | Path) -> object:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def create_split_manifest(
    ppi_path: str | Path,
    disease_path: str | Path,
    output_path: str | Path,
    *,
    num_splits: int = 30,
    split_fraction: float = 0.75,
    base_seed: int = 0,
) -> dict:
    """Materialize every disease split once for all downstream methods."""

    if num_splits < 1:
        raise ValueError("num_splits must be at least 1.")
    graph = load_ppi_graph(ppi_path)
    diseases = load_disease_genes(disease_path)
    graph_nodes = set(graph)
    split_rows = []
    for split_index in range(num_splits):
        seed = base_seed + split_index
        disease_splits = {}
        for disease_name in sorted(diseases):
            known = sorted(set(diseases[disease_name]) & graph_nodes)
            disease_splits[disease_name] = split_known_genes(
                known, train_fraction=split_fraction, random_state=seed
            )
        split_rows.append(
            {"split_index": split_index, "seed": seed, "diseases": disease_splits}
        )
    manifest = {
        "schema_version": CLUSTER_SCHEMA_VERSION,
        "ppi_path": str(Path(ppi_path).resolve()),
        "disease_path": str(Path(disease_path).resolve()),
        "num_splits": num_splits,
        "split_fraction": split_fraction,
        "base_seed": base_seed,
        "disease_names": sorted(diseases),
        "splits": split_rows,
    }
    _dump_pickle(manifest, output_path)
    return manifest


def _validated_manifest(path: str | Path) -> dict:
    manifest = _load_pickle(path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != CLUSTER_SCHEMA_VERSION:
        raise ValueError("Unsupported cluster split manifest.")
    if len(manifest.get("splits", [])) != manifest.get("num_splits"):
        raise ValueError("Split manifest is incomplete.")
    return manifest


def run_task(
    manifest_path: str | Path,
    group: str,
    split_index: int,
    output_path: str | Path,
    *,
    classical_methods: Sequence[str] = DEFAULT_METHODS,
    gcn_epochs: int = 100,
    gcn_hidden_dim: int = 32,
    gcn_disease_embedding_dim: int = 16,
    gcn_learning_rate: float = 0.01,
    gcn_weight_decay: float = 1e-4,
    gcn_negative_ratio: int = 5,
    gcn_inner_seed_fraction: float = 2.0 / 3.0,
    gcn_task_batch_size: int = 8,
) -> dict:
    """Run one independently retryable method-group/split shard."""

    if group not in METHOD_GROUPS:
        raise ValueError(f"Unknown method group {group!r}.")
    manifest = _validated_manifest(manifest_path)
    if not 0 <= split_index < manifest["num_splits"]:
        raise ValueError(f"split_index must be in [0, {manifest['num_splits'] - 1}].")
    split_row = manifest["splits"][split_index]
    graph = load_ppi_graph(manifest["ppi_path"])
    diseases = load_disease_genes(manifest["disease_path"])
    outer_splits = split_row["diseases"]

    if group == "classical":
        result = run_benchmark_simulation(
            graph,
            diseases,
            output_path,
            disease_set=manifest["disease_names"],
            method_set=classical_methods,
            num_runs=1,
            split_fraction=manifest["split_fraction"],
            base_seed=split_row["seed"],
            outer_splits=outer_splits,
        )
    else:
        # Keep manifest generation, classical workers, and result collection
        # usable in lightweight login-node environments without PyTorch.
        from bioGraph.gcn_prioritization.training import (
            evaluate_all_diseases,
            train_all_diseases,
        )

        trained = train_all_diseases(
            graph,
            diseases,
            epochs=gcn_epochs,
            hidden_dim=gcn_hidden_dim,
            disease_embedding_dim=gcn_disease_embedding_dim,
            learning_rate=gcn_learning_rate,
            weight_decay=gcn_weight_decay,
            negative_ratio=gcn_negative_ratio,
            inner_seed_fraction=gcn_inner_seed_fraction,
            seed=split_row["seed"],
            verbose=True,
            task_batch_size=gcn_task_batch_size,
            outer_splits=outer_splits,
        )
        evaluated = evaluate_all_diseases(trained)
        nodelist = list(trained["graph_data"].nodelist)
        runs = []
        for disease_name in manifest["disease_names"]:
            disease_result = evaluated["disease_results"][disease_name]
            rank_score = {
                row["gene_id"]: row["score"]
                for row in disease_result["ranking"]
            }
            runs.append(
                {
                    "disease": disease_name,
                    "seed": split_row["seed"],
                    "train_genes": disease_result["train_genes"],
                    "test_genes": disease_result["test_genes"],
                    "scores": {
                        "GCN": np.asarray(
                            [rank_score.get(gene, 0) for gene in nodelist], dtype=float
                        )
                    },
                }
            )
        result = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "config": {
                "disease_set": manifest["disease_names"],
                "method_set": ["GCN"],
                "num_runs": 1,
                "split_fraction": manifest["split_fraction"],
                "base_seed": split_row["seed"],
                "hyperparameters": {
                    "epochs": gcn_epochs,
                    "hidden_dim": gcn_hidden_dim,
                    "disease_embedding_dim": gcn_disease_embedding_dim,
                    "learning_rate": gcn_learning_rate,
                    "weight_decay": gcn_weight_decay,
                    "negative_ratio": gcn_negative_ratio,
                    "inner_seed_fraction": gcn_inner_seed_fraction,
                    "task_batch_size": gcn_task_batch_size,
                },
            },
            "nodelist": nodelist,
            "runs": runs,
        }
        validate_benchmark_results(result)
        _dump_pickle(result, output_path)
    return result


def collect_results(
    manifest_path: str | Path, shard_root: str | Path, output_path: str | Path
) -> dict:
    """Validate and merge every classical/GCN shard into one artifact."""

    manifest = _validated_manifest(manifest_path)
    shard_root = Path(shard_root)
    loaded: dict[tuple[int, str], Mapping] = {}
    missing = []
    for split_index in range(manifest["num_splits"]):
        for group in METHOD_GROUPS:
            path = shard_root / f"split_{split_index:03d}" / f"{group}.pkl"
            if not path.is_file():
                missing.append(str(path))
                continue
            shard = _load_pickle(path)
            validate_benchmark_results(shard)
            loaded[(split_index, group)] = shard
    if missing:
        preview = "\n  ".join(missing[:10])
        suffix = "\n  ..." if len(missing) > 10 else ""
        raise FileNotFoundError(f"Missing {len(missing)} shard(s):\n  {preview}{suffix}")

    first = loaded[(0, "classical")]
    nodelist = list(first["nodelist"])
    canonical_nodes = set(nodelist)
    classical_methods = list(first["config"]["method_set"])
    combined_runs = []
    for disease_name in manifest["disease_names"]:
        for split_index, split_row in enumerate(manifest["splits"]):
            rows = {}
            for group in METHOD_GROUPS:
                shard = loaded[(split_index, group)]
                shard_nodelist = list(shard["nodelist"])
                if set(shard_nodelist) != canonical_nodes:
                    raise ValueError("Shard node sets do not match.")
                expected_methods = (
                    classical_methods if group == "classical" else ["GCN"]
                )
                if list(shard["config"]["method_set"]) != expected_methods:
                    raise ValueError(f"Inconsistent {group} method configuration.")
                if (
                    shard["config"]["hyperparameters"]
                    != loaded[(0, group)]["config"]["hyperparameters"]
                ):
                    raise ValueError(f"Inconsistent {group} hyperparameters.")
                matches = [r for r in shard["runs"] if r["disease"] == disease_name]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one {group} row for {disease_name!r}, split "
                        f"{split_index}."
                    )
                row = matches[0]
                if shard_nodelist != nodelist:
                    source_index = {
                        node: index for index, node in enumerate(shard_nodelist)
                    }
                    order = [source_index[node] for node in nodelist]
                    row = {
                        **row,
                        "scores": {
                            method: np.asarray(scores)[order]
                            for method, scores in row["scores"].items()
                        },
                    }
                rows[group] = row
            classical_row, gcn_row = rows["classical"], rows["gcn"]
            expected = split_row["diseases"][disease_name]
            for row in (classical_row, gcn_row):
                if row["seed"] != split_row["seed"]:
                    raise ValueError(
                        f"Shard seed does not match split {split_index}."
                    )
                if (
                    row["train_genes"] != expected["train_genes"]
                    or row["test_genes"] != expected["test_genes"]
                ):
                    raise ValueError(
                        f"Shard does not use the manifest split for {disease_name!r}, "
                        f"split {split_index}."
                    )
            combined_runs.append(
                {
                    "disease": disease_name,
                    "seed": split_row["seed"],
                    "train_genes": expected["train_genes"],
                    "test_genes": expected["test_genes"],
                    "scores": {**classical_row["scores"], **gcn_row["scores"]},
                }
            )
    result = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config": {
            "disease_set": manifest["disease_names"],
            "method_set": classical_methods + ["GCN"],
            "num_runs": manifest["num_splits"],
            "split_fraction": manifest["split_fraction"],
            "base_seed": manifest["base_seed"],
            "hyperparameters": {
                "classical": first["config"]["hyperparameters"],
                "gcn": loaded[(0, "gcn")]["config"]["hyperparameters"],
            },
        },
        "nodelist": nodelist,
        "runs": combined_runs,
    }
    validate_benchmark_results(result)
    _dump_pickle(result, output_path)
    return result


def render_slurm_script(
    *,
    project_root: str | Path,
    manifest_path: str | Path,
    shard_root: str | Path,
    log_root: str | Path,
    group: str,
    num_splits: int,
    partition: str,
    time_limit: str,
    memory: str,
    cpus: int,
    python_command: str,
    worker_arguments: Sequence[str] = (),
    environment_command: str | None = None,
    account: str | None = None,
) -> str:
    """Render one SLURM array script for a method group."""

    q = shlex.quote
    directives = [
        f"#SBATCH --job-name=biograph-{group}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --array=0-{num_splits - 1}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --output={Path(log_root).resolve()}/{group}-%A_%a.out",
        f"#SBATCH --error={Path(log_root).resolve()}/{group}-%A_%a.err",
    ]
    if account:
        directives.append(f"#SBATCH --account={account}")
    activation = environment_command or ": # use the submitted shell environment"
    extra_arguments = "".join(f" \\\n  {q(value)}" for value in worker_arguments)
    return "\n".join(
        [
            "#!/bin/bash",
            *directives,
            "",
            "set -euo pipefail",
            f"PROJECT_ROOT={q(str(Path(project_root).resolve()))}",
            f"MANIFEST={q(str(Path(manifest_path).resolve()))}",
            f"SHARD_ROOT={q(str(Path(shard_root).resolve()))}",
            f"PYTHON_COMMAND={q(python_command)}",
            'cd "$PROJECT_ROOT"',
            activation,
            'SPLIT_NAME=$(printf "split_%03d" "$SLURM_ARRAY_TASK_ID")',
            'DESTINATION="$SHARD_ROOT/$SPLIT_NAME/' + group + '.pkl"',
            'mkdir -p "$(dirname "$DESTINATION")"',
            'TEMP_DESTINATION="${DESTINATION}.${SLURM_JOB_ID}.tmp"',
            'trap \'rm -f -- "$TEMP_DESTINATION"\' EXIT',
            '"$PYTHON_COMMAND" -m cluster.sim run-task \\\n  --manifest "$MANIFEST" \\\n  --group ' + q(group) + ' \\\n  --split-index "$SLURM_ARRAY_TASK_ID" \\\n  --output "$TEMP_DESTINATION"' + extra_arguments,
            'mv "$TEMP_DESTINATION" "$DESTINATION"',
            "trap - EXIT",
            "",
        ]
    )


def submit_scripts(paths: Sequence[str | Path]) -> list[str]:
    """Submit generated scripts and return the reported SLURM job IDs."""

    job_ids = []
    for path in paths:
        completed = subprocess.run(
            ["sbatch", "--parsable", str(path)], check=True, text=True,
            capture_output=True,
        )
        job_ids.append(completed.stdout.strip())
    return job_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    task = commands.add_parser("run-task", help="run one split/group shard")
    task.add_argument("--manifest", type=Path, required=True)
    task.add_argument("--group", choices=METHOD_GROUPS, required=True)
    task.add_argument("--split-index", type=int, required=True)
    task.add_argument("--output", type=Path, required=True)
    task.add_argument("--gcn-epochs", type=int, default=100)
    task.add_argument("--gcn-hidden-dim", type=int, default=32)
    task.add_argument("--gcn-disease-embedding-dim", type=int, default=16)
    task.add_argument("--gcn-learning-rate", type=float, default=0.01)
    task.add_argument("--gcn-weight-decay", type=float, default=1e-4)
    task.add_argument("--gcn-negative-ratio", type=int, default=5)
    task.add_argument("--gcn-inner-seed-fraction", type=float, default=2.0 / 3.0)
    task.add_argument("--gcn-task-batch-size", type=int, default=8)
    collect = commands.add_parser("collect", help="merge all completed shards")
    collect.add_argument("--manifest", type=Path, required=True)
    collect.add_argument("--shard-root", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run-task":
        run_task(
            args.manifest, args.group, args.split_index, args.output,
            gcn_epochs=args.gcn_epochs,
            gcn_hidden_dim=args.gcn_hidden_dim,
            gcn_disease_embedding_dim=args.gcn_disease_embedding_dim,
            gcn_learning_rate=args.gcn_learning_rate,
            gcn_weight_decay=args.gcn_weight_decay,
            gcn_negative_ratio=args.gcn_negative_ratio,
            gcn_inner_seed_fraction=args.gcn_inner_seed_fraction,
            gcn_task_batch_size=args.gcn_task_batch_size,
        )
    else:
        result = collect_results(args.manifest, args.shard_root, args.output)
        print(f"Collected {len(result['runs'])} disease/split rows into {args.output}")


if __name__ == "__main__":
    main()
