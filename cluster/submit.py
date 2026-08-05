"""Prepare reproducible split files and SLURM array jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# A directly executed script puts ``cluster/`` rather than the repository root
# on sys.path. Add the root so both documented invocation forms behave alike.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.sim import (
    _validated_manifest,
    create_split_manifest,
    render_slurm_script,
    submit_scripts,
)


GCN_CONFIG_KEYS = {
    "epochs",
    "hidden_dim",
    "disease_embedding_dim",
    "learning_rate",
    "weight_decay",
    "negative_ratio",
    "inner_seed_fraction",
    "task_batch_size",
}


def load_gcn_config(path: Path) -> dict:
    """Load and validate the complete GCN hyperparameter configuration."""

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("GCN config must contain a JSON object.")
    unknown = sorted(set(config) - GCN_CONFIG_KEYS)
    missing = sorted(GCN_CONFIG_KEYS - set(config))
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        raise ValueError("Invalid GCN config (" + "; ".join(details) + ").")
    positive_integers = (
        "epochs", "hidden_dim", "disease_embedding_dim", "negative_ratio",
        "task_batch_size",
    )
    for key in positive_integers:
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"GCN config value {key!r} must be a positive integer.")
    for key in ("learning_rate", "weight_decay", "inner_seed_fraction"):
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)):
            raise ValueError(f"GCN config value {key!r} must be numeric.")
    if config["learning_rate"] <= 0:
        raise ValueError("GCN learning_rate must be positive.")
    if config["weight_decay"] < 0:
        raise ValueError("GCN weight_decay must be nonnegative.")
    if not 0 < config["inner_seed_fraction"] < 1:
        raise ValueError("GCN inner_seed_fraction must be between zero and one.")
    return config


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-N", "--num-splits", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--split-fraction", type=float, default=0.75)
    parser.add_argument("--experiment", default="benchmark")
    parser.add_argument(
        "--reuse-splits", action="store_true",
        help=(
            "reuse the experiment's existing splits.pkl without modifying it; "
            "the array size is read from the manifest"
        ),
    )
    parser.add_argument(
        "--gcn-only", action="store_true",
        help="generate and optionally submit only the GCN array",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("/disk/data11/tfp/valeeb"),
        help="persistent cluster storage root (default: /disk/data11/tfp/valeeb)",
    )
    parser.add_argument("--ppi-path", type=Path, default=root / "data" / "raw" / "PPI202207.txt")
    parser.add_argument("--disease-path", type=Path, default=root / "data" / "raw" / "pcbi.1004120.s004.txt")
    parser.add_argument("-p", "--partition", default="standard")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--memory", default="16G")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--account")
    parser.add_argument(
        "--python",
        default="/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python",
        help="Python interpreter used by worker jobs",
    )
    parser.add_argument(
        "--gcn-config", type=Path, default=root / "cluster" / "gcn_config.json",
        help="JSON file containing all GCN hyperparameters",
    )
    parser.add_argument("--gcn-epochs", type=int)
    parser.add_argument("--gcn-hidden-dim", type=int)
    parser.add_argument("--gcn-disease-embedding-dim", type=int)
    parser.add_argument("--gcn-learning-rate", type=float)
    parser.add_argument("--gcn-weight-decay", type=float)
    parser.add_argument("--gcn-negative-ratio", type=int)
    parser.add_argument("--gcn-inner-seed-fraction", type=float)
    parser.add_argument("--gcn-task-batch-size", type=int)
    parser.add_argument(
        "--environment-command",
        help='shell setup inserted before Python, e.g. "source .venv/bin/activate"',
    )
    parser.add_argument(
        "--submit", action="store_true",
        help="submit both arrays with sbatch after generating their files",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    gcn_config = load_gcn_config(args.gcn_config)
    for key in GCN_CONFIG_KEYS:
        override = getattr(args, f"gcn_{key}")
        if override is not None:
            gcn_config[key] = override
    project_root = Path(__file__).resolve().parents[1]
    experiment_root = (args.output_root / args.experiment).resolve()
    shard_root = experiment_root / "shards"
    log_root = experiment_root / "logs"
    script_root = experiment_root / "slurm"
    for directory in (shard_root, log_root, script_root):
        directory.mkdir(parents=True, exist_ok=True)
    (experiment_root / "gcn_config.json").write_text(
        json.dumps(gcn_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = experiment_root / "splits.pkl"
    if args.reuse_splits:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Cannot reuse splits: manifest does not exist: {manifest_path}"
            )
        manifest = _validated_manifest(manifest_path)
        num_splits = manifest["num_splits"]
        print(f"Reusing {num_splits} existing splits (manifest left unchanged): "
              f"{manifest_path}")
    else:
        create_split_manifest(
            args.ppi_path,
            args.disease_path,
            manifest_path,
            num_splits=args.num_splits,
            split_fraction=args.split_fraction,
            base_seed=args.base_seed,
        )
        num_splits = args.num_splits
    scripts = []
    groups = ("gcn",) if args.gcn_only else ("classical", "gcn")
    for group in groups:
        path = script_root / f"{group}.slurm"
        worker_arguments = []
        if group == "gcn":
            worker_arguments = [
                "--gcn-epochs", str(gcn_config["epochs"]),
                "--gcn-hidden-dim", str(gcn_config["hidden_dim"]),
                "--gcn-disease-embedding-dim",
                str(gcn_config["disease_embedding_dim"]),
                "--gcn-learning-rate", str(gcn_config["learning_rate"]),
                "--gcn-weight-decay", str(gcn_config["weight_decay"]),
                "--gcn-negative-ratio", str(gcn_config["negative_ratio"]),
                "--gcn-inner-seed-fraction",
                str(gcn_config["inner_seed_fraction"]),
                "--gcn-task-batch-size", str(gcn_config["task_batch_size"]),
            ]
        path.write_text(
            render_slurm_script(
                project_root=project_root,
                manifest_path=manifest_path,
                shard_root=shard_root,
                log_root=log_root,
                group=group,
                num_splits=num_splits,
                partition=args.partition,
                time_limit=args.time,
                memory=args.memory,
                cpus=args.cpus,
                python_command=args.python,
                worker_arguments=worker_arguments,
                environment_command=args.environment_command,
                account=args.account,
            ),
            encoding="utf-8",
        )
        scripts.append(path)
    if not args.reuse_splits:
        print(f"Created shared splits: {manifest_path}")
    print("Created SLURM arrays:")
    for script in scripts:
        print(f"  {script}")
    if args.submit:
        for script, job_id in zip(scripts, submit_scripts(scripts)):
            print(f"Submitted {script.stem}: job {job_id}")
    else:
        print("Not submitted. Use --submit or call sbatch on each script.")
    if args.gcn_only:
        print("After the GCN array finishes, collect with the existing classical shards:")
    else:
        print("After both arrays finish, collect with:")
    print(
        f"  {args.python} -m cluster.sim collect --manifest {manifest_path} "
        f"--shard-root {shard_root} --output {experiment_root / 'results.pkl'}"
    )


if __name__ == "__main__":
    main()
