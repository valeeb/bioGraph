"""Prepare reproducible split files and SLURM array jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# A directly executed script puts ``cluster/`` rather than the repository root
# on sys.path. Add the root so both documented invocation forms behave alike.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.sim import create_split_manifest, render_slurm_script, submit_scripts


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-N", "--num-splits", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--split-fraction", type=float, default=0.75)
    parser.add_argument("--experiment", default="benchmark")
    parser.add_argument("--output-root", type=Path, default=root / "outputs" / "results" / "cluster")
    parser.add_argument("--ppi-path", type=Path, default=root / "data" / "raw" / "PPI202207.txt")
    parser.add_argument("--disease-path", type=Path, default=root / "data" / "raw" / "pcbi.1004120.s004.txt")
    parser.add_argument("-p", "--partition", default="standard")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--memory", default="16G")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--account")
    parser.add_argument("--python", default="python")
    parser.add_argument("--gcn-epochs", type=int, default=100)
    parser.add_argument("--gcn-hidden-dim", type=int, default=32)
    parser.add_argument("--gcn-disease-embedding-dim", type=int, default=16)
    parser.add_argument("--gcn-learning-rate", type=float, default=0.01)
    parser.add_argument("--gcn-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gcn-negative-ratio", type=int, default=5)
    parser.add_argument("--gcn-inner-seed-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--gcn-task-batch-size", type=int, default=8)
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
    project_root = Path(__file__).resolve().parents[1]
    experiment_root = (args.output_root / args.experiment).resolve()
    shard_root = experiment_root / "shards"
    log_root = experiment_root / "logs"
    script_root = experiment_root / "slurm"
    for directory in (shard_root, log_root, script_root):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment_root / "splits.pkl"
    create_split_manifest(
        args.ppi_path,
        args.disease_path,
        manifest_path,
        num_splits=args.num_splits,
        split_fraction=args.split_fraction,
        base_seed=args.base_seed,
    )
    scripts = []
    for group in ("classical", "gcn"):
        path = script_root / f"{group}.slurm"
        worker_arguments = []
        if group == "gcn":
            worker_arguments = [
                "--gcn-epochs", str(args.gcn_epochs),
                "--gcn-hidden-dim", str(args.gcn_hidden_dim),
                "--gcn-disease-embedding-dim", str(args.gcn_disease_embedding_dim),
                "--gcn-learning-rate", str(args.gcn_learning_rate),
                "--gcn-weight-decay", str(args.gcn_weight_decay),
                "--gcn-negative-ratio", str(args.gcn_negative_ratio),
                "--gcn-inner-seed-fraction", str(args.gcn_inner_seed_fraction),
                "--gcn-task-batch-size", str(args.gcn_task_batch_size),
            ]
        path.write_text(
            render_slurm_script(
                project_root=project_root,
                manifest_path=manifest_path,
                shard_root=shard_root,
                log_root=log_root,
                group=group,
                num_splits=args.num_splits,
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
    print(f"Created shared splits: {manifest_path}")
    print("Created SLURM arrays:")
    for script in scripts:
        print(f"  {script}")
    if args.submit:
        for script, job_id in zip(scripts, submit_scripts(scripts)):
            print(f"Submitted {script.stem}: job {job_id}")
    else:
        print("Not submitted. Use --submit or call sbatch on each script.")
    print("After both arrays finish, collect with:")
    print(
        f"  python -m cluster.sim collect --manifest {manifest_path} "
        f"--shard-root {shard_root} --output {experiment_root / 'results.pkl'}"
    )


if __name__ == "__main__":
    main()
