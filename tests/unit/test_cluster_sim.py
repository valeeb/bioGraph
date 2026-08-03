import pickle

import numpy as np

from bioGraph.sim import ARTIFACT_SCHEMA_VERSION
from cluster.sim import CLUSTER_SCHEMA_VERSION, collect_results, render_slurm_script


def _write_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _shard(method, scores):
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config": {
            "disease_set": ["disease"],
            "method_set": [method],
            "num_runs": 1,
            "split_fraction": 0.5,
            "base_seed": 3,
            "hyperparameters": {"parameter": 1},
        },
        "nodelist": [1, 2, 3, 4],
        "runs": [
            {
                "disease": "disease",
                "seed": 3,
                "train_genes": [1],
                "test_genes": [2],
                "scores": {method: np.asarray(scores)},
            }
        ],
    }


def test_collect_results_merges_groups_using_the_manifest_split(tmp_path):
    manifest = {
        "schema_version": CLUSTER_SCHEMA_VERSION,
        "num_splits": 1,
        "split_fraction": 0.5,
        "base_seed": 3,
        "disease_names": ["disease"],
        "splits": [
            {
                "split_index": 0,
                "seed": 3,
                "diseases": {
                    "disease": {"train_genes": [1], "test_genes": [2]}
                },
            }
        ],
    }
    manifest_path = tmp_path / "splits.pkl"
    _write_pickle(manifest_path, manifest)
    shard_root = tmp_path / "shards"
    _write_pickle(
        shard_root / "split_000" / "classical.pkl",
        _shard("RWR", [0.1, 0.2, 0.3, 0.4]),
    )
    _write_pickle(
        shard_root / "split_000" / "gcn.pkl",
        _shard("GCN", [0.4, 0.3, 0.2, 0.1]),
    )

    result = collect_results(manifest_path, shard_root, tmp_path / "results.pkl")

    assert result["config"]["method_set"] == ["RWR", "GCN"]
    assert set(result["runs"][0]["scores"]) == {"RWR", "GCN"}


def test_render_slurm_script_uses_array_and_scratch(tmp_path):
    script = render_slurm_script(
        project_root=tmp_path,
        manifest_path=tmp_path / "splits.pkl",
        shard_root=tmp_path / "shards",
        log_root=tmp_path / "logs",
        group="gcn",
        num_splits=30,
        partition="standard",
        time_limit="24:00:00",
        memory="16G",
        cpus=1,
        python_command="python",
    )

    assert "#SBATCH --array=0-29" in script
    assert "#SBATCH --partition=standard" in script
    assert 'SCRATCH_DIR="/scratch/${USER}/' in script
    assert "\n+  --" not in script
    assert "split_%03d" in script
