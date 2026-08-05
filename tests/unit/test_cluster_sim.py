import pickle
import sys
from copy import deepcopy
from subprocess import CompletedProcess

import networkx as nx
import numpy as np
import pytest

from bioGraph.sim import ARTIFACT_SCHEMA_VERSION
from cluster import sim as cluster_sim
from cluster import submit as cluster_submit
from cluster.sim import (
    CLUSTER_SCHEMA_VERSION,
    collect_results,
    create_split_manifest,
    render_slurm_script,
    run_task,
    submit_scripts,
)
from cluster.submit import load_gcn_config


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


def _manifest():
    return {
        "schema_version": CLUSTER_SCHEMA_VERSION,
        "ppi_path": "/input/ppi.txt",
        "disease_path": "/input/diseases.txt",
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


def _write_collection(tmp_path, manifest=None, classical=None, gcn=None):
    manifest = _manifest() if manifest is None else manifest
    manifest_path = tmp_path / "splits.pkl"
    shard_root = tmp_path / "shards"
    _write_pickle(manifest_path, manifest)
    _write_pickle(
        shard_root / "split_000" / "classical.pkl",
        _shard("RWR", [0.1, 0.2, 0.3, 0.4]) if classical is None else classical,
    )
    _write_pickle(
        shard_root / "split_000" / "gcn.pkl",
        _shard("GCN", [0.4, 0.3, 0.2, 0.1]) if gcn is None else gcn,
    )
    return manifest_path, shard_root


def test_collect_results_merges_groups_using_the_manifest_split(tmp_path):
    manifest_path, shard_root = _write_collection(tmp_path)

    result = collect_results(manifest_path, shard_root, tmp_path / "results.pkl")

    assert result["config"]["method_set"] == ["RWR", "GCN"]
    assert set(result["runs"][0]["scores"]) == {"RWR", "GCN"}


def test_collect_results_preserves_gcn_training_telemetry(tmp_path):
    gcn = _shard("GCN", [0.4, 0.3, 0.2, 0.1])
    telemetry = {
        "method": "GCN",
        "split_index": 0,
        "seed": 3,
        "epoch_losses": [0.8, 0.5],
        "epochs_completed": 2,
    }
    gcn["training_telemetry"] = [telemetry]
    manifest_path, shard_root = _write_collection(tmp_path, gcn=gcn)

    result = collect_results(manifest_path, shard_root, tmp_path / "results.pkl")

    assert result["training_telemetry"] == [telemetry]


def test_gcn_training_telemetry_summarizes_epoch_losses():
    telemetry = cluster_sim._gcn_training_telemetry(
        [0.9, 0.4, 0.5],
        split_index=2,
        seed=7,
        device="cpu",
        training_seconds=10.0,
        inference_seconds=2.5,
    )

    assert telemetry["epoch_losses"] == [0.9, 0.4, 0.5]
    assert telemetry["epochs_completed"] == 3
    assert telemetry["best_epoch"] == 2
    assert telemetry["best_loss"] == 0.4
    assert telemetry["final_loss"] == 0.5
    assert telemetry["total_seconds"] == 12.5


def test_gcn_training_telemetry_rejects_non_finite_losses():
    with pytest.raises(ValueError, match="non-finite"):
        cluster_sim._gcn_training_telemetry(
            [0.9, np.nan],
            split_index=0,
            seed=3,
            device="cpu",
            training_seconds=1.0,
            inference_seconds=0.1,
        )


def test_collect_results_reorders_scores_to_the_classical_node_order(tmp_path):
    gcn = _shard("GCN", [0.4, 0.3, 0.2, 0.1])
    gcn["nodelist"] = [4, 3, 2, 1]
    manifest_path, shard_root = _write_collection(tmp_path, gcn=gcn)

    result = collect_results(manifest_path, shard_root, tmp_path / "results.pkl")

    assert result["nodelist"] == [1, 2, 3, 4]
    assert result["runs"][0]["scores"]["GCN"].tolist() == [0.1, 0.2, 0.3, 0.4]


def test_render_slurm_script_uses_array_and_direct_persistent_output(tmp_path):
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
    assert "/scratch/" not in script
    assert '--output "$TEMP_DESTINATION"' in script
    assert 'mv "$TEMP_DESTINATION" "$DESTINATION"' in script
    assert "\n+  --" not in script
    assert "split_%03d" in script


def test_load_gcn_config_requires_all_known_hyperparameters(tmp_path):
    path = tmp_path / "gcn.json"
    path.write_text('{"epochs": 10}', encoding="utf-8")

    with pytest.raises(ValueError, match="missing keys"):
        load_gcn_config(path)


def test_submit_reuses_manifest_unchanged_and_generates_only_gcn(
    monkeypatch, tmp_path
):
    experiment_root = tmp_path / "existing"
    manifest_path = experiment_root / "splits.pkl"
    manifest = _manifest()
    manifest["num_splits"] = 2
    manifest["splits"] = [
        {**manifest["splits"][0], "split_index": index, "seed": 3 + index}
        for index in range(2)
    ]
    _write_pickle(manifest_path, manifest)
    original_manifest_bytes = manifest_path.read_bytes()
    config_path = tmp_path / "gcn.json"
    config_path.write_text(
        """{
          "epochs": 20,
          "hidden_dim": 64,
          "disease_embedding_dim": 32,
          "learning_rate": 0.001,
          "weight_decay": 0.0001,
          "negative_ratio": 10,
          "inner_seed_fraction": 0.6666666666666666,
          "task_batch_size": 16
        }""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster/submit.py",
            "--output-root", str(tmp_path),
            "--experiment", "existing",
            "--gcn-config", str(config_path),
            "--reuse-splits",
            "--gcn-only",
        ],
    )

    cluster_submit.main()

    assert manifest_path.read_bytes() == original_manifest_bytes
    assert not (experiment_root / "slurm" / "classical.slurm").exists()
    script = (experiment_root / "slurm" / "gcn.slurm").read_text()
    assert "#SBATCH --array=0-1" in script
    assert "--gcn-epochs \\\n  20" in script
    assert "--gcn-hidden-dim \\\n  64" in script


def test_create_split_manifest_is_reproducible(monkeypatch, tmp_path):
    graph = nx.Graph([(1, 2), (2, 3), (3, 4)])
    monkeypatch.setattr(cluster_sim, "load_ppi_graph", lambda path: graph)
    monkeypatch.setattr(
        cluster_sim, "load_disease_genes", lambda path: {"b": [1, 2], "a": [1, 2, 3, 4]}
    )

    first = create_split_manifest(
        "ppi", "diseases", tmp_path / "first.pkl", num_splits=2, base_seed=8
    )
    second = create_split_manifest(
        "ppi", "diseases", tmp_path / "second.pkl", num_splits=2, base_seed=8
    )

    assert first["splits"] == second["splits"]
    assert first["disease_names"] == ["a", "b"]
    assert [row["seed"] for row in first["splits"]] == [8, 9]


def test_create_split_manifest_requires_a_split(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        create_split_manifest("ppi", "diseases", tmp_path / "splits.pkl", num_splits=0)


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"schema_version": -1, "num_splits": 0, "splits": []},
        {"schema_version": CLUSTER_SCHEMA_VERSION, "num_splits": 2, "splits": []},
    ],
)
def test_run_task_rejects_invalid_manifest(manifest, tmp_path):
    path = tmp_path / "manifest.pkl"
    _write_pickle(path, manifest)

    with pytest.raises(ValueError, match="manifest"):
        run_task(path, "classical", 0, tmp_path / "result.pkl")


def test_run_task_rejects_group_and_split_index(tmp_path):
    manifest_path = tmp_path / "manifest.pkl"
    _write_pickle(manifest_path, _manifest())

    with pytest.raises(ValueError, match="Unknown method group"):
        run_task(manifest_path, "unknown", 0, tmp_path / "result.pkl")
    with pytest.raises(ValueError, match="split_index"):
        run_task(manifest_path, "classical", 1, tmp_path / "result.pkl")


def test_classical_run_task_forwards_manifest_split(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.pkl"
    _write_pickle(manifest_path, _manifest())
    graph = nx.Graph([(1, 2)])
    received = {}
    expected = _shard("RWR", [0.1, 0.2, 0.3, 0.4])
    monkeypatch.setattr(cluster_sim, "load_ppi_graph", lambda path: graph)
    monkeypatch.setattr(cluster_sim, "load_disease_genes", lambda path: {"disease": [1, 2]})

    def fake_benchmark(graph, diseases, output, **kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(cluster_sim, "run_benchmark_simulation", fake_benchmark)

    result = run_task(
        manifest_path, "classical", 0, tmp_path / "result.pkl", classical_methods=["RWR"]
    )

    assert result is expected
    assert received["base_seed"] == 3
    assert received["outer_splits"] == _manifest()["splits"][0]["diseases"]


def test_collect_results_reports_missing_shards(tmp_path):
    manifest_path = tmp_path / "manifest.pkl"
    _write_pickle(manifest_path, _manifest())

    with pytest.raises(FileNotFoundError, match="Missing 2 shard"):
        collect_results(manifest_path, tmp_path / "shards", tmp_path / "result.pkl")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nodelist", "node sets"),
        ("methods", "method configuration"),
        ("row_count", "Expected one gcn row"),
        ("seed", "Shard seed"),
        ("split", "manifest split"),
    ],
)
def test_collect_results_rejects_inconsistent_shards(mutation, message, tmp_path):
    gcn = _shard("GCN", [0.4, 0.3, 0.2, 0.1])
    if mutation == "nodelist":
        gcn["nodelist"] = [1, 2, 3, 5]
    elif mutation == "methods":
        gcn["config"]["method_set"] = ["RWR"]
        gcn["runs"][0]["scores"] = {"RWR": gcn["runs"][0]["scores"].pop("GCN")}
    elif mutation == "row_count":
        gcn["runs"][0]["disease"] = "other"
    elif mutation == "seed":
        gcn["runs"][0]["seed"] = 9
    else:
        gcn["runs"][0]["train_genes"] = [2]
        gcn["runs"][0]["test_genes"] = [1]
    manifest_path, shard_root = _write_collection(tmp_path, gcn=gcn)

    with pytest.raises(ValueError, match=message):
        collect_results(manifest_path, shard_root, tmp_path / "result.pkl")


def _valid_gcn_config():
    return {
        "epochs": 10,
        "hidden_dim": 8,
        "disease_embedding_dim": 4,
        "learning_rate": 0.01,
        "weight_decay": 0.001,
        "negative_ratio": 2,
        "inner_seed_fraction": 0.6,
        "task_batch_size": 3,
    }


def test_load_gcn_config_accepts_complete_config(tmp_path):
    path = tmp_path / "gcn.json"
    config = _valid_gcn_config()
    path.write_text(__import__("json").dumps(config), encoding="utf-8")

    assert load_gcn_config(path) == config


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("epochs", True, "positive integer"),
        ("hidden_dim", 0, "positive integer"),
        ("learning_rate", "fast", "numeric"),
        ("learning_rate", 0, "must be positive"),
        ("weight_decay", -1, "nonnegative"),
        ("inner_seed_fraction", 1, "between zero and one"),
    ],
)
def test_load_gcn_config_rejects_invalid_values(key, value, message, tmp_path):
    config = _valid_gcn_config()
    config[key] = value
    path = tmp_path / "gcn.json"
    path.write_text(__import__("json").dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_gcn_config(path)


def test_load_gcn_config_rejects_non_object_and_unknown_keys(tmp_path):
    path = tmp_path / "gcn.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_gcn_config(path)

    config = _valid_gcn_config()
    config["surprise"] = 1
    path.write_text(__import__("json").dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys: surprise"):
        load_gcn_config(path)


def test_render_slurm_script_includes_optional_settings(tmp_path):
    script = render_slurm_script(
        project_root=tmp_path,
        manifest_path=tmp_path / "splits.pkl",
        shard_root=tmp_path / "shards",
        log_root=tmp_path / "logs",
        group="gcn",
        num_splits=2,
        partition="gpu",
        time_limit="01:00:00",
        memory="8G",
        cpus=2,
        python_command="python command",
        worker_arguments=["--gcn-epochs", "5"],
        environment_command="source activate qdgp",
        account="research",
    )

    assert "#SBATCH --account=research" in script
    assert "source activate qdgp" in script
    assert "'python command'" in script
    assert "--gcn-epochs" in script


def test_submit_scripts_invokes_sbatch_and_returns_job_ids(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, stdout="12345\n")

    monkeypatch.setattr(cluster_sim.subprocess, "run", fake_run)

    assert submit_scripts([tmp_path / "job.slurm"]) == ["12345"]
    assert calls[0][0][:2] == ["sbatch", "--parsable"]
    assert calls[0][1]["check"] is True


@pytest.mark.parametrize("submit", [False, True])
def test_submit_main_generates_both_scripts(
    monkeypatch, capsys, tmp_path, submit
):
    config_path = tmp_path / "input-config.json"
    config_path.write_text(__import__("json").dumps(_valid_gcn_config()), encoding="utf-8")
    arguments = [
        "cluster.submit",
        "--output-root", str(tmp_path),
        "--experiment", "experiment",
        "--gcn-config", str(config_path),
        "--gcn-epochs", "12",
        "--num-splits", "2",
    ]
    if submit:
        arguments.append("--submit")
    monkeypatch.setattr(cluster_submit.sys, "argv", arguments)
    manifests = []
    rendered = []
    monkeypatch.setattr(
        cluster_submit,
        "create_split_manifest",
        lambda *args, **kwargs: manifests.append((args, kwargs)),
    )

    def fake_render(**kwargs):
        rendered.append(kwargs)
        return f"script for {kwargs['group']}"

    monkeypatch.setattr(cluster_submit, "render_slurm_script", fake_render)
    monkeypatch.setattr(
        cluster_submit, "submit_scripts", lambda paths: ["101", "102"]
    )

    cluster_submit.main()

    output = capsys.readouterr().out
    experiment = tmp_path / "experiment"
    assert manifests[0][1]["num_splits"] == 2
    assert [item["group"] for item in rendered] == ["classical", "gcn"]
    assert "--gcn-epochs" in rendered[1]["worker_arguments"]
    assert "12" in rendered[1]["worker_arguments"]
    assert (experiment / "slurm" / "classical.slurm").read_text() == "script for classical"
    assert (experiment / "gcn_config.json").is_file()
    if submit:
        assert "Submitted classical: job 101" in output
    else:
        assert "Not submitted" in output


def test_cluster_sim_main_dispatches_commands(monkeypatch, capsys, tmp_path):
    run_args = [
        "cluster.sim", "run-task", "--manifest", str(tmp_path / "manifest.pkl"),
        "--group", "classical", "--split-index", "0", "--output",
        str(tmp_path / "shard.pkl"),
    ]
    calls = []
    monkeypatch.setattr(cluster_sim.sys if hasattr(cluster_sim, "sys") else __import__("sys"), "argv", run_args)
    monkeypatch.setattr(cluster_sim, "run_task", lambda *args, **kwargs: calls.append((args, kwargs)))
    cluster_sim.main()
    assert calls[0][0][1:3] == ("classical", 0)

    collect_args = [
        "cluster.sim", "collect", "--manifest", str(tmp_path / "manifest.pkl"),
        "--shard-root", str(tmp_path / "shards"), "--output", str(tmp_path / "all.pkl"),
    ]
    monkeypatch.setattr(__import__("sys"), "argv", collect_args)
    monkeypatch.setattr(cluster_sim, "collect_results", lambda *args: {"runs": [1, 2]})
    cluster_sim.main()
    assert "Collected 2 disease/split rows" in capsys.readouterr().out
