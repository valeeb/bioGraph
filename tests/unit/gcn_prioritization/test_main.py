from argparse import Namespace

import pytest

pytest.importorskip("torch")

from bioGraph.gcn_prioritization import main as cli  # noqa: E402


def _args(tmp_path, disease_name="disease"):
    return Namespace(
        ppi_path=tmp_path / "ppi.txt",
        disease_path=tmp_path / "diseases.txt",
        disease_name=disease_name,
        epochs=2,
        hidden_dim=4,
        disease_embedding_dim=3,
        lr=0.02,
        weight_decay=0.001,
        negative_ratio=2,
        train_fraction=0.5,
        inner_seed_fraction=0.5,
        seed=7,
        k_values=[1],
    )


def test_main_rejects_unknown_disease(monkeypatch, small_graph, tmp_path):
    monkeypatch.setattr(cli, "parse_args", lambda: _args(tmp_path, "missing"))
    monkeypatch.setattr(cli, "load_ppi_graph", lambda path: small_graph)
    monkeypatch.setattr(cli, "load_disease_genes", lambda path: {"disease": [1, 2]})

    with pytest.raises(ValueError, match="Unknown disease.*Available diseases: disease"):
        cli.main()


def test_main_forwards_training_options_and_prints_results(
    monkeypatch, capsys, small_graph, tmp_path
):
    received = {}

    def fake_train(graph, diseases, **kwargs):
        received.update(kwargs)
        return {"trained": True}

    evaluated = {
        "disease_results": {
            "disease": {
                "ranking": [
                    {"gene_id": 3, "symbol": "GENE3", "score": 0.9},
                    {"gene_id": 4, "symbol": "GENE4", "score": 0.2},
                ],
                "test_genes": [3],
                "train_genes": [1, 2],
                "known_in_graph": 3,
                "known_not_in_graph": 0,
            }
        },
        "losses": [0.5, 0.25],
        "device": "cpu",
    }
    monkeypatch.setattr(cli, "parse_args", lambda: _args(tmp_path))
    monkeypatch.setattr(cli, "load_ppi_graph", lambda path: small_graph)
    monkeypatch.setattr(cli, "load_disease_genes", lambda path: {"disease": [1, 2, 3]})
    monkeypatch.setattr(cli, "train_all_diseases", fake_train)
    monkeypatch.setattr(cli, "evaluate_all_diseases", lambda trained: evaluated)

    cli.main()

    output = capsys.readouterr().out
    assert received["epochs"] == 2
    assert received["learning_rate"] == 0.02
    assert "Joint multi-disease GCN prioritization" in output
    assert "recall@1" in output
    assert "GENE3" in output
