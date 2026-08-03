import pytest

from bioGraph.data.loading import load_disease_genes, load_ppi_graph


def test_load_ppi_graph_reads_nodes_symbols_and_undirected_edges(tmp_path):
    path = tmp_path / "ppi.txt"
    path.write_text(
        "1\tGENE1\t2\tGENE2\n2\tGENE2\t3\tGENE3\n3\tGENE3\t3\tGENE3\ninvalid\n",
        encoding="utf-8",
    )

    graph = load_ppi_graph(path)

    assert set(graph.nodes) == {1, 2, 3}
    assert set(graph.edges) == {(1, 2), (2, 3)}
    assert graph.nodes[2]["symbol"] == "GENE2"


def test_load_disease_genes_reads_gene_lists(tmp_path):
    path = tmp_path / "diseases.txt"
    path.write_text(
        "disease genes\nbreast neoplasms 1/2/4\nalzheimer disease 3/5\n",
        encoding="utf-8",
    )

    assert load_disease_genes(path) == {
        "breast neoplasms": [1, 2, 4],
        "alzheimer disease": [3, 5],
    }


@pytest.mark.parametrize("loader", [load_ppi_graph, load_disease_genes])
def test_loaders_reject_missing_files(loader, tmp_path):
    with pytest.raises(FileNotFoundError):
        loader(tmp_path / "missing.txt")


@pytest.mark.parametrize(
    ("loader", "contents", "message"),
    [
        (load_ppi_graph, "header\ninvalid\n", "No valid interactions"),
        (load_disease_genes, "disease genes\ninvalid\n", "No disease records"),
    ],
)
def test_loaders_reject_files_without_valid_records(loader, contents, message, tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        loader(path)
