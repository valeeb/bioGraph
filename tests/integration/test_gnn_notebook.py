import json
from pathlib import Path


def test_gnn_notebook_describes_the_supported_joint_workflow():
    notebook_path = Path(__file__).parents[2] / "notebooks" / "GNN.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["kernelspec"]["name"] == "qdgp"
    assert "train_all_diseases(" in source
    assert "train_single_disease(" not in source
    assert "disease_id=result['disease_to_id'][disease_name]" in source
    assert not any(cell.get("outputs") for cell in notebook["cells"])
