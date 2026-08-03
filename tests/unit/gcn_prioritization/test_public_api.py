import pytest

pytest.importorskip("torch")

from bioGraph.gcn_prioritization import (  # noqa: E402
    evaluate_all_diseases as public_evaluate_all,
    predict_from_seed_genes as public_predict,
    train_all_diseases as public_train_all,
    train_single_disease as public_train_single,
)
from bioGraph.gcn_prioritization.inference import (  # noqa: E402
    predict_from_seed_genes,
)
from bioGraph.gcn_prioritization.training import (  # noqa: E402
    evaluate_all_diseases,
    train_all_diseases,
    train_single_disease,
)


def test_package_exports_preserve_the_existing_public_api():
    assert public_evaluate_all is evaluate_all_diseases
    assert public_predict is predict_from_seed_genes
    assert public_train_all is train_all_diseases
    assert public_train_single is train_single_disease
