"""Disease-conditioned GCN prioritization.

The public training API remains available from :mod:`training`. Internally,
graph tensors, models, task construction, objectives, and inference are kept in
small focused modules.
"""

from bioGraph.gcn_prioritization.inference import predict_from_seed_genes
from bioGraph.gcn_prioritization.training import (
    train_all_diseases,
    train_single_disease,
)

__all__ = [
    "predict_from_seed_genes",
    "train_all_diseases",
    "train_single_disease",
]
