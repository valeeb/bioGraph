# bioGraph

bioGraph is a reproducible Python framework for benchmarking classical,
diffusion-based, quantum-inspired, and graph neural network methods for
disease-gene prioritization in biological networks. Given a biological graph
and a set of known disease genes, the methods rank all genes by their predicted
disease relevance.

The repository implements six within-disease prioritization methods:

- aNBR (absolute neighborhood scoring):
  scores each gene by the number of disease-relevant neighbors
- rNBR (relative neighborhood scoring):
  scores each gene by the fraction of its neighbors that are disease relevant
- RWR (random walk with restart):
  scores genes using a converging random walk (based on the implementation from
  [Pathways](https://github.com/mims-harvard/pathways/blob/master/prediction/randomWalk.py))
- DIAMOND (disease module detection algorithm):
  based on the [qDGP implementation](https://github.com/markgolds/qdgp)
- DK (diffusion kernel):
  treats seed genes as finite-temperature sources and simulates diffusion
- QA (quantum algorithm):
  treats seed genes as occupied sites in a tight-binding lattice and simulates
  quantum time evolution

It also includes three cross-disease prediction methods:

- GCN (graph convolutional network):
  a four-layer network trained across disease splits to predict gene-relevance
  scores from an input set of known disease genes
- QA\*:
  extends QA by assigning larger Hamiltonian weights to edges with greater
  disease relevance across diseases
- DK\*:
  extends DK by assigning larger Laplacian weights to edges with greater
  disease relevance across diseases

To prevent information leakage in the cross-disease methods, validation genes
are excluded from training and from modifications to the Hamiltonian or
Laplacian.

## Repository layout

```text
bioGraph/                 Python package
  data/                   Data loading and splitting
  evaluation/             Ranking metrics
  methods/                Prioritization methods and ranking helpers
  gcn_prioritization/     GCN model, training, and command-line entry point
data/
  raw/                    Original input datasets
  processed/              Derived reusable datasets
notebooks/                Exploratory and benchmarking notebooks
outputs/
  figures/                Generated plots and PDFs
  reports/                Human-readable result summaries
  results/                Serialized experiment results
tests/                    Unit, equivalence, and integration tests
cluster/                  SLURM submission, workers, and result collection
```

Run notebooks from the repository root so their project-relative paths resolve
consistently. The GCN example can be run with:

```bash
python -m bioGraph.gcn_prioritization.main --disease-name "breast neoplasms"
```

Run the fast test suite with `pytest`. See `tests/README.md` for the suite
structure and marker-specific commands.

For reproducible SLURM array simulations, including the shared outer splits
used by the classical methods and GCN, see [`cluster/README.md`](cluster/README.md).
Cluster results default to persistent storage under `/disk/data11/tfp/valeeb`.
