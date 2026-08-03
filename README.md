# bioGraph

Research code and notebooks for biological-network disease-gene prioritization. The repository implements 6 different methods for disease gen priorization. The input is the graph and set of disease gens, the output a ranking of the disease relevance of all gens:
- aNBR (absolute neighborhood scoring):
  assigns as score to each gen based on the number of disease relevant neighbors
- rNBR (relative neighborhood scoring):
  assigns as score to each gen based on the fraction of disease relevant neighbors over all neighbors
- RWR (random walk with restart):
  assigns as score to each gen based on a converging random walk (uses implementation of https://github.com/mims-harvard/pathways/blob/master/prediction/randomWalk.py)
- DIAMOND (disease module detection algorithm)
  based on https://github.com/markgolds/qdgp
- DK (Diffusion kernel):
  identify seeds as finite temperature. Perform a diffusion simulation
- QA (Quantum algorithm):
  identify seeds with occupied sites in a tight-binding lattice. Perform a quantum time evolution.


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
