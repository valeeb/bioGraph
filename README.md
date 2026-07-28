# bioGraph

Research code and notebooks for biological-network disease-gene prioritization.

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
tests/                    Future automated tests
```

Run notebooks from the repository root so their project-relative paths resolve
consistently. The GCN example can be run with:

```bash
python -m bioGraph.gcn_prioritization.main --disease-name "breast neoplasms"
```
