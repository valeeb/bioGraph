# Disease-conditioned GCN prioritization

This package ranks candidate genes for several diseases using one shared graph
neural network. The design separates two kinds of information:

- **Shared biological structure:** the protein–protein interaction (PPI) graph
  and the rules for propagating information across it.
- **Disease-specific information:** the known seed genes and a learned vector
  (embedding) for each disease.

The model is trained jointly across diseases. There is no separate GCN
pretraining stage and no copy of the model that is fine-tuned for an individual
disease.

## The model at a glance

For one disease, a training sample contains:

1. the common PPI graph;
2. a binary field over its nodes indicating the currently visible seed genes of disease;
3. the integer ID of the disease.

The computation is:
[toDO This diagram is not correct yet.]
```text
Shared PPI graph ───────────────┐
                                │
Disease-specific seed indicator ├─► shared GCN encoder
                                │         │
Constant node feature ──────────┘         ▼
                               gene representation hᵢ,d
                                          │
Disease ID ─► embedding table ─► embedding e_d
                                          │
                         ┌────────────────┘
                         ▼
           nonlinear + multiplicative interaction
                         │
                         ▼
               gene score sᵢ,d
```

Only the seed indicator and disease ID change between disease tasks. The graph
and all GCN parameters are shared.

## 1. Graph representation

Let the PPI graph have adjacency matrix $A$. Before message passing, the code
adds self-connections and applies symmetric degree normalization:

$$
\hat A = A + I,
\qquad
\tilde A = \hat D^{-1/2}\hat A\hat D^{-1/2}.
$$

Multiplication by $\tilde A$ replaces each node's state by a degree-weighted
combination of its own state and those of its neighbors. This resembles one
discrete diffusion step on the network, although the feature channels are mixed
by learned linear maps after propagation.

The normalized matrix is stored sparsely because a PPI network contains far
fewer edges than the $N^2$ possible gene pairs.

Each gene begins with two features:

$$
x_i = (1, q_i),
$$

where $q_i=1$ if gene $i$ is a visible seed for the current disease and is
zero otherwise. The constant channel allows the network to learn a baseline
representation of graph structure even away from the seeds.

This logic lives in [`data.py`](data.py).

## 2. Shared GCN encoder

The encoder applies two graph-propagation steps. In schematic form,

$$
H^{(1)} = \operatorname{Dropout}\!\left[
  \operatorname{ReLU}(\tilde A X W_1 + b_1)
\right],
$$

$$
U^{(2)} = \operatorname{Dropout}\!\left[
  \operatorname{ReLU}(\tilde A H^{(1)} W_2 + b_2)
\right],
\qquad
H = H^{(1)} + U^{(2)}.
$$

The final addition is a residual connection. It gives the model access to both
the first propagation result and its two-step update, and also helps gradients
flow through the network during optimization.

The row \(h_i\) of \(H\) is the shared hidden representation of gene \(i\).
Because the same encoder is used for every disease, it can learn propagation
patterns that recur across disease mechanisms.

The implementation is `GCNEncoder` in [`model.py`](model.py).

## 3. Disease embeddings and gene scores

Every disease $d$ has a trainable embedding vector $e_d$. These vectors are
initialized randomly and optimized together with the GCN from the first
training step.

For every gene, the model concatenates its graph representation with the
current disease embedding:

$$
z_{i,d} = [h_i; e_d].
$$

A small nonlinear network maps $z_{i,d}$ to a scalar. The model also includes
an explicit multiplicative interaction between the gene and disease vectors:

$$
s_{i,d} = \operatorname{MLP}(z_{i,d})
 + \frac{h_i^\mathsf{T}P e_d}{\sqrt{D}},
$$

where \(P\) projects the disease embedding into the \(D\)-dimensional gene
representation space.

The nonlinear and multiplicative interactions are important. If the embedding
entered through only a linear additive term, it would add the same constant to
every gene score for a disease. That constant would cancel from the pairwise
ranking loss and could not change the gene ordering.

This is implemented by `DiseaseConditionedGCN` in [`model.py`](model.py).

## 4. Training objective

For each disease, its known genes are first divided into:

- **outer-training genes**, which may be used during optimization;
- **held-out test genes**, which are used only for final evaluation.

During each epoch, the outer-training genes are split again. One subset becomes
the visible seed indicator, and the other supplies positive training labels.
Negatives are sampled from genes with no known association to that disease.

For a positive gene and a sampled negative gene, the loss is exactly

```python
torch.nn.functional.softplus(
    -(positive_score - negative_score)
).mean()
```

Equivalently, for a score difference $\Delta s=s_+-s_-$, the contribution is

$$
\ell(\Delta s)=\log(1+e^{-\Delta s}).
$$

Thus the loss becomes small when the positive gene is ranked well above the
negative gene. It constrains relative ordering rather than an absolute score
scale.

Disease tasks are shuffled and processed in small batches. A single optimizer
updates the following parameters jointly:

- the shared GCN encoder;
- the disease-embedding table;
- the gene–disease scoring layers.

The objective is defined in [`objectives.py`](objectives.py), while the training
loop is in [`training.py`](training.py).

## 5. Avoiding test leakage

The held-out genes for a disease must not affect its optimization. In this
implementation they are never used as:

- visible input seeds;
- positive training labels.

Therefore there is a set of sets of gens for each disease which the GCNencoder never sees as disease gens. 

The split validation and task construction are
centralized in [`tasks.py`](tasks.py).

## 6. Testing the perfomance.
After training is complete, the performance of the model for the given outer split is evaluated. 

For this one final forward pass on all outer-training genes is performed which scores all gens and leads to a ranking of genes. (The outer-training genes are not part of the ranking.) Performance is then measured against the untouched test genes. This can be done for each disease because we left out gens for each disease. 

## 7. Avoiding a dependence of the performance on the outer split
The obtain an average performance over severall outer splits (similar to the algorithms in [`methods/ranking.py`](../methods/ranking.py)) ideally several (at least 10) GCNs for different out splits should be trained. 

## Tensor shapes

For $N$ genes, a batch of $B$ disease tasks, hidden width $D$, and disease
embedding width $E$, the main shapes are:

| Quantity | Shape | Meaning |
|---|---:|---|
| Normalized adjacency | `N × N` sparse | Shared PPI graph |
| Input features | `N × B × 2` | Constant and seed channels |
| Disease IDs | `B` | Embedding-table indices |
| GCN output | `N × B × D` | Gene representations |
| Disease embeddings | `B × E` | Disease-specific vectors |
| Scores | `N × B` | One score per gene and task |

For inference on one disease, the batch dimension is omitted.

## Package structure

| File | Responsibility |
|---|---|
| [`data.py`](data.py) | NetworkX-to-tensor conversion and seed features |
| [`model.py`](model.py) | Shared encoder and disease-conditioned scorer |
| [`tasks.py`](tasks.py) | Disease splits, negative pools, leakage checks |
| [`objectives.py`](objectives.py) | Pairwise ranking loss |
| [`training.py`](training.py) | Joint optimization and evaluation |
| [`inference.py`](inference.py) | Scoring seed sets and producing rankings |
| [`main.py`](main.py) | Command-line example |

The complete interactive example is
[`notebooks/GNN.ipynb`](../../notebooks/GNN.ipynb).

## Minimal usage

```python
from bioGraph.data.loading import load_disease_genes, load_ppi_graph
from bioGraph.gcn_prioritization import (
    predict_from_seed_genes,
    train_all_diseases,
)

graph = load_ppi_graph("data/processed/subgraph_5377.txt")
diseases = load_disease_genes("data/raw/pcbi.1004120.s004.txt")

result = train_all_diseases(
    graph,
    diseases,
    hidden_dim=32,
    disease_embedding_dim=16,
    epochs=100,
    keep_details=True,
)

disease_name = "breast neoplasms"
disease_id = result["disease_to_id"][disease_name]
seed_genes = result["disease_results"][disease_name]["train_genes"]

ranking = predict_from_seed_genes(
    result["model"],
    result["graph_data"],
    seed_genes,
    disease_id=disease_id,
)
```

`ranking` is a list ordered from the highest-scoring candidate to the lowest.
Each row contains the gene ID, gene symbol, and model score.
