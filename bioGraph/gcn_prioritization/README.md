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
H_d^{(1)} = \operatorname{Dropout}\!\left[
  \operatorname{ReLU}(\tilde A X_d W_1 + b_1)
\right],
$$

$$
U_d^{(2)} = \operatorname{Dropout}\!\left[
  \operatorname{ReLU}(\tilde A H_d^{(1)} W_2 + b_2)
\right],
\qquad
H_d = H_d^{(1)} + U_d^{(2)}.
$$

The final addition is a residual connection. It gives the model access to both
the first propagation result and its two-step update, and also helps gradients
flow through the network during optimization.

The row $h_{i,d}$ of $H_d$ is the hidden representation of gene $i$ for
disease task $d$.
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
z_{i,d} = [h_{i,d}; e_d].
$$

A small nonlinear network maps $z_{i,d}$ to a scalar. The model also includes
an explicit multiplicative interaction between the gene and disease vectors:

$$
s_{i,d} = \operatorname{MLP}(z_{i,d})
 + \frac{h_{i,d}^\mathsf{T}P e_d}{\sqrt{H}},
$$

where $P$ projects the disease embedding into the $H$-dimensional gene
representation space.

The nonlinear and multiplicative interactions are important. If the embedding
entered through only a linear additive term, it would add the same constant to
every gene score for a disease. That constant would cancel from the pairwise
ranking loss and could not change the gene ordering.

This is implemented by `DiseaseConditionedGCN` in [`model.py`](model.py).

## 4. Training objective

For each disease, its known genes are first divided into:

- **outer-training genes**, which may be used during optimization;
- **held-out test genes**, whose associations are revealed only for final
  evaluation.

During each epoch, the outer-training genes are split again. One subset becomes
the visible seed indicator, and the other supplies positive training labels.
Comparison genes are sampled from all graph genes except the current seed and
positive-target genes. They are often called "sampled negatives," but they are
not verified biological negatives. The loss treats them as lower-ranked
examples for that training step, so they are not mathematically neutral.

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

The disease associations of the outer-test genes are completely hidden during
optimization. They are never supplied as:

- visible input seeds;
- positive training labels.

Thus, for each disease, there is a subset of known associations whose disease
labels are never shown to the model during optimization. The genes themselves
remain ordinary nodes in the PPI graph and participate in message passing. They
are also eligible for random selection as unlabelled comparison genes, exactly
like every other gene outside the outer-training set. The comparison-pool code
does not receive the outer-test set and therefore cannot filter on it.

The split validation and task construction are
centralized in [`tasks.py`](tasks.py).

## 6. Testing the performance
After training is complete, the performance of the model for the given outer split is evaluated. 

For each disease, all outer-training genes are marked as seeds in one
seed-indicator vector. The model then scores every graph node. Known seeds are
removed from the candidate ranking, which is evaluated against the held-out
positive genes. This can be done for every disease because a separate subset of
its known genes was held out.

## 7. Avoiding a dependence of the performance on the outer split
To obtain an average performance over several sets of outer splits (similar to
the algorithms in [`methods/ranking.py`](../methods/ranking.py)), several joint
GCN models should be trained using independently generated sets of outer
splits.


## Data splitting

For each disease, the associated genes are split into two stages.

```text
outer_split = 0.75
seed_split  = 2/3
```

### Outer split

The outer split determines which disease genes are available during training and which are kept completely hidden until the final evaluation.

Example:

| Disease | Associated genes |
|---------|------------------|
| d₁ | 1, 2, 3, 4, 5, 6, **[7, 8]** |
| d₂ | 2, 8, 9, **[10]** |

Genes inside **[...]** are never used as training labels. They are only used for the final evaluation.

### Training

For every disease, multiple training samples are generated by repeatedly splitting the visible genes into

- **seed genes** (model input)
- **target genes** (positive labels)

Example:

#### d₁

Visible genes: `{1,2,3,4,5,6}`

```text
Seeds        → Targets

1,2,3,4      → 5,6
1,3,5,6      → 2,4
2,4,5,6      → 1,3
...
```

#### d₂

Visible genes: `{2,8,9}`

```text
Seeds        → Targets

2,8          → 9
2,9          → 8
8,9          → 2
...
```

All diseases contribute such training samples, and a single shared GCN is trained jointly on them.

### Testing

After training, the model is evaluated on the held-out outer split.

For each disease:

```text
d₁

Input (seeds):
1,2,3,4,5,6

Rank:
all nodes in the graph

Ground truth:
7,8
```

```text
d₂

Input (seeds):
2,8,9

Rank:
all nodes in the graph

Ground truth:
10
```

The ranking is evaluated using the hidden outer-split genes only.

### Cross-validation

After evaluation, a new outer split is generated and the entire procedure is repeated.

Example:

| Disease | Associated genes |
|---------|------------------|
| d₁ | 1, 2, 3, **[4, 5]**, 6, 7, 8 |
| d₂ | **[2]**, 8, 9, 10 |

A new GCN is initialized and trained from scratch on the new split before being evaluated again.

## Tensor shapes

For $N$ genes, a batch of $B$ disease tasks, hidden width $H$, and disease
embedding width $E$, the main shapes are:

| Quantity | Shape | Meaning |
|---|---:|---|
| Normalized adjacency | `N × N` sparse | Shared PPI graph |
| Input features | `N × B × 2` | Constant and seed channels |
| Disease IDs | `B` | Embedding-table indices |
| GCN output | `N × B × H` | Gene representations |
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
