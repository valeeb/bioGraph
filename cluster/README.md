# Running simulations on the SLURM cluster

This workflow runs 30 reproducible outer splits across all diseases. It creates
two SLURM arrays:

- `classical`: aNBR, rNBR, RWR, DK, DK*, QA0, QA1, QA*, and DIAMOND;
- `gcn`: one jointly trained disease-conditioned GCN.

Each array has one task per outer split, giving 60 tasks for `N=30`. Both tasks
for a given index consume the same saved disease-by-disease splits. Workers
write temporary results to `/scratch/$USER`, copy completed pickle shards to
`/disk/data11/tfp/valeeb`, and then remove their scratch directory.

## Fixed cluster paths

```text
Project:     /home/fkp/vleeb/bioGraph
Python:      /home/fkp/vleeb/miniconda3/envs/qdgp/bin/python
Results:     /disk/data11/tfp/valeeb/<experiment>/
Partition:   standard
Time limit:  24 hours
Memory:      16 GB per task
CPUs:        1 per task
```

There is no need to activate the Conda environment. The commands call its
Python executable directly.

## 1. Install the runtime dependencies

The environment needs NumPy, SciPy, NetworkX, tqdm, and PyTorch. Install the
repository's cluster requirements with:

```bash
cd /home/fkp/vleeb/bioGraph

/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python -m pip install \
  -r /home/fkp/vleeb/bioGraph/cluster/requirements.txt
```

This command requires access to a Python package index. If the login node has
no package-index access, ask the cluster administrator which local package
mirror or module should be used for these dependencies.

## 2. Check the installation

Log into the cluster and copy this entire block:

```bash
set -e
cd /home/fkp/vleeb/bioGraph

test -x /home/fkp/vleeb/miniconda3/envs/qdgp/bin/python \
  && echo "Python executable found"
test -r /home/fkp/vleeb/bioGraph/data/raw/PPI202207.txt \
  && echo "PPI input found"
test -r /home/fkp/vleeb/bioGraph/data/raw/pcbi.1004120.s004.txt \
  && echo "Disease input found"
test -w /disk/data11/tfp/valeeb \
  && echo "Result directory is writable"

/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python -c \
  "import networkx, numpy, scipy, torch; import bioGraph; print('Environment OK')"
```

Every command should succeed, and the final command should print
`Environment OK`. Because `set -e` is enabled, the block stops immediately if
any check fails. Do not submit jobs until the final import check succeeds.

If the import check fails, compare the activated environment with the configured
absolute interpreter:

```bash
which python
readlink -f "$(which python)"
python -c "import sys; print(sys.executable)"
python -c "import networkx, numpy, scipy, torch; print('Active Python OK')"

/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python -c \
  "import sys; print(sys.executable)"
/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python -m pip show \
  networkx numpy scipy torch
```

Use the interpreter for which the complete import command succeeds. If neither
one succeeds, the environment dependencies must be installed before submitting
the simulations.

## 3. Generate and submit the jobs

Choose a unique experiment name. The example below uses `benchmark_30`:

```bash
cd /home/fkp/vleeb/bioGraph

/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python cluster/submit.py \
  -N 30 \
  --base-seed 0 \
  --split-fraction 0.75 \
  --experiment benchmark_30 \
  --output-root /disk/data11/tfp/valeeb \
  --python /home/fkp/vleeb/miniconda3/envs/qdgp/bin/python \
  -p standard \
  --time 24:00:00 \
  --memory 16G \
  --cpus 1 \
  --gcn-epochs 100 \
  --submit
```

This creates and submits:

```text
/disk/data11/tfp/valeeb/benchmark_30/
  splits.pkl
  logs/
  shards/
  slurm/classical.slurm
  slurm/gcn.slurm
```

Do not regenerate `splits.pkl` with different parameters while its jobs are
running. Use another experiment name for a different configuration.

## 4. Monitor the jobs

```bash
squeue -u "$USER"
```

SLURM output and error logs appear in:

```text
/disk/data11/tfp/valeeb/benchmark_30/logs/
```

After completion, there should be 30 files of each kind:

```bash
find /disk/data11/tfp/valeeb/benchmark_30/shards \
  -name classical.pkl -type f | wc -l

find /disk/data11/tfp/valeeb/benchmark_30/shards \
  -name gcn.pkl -type f | wc -l
```

Both commands should print `30`.

## 5. Collect the results

Only run collection after both arrays have finished:

```bash
cd /home/fkp/vleeb/bioGraph

/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python -m cluster.sim collect \
  --manifest /disk/data11/tfp/valeeb/benchmark_30/splits.pkl \
  --shard-root /disk/data11/tfp/valeeb/benchmark_30/shards \
  --output /disk/data11/tfp/valeeb/benchmark_30/results.pkl
```

The final combined result is:

```text
/disk/data11/tfp/valeeb/benchmark_30/results.pkl
```

It contains full score vectors for every method, disease, and outer split.
Collection stops with an error instead of producing partial results if a shard
is missing, malformed, uses a different node order, has different
hyperparameters, or does not match the shared split manifest.

## Longer runs

If 24 hours is insufficient, create a new experiment on the `long` partition:

```bash
cd /home/fkp/vleeb/bioGraph

/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python cluster/submit.py \
  -N 30 \
  --experiment benchmark_30_long \
  --python /home/fkp/vleeb/miniconda3/envs/qdgp/bin/python \
  -p long \
  --time 72:00:00 \
  --memory 16G \
  --cpus 1 \
  --gcn-epochs 100 \
  --submit
```

If this cluster requires a SLURM account, add `--account ACCOUNT_NAME` to the
submission command.
