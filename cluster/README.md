# Running simulations on the SLURM cluster

The workflow creates one shared outer-split manifest and submits two SLURM
arrays:

- `classical`: aNBR, rNBR, RWR, DK, DK*, QA0, QA1, QA*, and DIAMOND;
- `gcn`: the jointly trained disease-conditioned GCN.

Every array task evaluates all diseases. Both method groups use exactly the
same saved train/test splits. Pickle shards and the final result are written
directly under `/disk/data11/tfp/valeeb/<experiment>/`. A job-specific temporary
filename prevents an incomplete task from appearing as a completed shard.

GCN hyperparameters are configured in:

```text
/home/fkp/vleeb/bioGraph/cluster/gcn_config.json
```

Edit that file before submitting. The resolved configuration is copied into
the experiment directory as `gcn_config.json`.

## Minimal working example

This submits 30 outer splits to the `standard` partition using the repository's
defaults:

```bash
cd /home/fkp/vleeb/bioGraph

/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python cluster/submit.py \
  --experiment benchmark_30 \
  --submit
```

Monitor the jobs with:

```bash
squeue -u "$USER"
```

After both arrays have finished, collect the shards:

```bash
cd /home/fkp/vleeb/bioGraph

/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python -m cluster.sim collect \
  --manifest /disk/data11/tfp/valeeb/benchmark_30/splits.pkl \
  --shard-root /disk/data11/tfp/valeeb/benchmark_30/shards \
  --output /disk/data11/tfp/valeeb/benchmark_30/results.pkl
```

The combined result is:

```text
/disk/data11/tfp/valeeb/benchmark_30/results.pkl
```

Use a new experiment name when changing parameters. Do not regenerate an
experiment's split manifest while its jobs are running.

## Fully explicit submission example

This example shows every input accepted by `cluster/submit.py`. Replace
`ACCOUNT_NAME` if the cluster requires an account; otherwise remove the
`--account ACCOUNT_NAME` line.

```bash
cd /home/fkp/vleeb/bioGraph

/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python cluster/submit.py \
  --num-splits 30 \
  --base-seed 0 \
  --split-fraction 0.75 \
  --experiment benchmark_30_explicit \
  --output-root /disk/data11/tfp/valeeb \
  --ppi-path /home/fkp/vleeb/bioGraph/data/raw/PPI202207.txt \
  --disease-path /home/fkp/vleeb/bioGraph/data/raw/pcbi.1004120.s004.txt \
  --partition standard \
  --time 24:00:00 \
  --memory 16G \
  --cpus 1 \
  --account ACCOUNT_NAME \
  --python /home/fkp/vleeb/miniconda3/envs/qdgp/bin/python \
  --gcn-config /home/fkp/vleeb/bioGraph/cluster/gcn_config.json \
  --gcn-epochs 100 \
  --gcn-hidden-dim 32 \
  --gcn-disease-embedding-dim 16 \
  --gcn-learning-rate 0.01 \
  --gcn-weight-decay 0.0001 \
  --gcn-negative-ratio 5 \
  --gcn-inner-seed-fraction 0.6666666666666666 \
  --gcn-task-batch-size 8 \
  --environment-command ":" \
  --submit
```

Omit `--submit` to generate the split manifest and `.slurm` files without
submitting them. The individual `--gcn-*` arguments override values read from
`--gcn-config`; they are shown above only because this example explicitly sets
every possible input. Run `cluster/submit.py --help` for the command-line
reference.
