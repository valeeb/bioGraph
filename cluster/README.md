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

## Rerun only the GCN on existing splits

Use ``--reuse-splits --gcn-only`` to rebuild the GCN array from the current
``gcn_config.json`` without writing the experiment's existing ``splits.pkl``.
The number of array tasks is read from that manifest, so an existing 50-split
experiment produces array indices 0--49 automatically:

```bash
cd /home/fkp/vleeb/bioGraph

/home/fkp/vleeb/miniconda3/envs/qdgp/bin/python cluster/submit.py \
  --experiment YOUR_EXPERIMENT \
  --output-root /disk/data11/tfp/valeeb \
  --gcn-config /home/fkp/vleeb/bioGraph/cluster/gcn_config.json \
  --reuse-splits \
  --gcn-only \
  --submit
```

This regenerates only ``slurm/gcn.slurm`` and replaces completed GCN shards;
the split manifest and classical shards are left unchanged. Omit ``--submit``
to inspect the generated script before submitting it.

## Complete submission example

This example shows the cluster and data inputs commonly configured for a
submission. GCN hyperparameters are read from `cluster/gcn_config.json` rather
than repeated on the command line. Replace `ACCOUNT_NAME` if the cluster
requires an account; otherwise remove the `--account ACCOUNT_NAME` line.

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
  --environment-command ":" \
  --submit
```

Omit `--submit` to generate the split manifest and `.slurm` files without
submitting them. `--gcn-config` can also be omitted when using the default
`cluster/gcn_config.json`. Individual `--gcn-*` arguments remain available for
one-off overrides, but the config file should normally be updated instead. Run
`cluster/submit.py --help` for the complete command-line reference.
