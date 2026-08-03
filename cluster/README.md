# SLURM simulations

The cluster workflow creates every outer train/test split once and then runs two
independent SLURM arrays against those immutable splits:

- `classical`: aNBR, rNBR, RWR, DK, DK*, QA0, QA1, QA*, and DIAMOND;
- `gcn`: one jointly trained disease-conditioned GCN.

Each array has one task per outer split. With `N=30`, this means 30 classical
tasks and 30 GCN tasks. Every task covers all diseases. Results are first
written under `/scratch/$USER`, copied to shared storage only after completion,
and kept as independently retryable shards.

From the repository root, generate the split manifest and SLURM scripts:

```bash
python cluster/submit.py -N 30 -p standard \
  --environment-command "source /path/to/environment/bin/activate"
```

This does not submit anything by default. Inspect the generated `.slurm` files,
then either add `--submit` to the command or submit both generated scripts with
`sbatch`. For example, to generate and submit immediately:

```bash
python cluster/submit.py -N 30 -p standard --submit \
  --environment-command "source /path/to/environment/bin/activate"
```

The default experiment directory is
`outputs/results/cluster/benchmark/`. Use `--experiment NAME` to keep distinct
configurations separate.

After both arrays finish, combine and validate all shards:

```bash
python -m cluster.sim collect \
  --manifest outputs/results/cluster/benchmark/splits.pkl \
  --shard-root outputs/results/cluster/benchmark/shards \
  --output outputs/results/cluster/benchmark/results.pkl
```

Collection fails rather than silently producing partial results if a shard is
missing, malformed, uses another node order, or does not match the shared split
manifest. The combined pickle contains full score vectors for every method,
disease, and outer split.
