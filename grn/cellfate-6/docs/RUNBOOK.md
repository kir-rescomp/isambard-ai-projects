# Runbook

Everything here assumes:

```
SCRATCH=/lus/lfs1aip2/scratch/u6pl/mat611.u6pl/grn/2-grn-gh200
module load cuda/12.6
source ${SCRATCH}/venv/bin/activate
```

## Step 0 — build the environment

```bash
bash scripts/00_build_env.sh
```

This fails fast if the aarch64 CUDA torch wheel is unavailable, which is the
only real deployment risk in the whole project. It prints the device name,
compute capability and HBM size, so keep the output.

## Step 1 — one GPU, interactive

```bash
srun --nodes=1 --gpus-per-node=1 --time=1:00:00 --pty bash
bash scripts/01_interactive_1gpu.sh
```

Three gates before moving on:

1. `cellfate selftest --device cuda:0` prints `0 failure(s)`.
2. Every generation log shows `clamped=0`. If not, halve `we.tau` and rerun.
   Clamping is the only source of bias in the stepper and it must be zero.
3. You have a walker-steps/s figure and know the walker count at which the GPU
   saturates.

The throughput sweep runs 10^4 to 10^7 walkers. The GH200 question worth
answering here is how far past HBM the coherent NVLink-C2C host memory lets the
walker array grow before throughput falls off, since the walker population is
the only large array in the code. That number is also a clean result for a
GH200 benchmarking report in its own right.

## Step 2 — four GPUs, one node

```bash
sbatch scripts/02_node_4gpu.sh
```

The four GPUs share one walker population; walkers migrate to the rank owning
their bin via a variable-length all-to-all each generation. Two things to check
in the output:

- total weight stays at 1.0 to about 1e-15
- the MFPT from `toggle_validation_4gpu.json` matches the single-GPU value from
  step 1 within the reported standard error

Rank-invariance of the rate was verified on CPU with gloo at 1, 2 and 4 ranks
(513 / 598 / 526, all overlapping within error), so a discrepancy here points at
NCCL configuration rather than at the algorithm.

## Step 3 — multi-node

### Isambard-AI launch requirements

Three site-specific things, all of them mandatory and none of them optional
niceties:

1. **`module load brics/nccl`, loaded after activating the venv.** The NCCL
   bundled in the PyTorch wheel has no libfabric plugin and cannot use the
   Slingshot fabric. Single-node jobs succeed anyway because NCCL stays on
   NVLink, so a multi-node failure is the first symptom. Every run records the
   loaded library path in its provenance block; a path under
   `site-packages/nvidia/nccl` means the module did not take effect.
2. **`srun --mpi=pmi2`**, with `RANK=$PMI_RANK`, `WORLD_SIZE=$SLURM_GPUS` and
   `LOCAL_RANK=$SLURM_LOCALID` exported inside the task.
3. **`--gpus=<total>` rather than `--gpus-per-node`** at the sbatch level, since
   `SLURM_GPUS` is what the rank derivation reads.

Do not set `NCCL_SOCKET_IFNAME`. With `brics/nccl` loaded, NCCL selects the
fabric itself, and forcing an interface name is a way to break a working setup.

### Step 3a — fabric check first

```bash
sbatch scripts/03a_nccl_check.sh          # 2 nodes, ~5 minutes
```

Checks `all_reduce`, variable-split `all_to_all_single`, and bus bandwidth.
Anything under about 5 GB/s means a TCP fallback rather than Slingshot. Do not
debug fabric problems through a 16-node application job.

### Replicas: how the shared-ensemble layout scales

Bins are the unit of ownership, so **one ensemble can use at most `n_bins`
ranks**. Beyond that, additional ranks receive no walkers. Running 64 GPUs
against a 32-bin coordinate leaves 32 ranks idle and, before this was guarded,
crashed them.

Past the bin count, scale by running independent replicas:

```
world_size / replicas <= n_bins
```

64 GPUs with 32 bins means `--replicas 16`, giving 16 ensembles of 4 ranks.
Each replica is a complete ensemble carrying total probability 1.0, and the
spread of their rate estimates is a better error bar than the blocked standard
error within any single run. Oversubscription now raises a clear error at
construction naming all three remedies.

### Step 3b — multi-node, one shared ensemble

```bash
sbatch --nodes=16 scripts/03_multinode.sh
```

Use this when one cytokine point needs more walkers than a node holds. Confirm
`NCCL_SOCKET_IFNAME` matches the actual high-speed interface (`ip -o link` on a
compute node); an unset value silently falls back to a slow path rather than
failing, which is easy to miss.

### Step 3c — calibrate `scale`, then sweep

Measured at 64 GPUs, 16 replicas, scale=3: MFPT 2.7518e8 +/- 6.8e5, which is
2.75e7 protein lifetimes. That is outside any observable range, so calibrate
before spending the grid:

```bash
sbatch --nodes=2 --gpus=8 scripts/05_scale_ladder.sh
```

Pick the scale whose MFPT lands in the tens-to-hundreds of protein lifetimes
implied by reported ex-Treg plasticity, then centre the cytokine grid there.

### Step 3d — sharded sweep

```bash
sbatch --nodes=64 scripts/04_sweep.sh
```

This is where most of the allocation should go. No collectives at all, perfect
scaling, and node loss is survivable: rerun and completed grid points are
skipped. Aggregation into a single Parquet table runs automatically at the end.

## Sizing the campaign

Measured on Isambard-AI, GH200 120GB, torch 2.14 + CUDA 12.6, NCCL 2.29.3:

| Quantity | Measured |
|---|---|
| Single GPU, float64, 10^6 walkers | 3.17e8 walker-steps/s |
| Four GPUs, aggregate | 1.39e9 walker-steps/s (essentially linear) |
| Peak GPU memory, 10^6 walkers | 0.63 GiB |
| Fixed cost per WE generation | about 99 ms |
| Toggle validation, 4 GPUs | MFPT 534 +/- 18, against 522 on one GPU |
| Th17/Treg, scale=3, 3000 generations | MFPT 2.74e8 +/- 1.9e7 |

**The production configuration was using 0.6 per cent of the GPU.** At
`walkers_per_bin=128` the whole population is 3,968 walkers, about 992 per GPU.
The arithmetic takes 0.63 ms per generation and the measured generation takes
100 ms, so the run is entirely dominated by fixed per-generation cost: kernel
launches for 200 substeps, the per-bin resampling loop, and the all-to-all.

Compute equals overhead at roughly 630,000 walkers, about 20,000 per bin. Below
that, walkers are close to free:

| walkers_per_bin | total walkers | ms/generation | wall cost | error bars |
|---|---|---|---|---|
| 128 (old default) | 3,968 | 100 | 1.00x | baseline |
| 512 | 15,872 | 102 | 1.02x | 2x smaller |
| 2,048 | 63,488 | 109 | 1.09x | 4x smaller |
| 8,192 (new default) | 253,952 | 139 | 1.39x | 8x smaller |
| 32,768 | 1,015,808 | 260 | 2.59x | 16x smaller |

`configs/th17_treg.yaml` now uses 8,192 and the sweep config 4,096. This is a
free eightfold improvement in statistical resolution, and it is why the CUDA
kernel should stay unwritten: the device is not the bottleneck and would not be
even after a 250-fold increase in population.

### Cost of the campaign

One 3,000-generation run costs about 0.33 GPU-hours at the old settings, and
about 0.46 at the new ones.

| Campaign | GPU-hours |
|---|---|
| 49 cytokine points, one parameter draw each | ~23 |
| 49 points x 100 parameter draws | ~2,250 |
| 49 points x 300 parameter draws | ~6,750 |

The parameter ensemble is what makes this a campaign rather than an afternoon,
and it is also the scientifically strongest axis: it converts the standard
objection to hand-curated regulatory models into the thing the compute buys.

### Calibrating `scale` before spending the allocation

`scale=3` gives a Treg to Th17 mean first passage time of 2.7e8 time units.
The protein lifetime in the model is 10 time units, so that is 2.7e7 protein
lifetimes: a transition that never happens on any biological timescale. A rate
that far outside the observable range makes the cytokine surface uninformative
over most of its area.

Run a short `scale` ladder before committing to the grid:

```bash
for S in 1.0 1.5 2.0 2.5 3.0; do
  cellfate run configs/th17_treg.yaml --device cuda:0 \
      --override model.scale=$S --override we.n_generations=800 \
      --out results/scale_ladder/scale_$S.json
done
```

Choose the `scale` whose symmetric-point MFPT sits in the range implied by
observed ex-Treg plasticity, then run the cytokine grid around it. Reported
Treg to Th17 conversion is a few per cent over days, so the target is tens to
hundreds of protein lifetimes, not tens of millions. Fixing this before the
grid is the difference between a surface with structure and a surface that
reads "never" everywhere.

## Filesystem discipline

Results are one JSON document per run plus one `.npy` transition matrix. At
10^5 runs that is 2 x 10^5 files, which is enough to notice on Lustre. Run
`cellfate aggregate` and delete the per-run documents once a sweep is complete,
or write sweeps into per-block subdirectories. Raw trajectories are never
written by design.

## Known limitations

- The stepper is a synchronised tau-leap with recursive subdivision, not exact
  SSA. Validated against exact SSA and analytic moments; revalidate after any
  change to `tau`.
- The progress coordinate is one-dimensional. `BinScheme` is where
  multi-dimensional binning belongs.
- `src/cellfate/cuda/` is empty by agreement: the kernel comes after
  benchmarking, because if the torch backend already saturates the device the
  kernel is wasted effort.
- Everything was validated on CPU with the gloo backend. The NCCL path and the
  aarch64 wheels are unverified until step 1 runs.
