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

## Step 3a — multi-node, one shared ensemble

```bash
sbatch --nodes=16 scripts/03_multinode.sh
```

Use this when one cytokine point needs more walkers than a node holds. Confirm
`NCCL_SOCKET_IFNAME` matches the actual high-speed interface (`ip -o link` on a
compute node); an unset value silently falls back to a slow path rather than
failing, which is easy to miss.

## Step 3b — sharded sweep

```bash
sbatch --nodes=64 scripts/04_sweep.sh
```

This is where most of the allocation should go. No collectives at all, perfect
scaling, and node loss is survivable: rerun and completed grid points are
skipped. Aggregation into a single Parquet table runs automatically at the end.

## Sizing the campaign

After step 1 you can compute the whole budget. With `T` = walker-steps/s/GPU:

```
one WE run  = n_generations x n_substeps x (walkers_per_bin x occupied_bins)
            = 3000 x 200 x (128 x 32)  ~= 2.5e9 walker-steps
GPU-seconds = 2.5e9 / T
```

At a plausible 10^8 walker-steps/s this is about 25 GPU-seconds per run, which
means the 49-point cytokine sweep costs well under a GPU-hour. That is the
point made earlier: this workload is efficient enough that the science must be
scaled deliberately to justify the hardware. The three axes, in the order they
should be added:

1. **Cytokine plane** — 7 x 7, already in `configs/th17_treg_sweep.yaml`.
2. **Parameter ensemble** — 200 to 500 draws over the uncertain kinetic
   constants per cytokine point. This is the scientifically important one: it
   converts the standard criticism of hand-curated GRN models into the thing
   the compute buys you.
3. **Committor sensitivity** — per-species perturbation at the barrier, which
   produces the ranked list of transition-controlling molecules and hence the
   experimentally testable prediction.

The product is 10^4 to 10^5 independent runs, which is a campaign of the right
shape for hundreds of GH200s.

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
