# cellfate

GPU-native weighted-ensemble stochastic simulation of gene regulatory networks,
for mapping cell-fate landscapes and measuring the rate of rare fate
transitions.

One GPU carries 10^5 to 10^6 independent stochastic cells. Hundreds of GPUs
carry the weighted-ensemble population that resolves transitions too rare for
direct simulation to observe at all. The scientific output is a transition
rate, a committor function, and a quasi-potential landscape, not a pile of
trajectories.

---

## What the code actually computes

Direct stochastic simulation of a bistable regulatory network spends
essentially all of its time waiting at the bottom of a basin. The weighted
ensemble method carries a population of weighted walkers, bins them along a
progress coordinate, and resamples so that under-populated regions receive
effort in proportion to interest rather than in proportion to probability.
Weight is conserved exactly, so the resulting rate estimate is unbiased.

Under recycling boundary conditions the steady-state probability flux into the
target basin is the reciprocal of the mean first passage time. That is the
number the campaign is built to produce, together with:

- the **committor** `q(x)`, the probability of reaching the target basin before
  returning to the source, obtained from the bin-to-bin Markov state model.
  The committor is the natural reaction coordinate, and its sensitivity to
  individual species is what yields an experimentally testable prediction about
  which molecules control the transition.
- the **quasi-potential**, `-log` of the stationary bin occupancy.

The default biological model is a RORgt / FOXP3 mutual-repression switch with
explicit mRNA, cytokine inputs entering through STAT3 (IL-6) and STAT5
(IL-2 with TGF-beta). Bursty transcription at low mRNA copy number is the noise
source that drives the transition, which is why the mRNA layer is modelled
explicitly rather than adiabatically eliminated.

## Validation status

`cellfate selftest` runs, and must pass, on any new build:

| Check | Criterion | Measured (CPU reference run) |
|---|---|---|
| Propensity evaluation vs NumPy definition | max relative error < 1e-10 | 0.0 for all three models |
| Birth-death stationary mean, Poisson(k/d) | within 0.4 of 20.0 | 20.006 |
| Birth-death stationary variance | within 1.2 of 20.0 | 20.497 |
| Weight conservation over 1500 generations | \|W - 1\| < 1e-9 | 1.000000000000002 |
| **WE mean first passage time vs brute force** | ratio in [0.6, 1.7] | **548.1 ± 22.7 vs 546.2, ratio 1.004** |
| Committor monotone in the progress coordinate | non-decreasing | passes |
| Committor at the midpoint of a symmetric barrier | in [0.25, 0.75] | 0.467 |

The rate check is the one that matters. It compares the weighted-ensemble
estimate against direct simulation on a deliberately low barrier
(`toggle`, `scale=0.5`) where brute force can still cross. Agreement to
0.4 per cent on a barrier that brute force needs 3000 walkers to resolve is the
evidence that the resampling and weight bookkeeping are correct.

## Install

Isambard-AI is aarch64 with Grace Hopper superchips, so wheel availability is
the first thing to establish, not the last.

```bash
bash scripts/00_build_env.sh          # uv venv + aarch64 CUDA torch + editable install
source .venv/bin/activate
cellfate selftest --device cuda:0     # must print "0 failure(s)"
```

If a CUDA-enabled aarch64 torch wheel is unavailable for the driver on the
machine, fall back to the NVIDIA container and install `cellfate` inside it.
The package itself has no compiled dependencies beyond torch and numpy.

## Scaling ladder

The three steps below are the intended order and each has a script.

### 1. One GPU, interactive

```bash
srun --partition=workq --gres=gpu:1 --time=1:00:00 --pty bash
source .venv/bin/activate

cellfate selftest --device cuda:0          # correctness
cellfate bench --walkers 200000 --steps 200 --json bench_1gpu.json
cellfate run configs/toggle_validation.yaml --out results/validation.json
```

`bench` reports walker-steps per second. Record it: it is the denominator for
every projection you make about the size of the campaign, and it is also the
number that goes in a grant report.

Sweep `--walkers` from 10^4 to 10^7 to find where the GPU saturates. On a
GH200 the interesting question is how far past HBM capacity the coherent
NVLink-C2C host memory lets you push before throughput falls off, since the
walker population is the only large array in the code.

### 2. Four GPUs, one node

```bash
sbatch scripts/02_node_4gpu.sh
```

Uses `torchrun --nproc_per_node=4`. Resampling is exact across the whole node:
walkers migrate to the rank owning their bin via one all-to-all per generation,
so the four GPUs share a single walker population rather than running four
independent ensembles. Confirm that total weight remains 1.0 and that the MFPT
matches the single-GPU value within error.

### 3. Multi-node

```bash
sbatch --nodes=16 scripts/03_multinode.sh          # 64 GPUs, one shared ensemble
sbatch --nodes=64 scripts/04_sweep.sh              # 256 GPUs, sharded parameter sweep
```

Two layouts, and the choice matters:

- **One shared ensemble** (`cellfate run`) puts every rank on the same walker
  population. Use it when a single cytokine point needs more walkers than one
  node holds, or when the barrier is high enough that bin coverage is the
  binding constraint. Communication is one all-to-all of walker payloads per
  generation, scaling with bin count rather than walker count.
- **Sharded sweep** (`cellfate sweep --shard-by-rank`) gives each rank its own
  grid point with no collectives at all. Use it for the cytokine grid and for
  parameter-ensemble work, which is where most of the allocation should go.

## Where the compute should go

A single transition rate does not need hundreds of GPUs; a well-written leaping
kernel is fast enough that one point converges in tens of GPU-hours. The
allocation is justified by deliberately scaling the question:

1. **Cytokine plane.** 7 x 7 grid of IL-6 against TGF-beta with IL-2, giving the
   transition rate as a surface rather than a number.
2. **Parameter ensemble.** The kinetic constants of any hand-curated regulatory
   model are uncertain. Sampling them, rather than asserting them, converts the
   main weakness of this class of modelling into the thing the compute buys.
   Budget 200 to 500 parameter draws per cytokine point.
3. **Committor sensitivity.** Per-species perturbation of the converged model at
   the barrier, which is the step that produces the ranked list of
   transition-controlling molecules.

That product is 10^4 to 10^5 independent weighted-ensemble runs, which is a
campaign of the right shape for the hardware.

## Layout

```
src/cellfate/
  model.py                 network specification, compilation, NumPy reference propensities
  models/__init__.py       birth_death, toggle, th17_treg, progress coordinates
  engine/torch_backend.py  vectorised tau-leaping, recursive subdivision, tau selection
  engine/ssa.py            exact Gillespie SSA, validation ground truth
  we/resample.py           binning, systematic and split/merge resampling
  we/stats.py              flux, MFPT, Markov matrix, committor, quasi-potential
  we/driver.py             generation loop, recycling, checkpointing, brute-force reference
  dist/comm.py             bin ownership, walker all-to-all, collective reductions
  io/results.py            provenance capture and result serialisation
  cli.py                   selftest, bench, run, sweep, aggregate
scripts/                   environment build and the Slurm ladder
configs/                   production, sweep, and validation configurations
tests/                     pytest suite, runs on CPU in about a minute
```

## Known limitations

- The leaping stepper is a synchronised tau-leap with recursive subdivision on
  negative populations, not an exact SSA. Accuracy is validated against exact
  SSA and against analytic moments, but any change to `tau` should be
  re-validated with `cellfate selftest`.
- Clamping at maximum subdivision depth introduces bias. The count is reported
  in every generation log and should be zero; if it is not, reduce `tau`.
- The progress coordinate is one-dimensional. Multi-dimensional binning is a
  natural extension and the `BinScheme` interface is where it belongs.
- The fused CUDA kernel is deliberately not written yet. It comes after
  benchmarking: if the torch backend already saturates a GH200, the kernel is
  wasted effort.
- Everything was validated on CPU with the gloo backend. The NCCL path and the
  aarch64 wheels are unverified until step 1 runs on the machine.
