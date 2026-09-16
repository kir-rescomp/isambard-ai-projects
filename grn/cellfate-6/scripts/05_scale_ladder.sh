#!/bin/bash
# Calibrate the system-size parameter BEFORE committing the cytokine grid.
#
#   sbatch --nodes=2 --gpus=8 scripts/05_scale_ladder.sh
#
# At scale=3 the Treg to Th17 mean first passage time is 2.75e8 time units,
# which is 2.75e7 protein lifetimes: a transition that never occurs on any
# biological timescale.  Reported ex-Treg plasticity is a few per cent over
# days, so the target is tens to hundreds of protein lifetimes.  Running the
# cytokine grid at a scale outside the observable range produces a surface that
# reads "never" almost everywhere and carries no information.
#
# Each rank takes one scale value independently; no collectives at all.
#SBATCH --job-name        cellfate-scale-ladder
#SBATCH --nodes           2
#SBATCH --ntasks-per-node 4
#SBATCH --gpus            8
#SBATCH --time            02:00:00
#SBATCH --output          slog/%x-%j.out
set -uo pipefail

SCRATCH=/lus/lfs1aip2/scratch/u6pl/mat611.u6pl/grn/2-grn-gh200
OUTDIR="${SCRATCH}/results/scale_ladder"
mkdir -p "${OUTDIR}" slog

source "${SCRATCH}/venv/bin/activate"
module load brics/nccl
export OMP_NUM_THREADS=4
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SCALES=(1.0 1.25 1.5 1.75 2.0 2.25 2.5 3.0)

srun -N "${SLURM_NNODES}" \
     --gpus="${SLURM_GPUS}" \
     --mpi=pmi2 \
     --ntasks-per-node="${SLURM_NTASKS_PER_NODE}" \
     bash -c 'S='"${SCALES[*]}"'; IFS=" " read -ra A <<< "$S"; V=${A[$PMI_RANK]}; \
              echo "rank $PMI_RANK -> scale $V"; \
              cellfate run configs/th17_treg.yaml \
                --device "cuda:${SLURM_LOCALID}" \
                --override model.scale=$V \
                --override we.n_generations=1200 \
                --out '"${OUTDIR}"'/scale_$V.json --quiet'

echo
echo "MFPT in protein lifetimes (target: tens to hundreds):"
OUTDIR="${OUTDIR}" python3 - <<'PY'
import json, glob, os
for p in sorted(glob.glob(os.environ.get("OUTDIR", "") + "/scale_*.json")):
    d = json.load(open(p))
    m = d["summary"]["mfpt"]
    dp = d["model"].get("d_p", 0.1)
    print(f"  {os.path.basename(p):24s} MFPT {m:.3e}  = {m*dp:.3e} protein lifetimes")
PY
