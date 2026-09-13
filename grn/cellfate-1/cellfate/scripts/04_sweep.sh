#!/bin/bash
# Step 3b: multi-node SHARDED parameter sweep.
#
#   sbatch --nodes=64 scripts/04_sweep.sh     # 256 GPUs, one grid point per GPU
#
# This is where most of the allocation should go.  Each rank takes grid points
# independently with no collectives at all, so it scales perfectly and tolerates
# node loss: rerun the script and completed points are skipped.
#SBATCH --job-name=cellfate-sweep
#SBATCH --nodes=64
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

SCRATCH=/lus/lfs1aip2/scratch/u6pl/mat611.u6pl/grn/2-grn-gh200
OUTDIR="${SCRATCH}/results/sweep_cytokine"
mkdir -p "${OUTDIR}"

module load cuda/12.6
# shellcheck disable=SC1091
source "${SCRATCH}/venv/bin/activate"

export OMP_NUM_THREADS=4

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# RANK and WORLD_SIZE come from Slurm here rather than torchrun, because
# --shard-by-rank deliberately avoids initialising a process group at all.
srun --export=ALL bash -c '
  export RANK="${SLURM_PROCID}"
  export WORLD_SIZE="${SLURM_NTASKS}"
  export LOCAL_RANK="${SLURM_LOCALID}"
  cellfate sweep configs/th17_treg_sweep.yaml \
    --outdir "'"${OUTDIR}"'" \
    --device "cuda:${SLURM_LOCALID}" \
    --shard-by-rank
'

cellfate aggregate "${OUTDIR}" --out "${OUTDIR}/summary"

cat <<MSG

Sweep complete.  ${OUTDIR}/summary.parquet holds one row per cytokine point
with MFPT and its standard error.  Rerunning this script skips finished points,
so a partial allocation can be resumed rather than restarted.
MSG
