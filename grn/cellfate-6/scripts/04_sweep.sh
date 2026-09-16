#!/bin/bash
# Step 3b: multi-node SHARDED parameter sweep.
#
#   sbatch --nodes=64 --gpus=256 scripts/04_sweep.sh
#
# This is where most of the allocation should go.  Each rank takes grid points
# independently and never initialises a process group, so this layout cannot
# fail on the fabric and tolerates node loss: rerun and finished points are
# skipped.  brics/nccl is loaded anyway so the environment matches step 3a.
#SBATCH --job-name        cellfate-sweep
#SBATCH --nodes           64
#SBATCH --ntasks-per-node 4
#SBATCH --gpus            256
#SBATCH --time            12:00:00
#SBATCH --output          slog/%x-%j.out
set -uo pipefail

SCRATCH=/lus/lfs1aip2/scratch/u6pl/mat611.u6pl/grn/2-grn-gh200
OUTDIR="${SCRATCH}/results/sweep_cytokine"
mkdir -p "${OUTDIR}" slog

source "${SCRATCH}/venv/bin/activate"
module load brics/nccl

export OMP_NUM_THREADS=4
cd "$(dirname "${BASH_SOURCE[0]}")/.."

srun -N "${SLURM_NNODES}" \
     --gpus="${SLURM_GPUS}" \
     --mpi=pmi2 \
     --ntasks-per-node="${SLURM_NTASKS_PER_NODE}" \
     bash -c 'export WORLD_SIZE=$SLURM_GPUS; export RANK=$PMI_RANK; export LOCAL_RANK=$SLURM_LOCALID; \
              cellfate sweep configs/th17_treg_sweep.yaml --outdir '"${OUTDIR}"' --shard-by-rank'

cellfate aggregate "${OUTDIR}" --out "${OUTDIR}/summary"
echo "Sweep complete: ${OUTDIR}/summary.parquet"
