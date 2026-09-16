#!/bin/bash
# Run this BEFORE any multi-node cellfate job.  Two nodes, a few minutes.
#
#   sbatch scripts/03a_nccl_check.sh
#
# It isolates fabric and launcher configuration from the application: if this
# fails the problem is NCCL or Slurm, not the simulator.
#SBATCH --job-name        cellfate-nccl-check
#SBATCH --nodes           2
#SBATCH --ntasks-per-node 4
#SBATCH --gpus            8
#SBATCH --time            00:20:00
#SBATCH --output          slog/%x-%j.out
set -uo pipefail

SCRATCH=${PWD}
mkdir -p slog
source "${SCRATCH}/venv/bin/activate"

# The site NCCL build.  Load it AFTER activating the environment.  Without it
# NCCL comes from the PyTorch wheel, which has no libfabric plugin and cannot
# use the Slingshot fabric at all.  Single-node jobs still work because NCCL
# stays on NVLink, so a multi-node failure is the first symptom.
module load brics/nccl

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV

cd "$(dirname "${BASH_SOURCE[0]}")/.."

srun -N "${SLURM_NNODES}" \
     --gpus="${SLURM_GPUS}" \
     --mpi=pmi2 \
     --ntasks-per-node="${SLURM_NTASKS_PER_NODE}" \
     bash -c 'export WORLD_SIZE=$SLURM_GPUS; export RANK=$PMI_RANK; export LOCAL_RANK=$SLURM_LOCALID; python3 ${SCRATCH}/scripts/nccl_smoke_test.py'

echo "exit code: $?"
