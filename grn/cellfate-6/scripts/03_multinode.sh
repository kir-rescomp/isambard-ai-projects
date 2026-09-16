#!/bin/bash
# Step 3a: multi-node, ONE shared walker population.
#
#   sbatch --nodes=16 --gpus=64 scripts/03_multinode.sh
#
# Run scripts/03a_nccl_check.sh first.  Do not debug fabric problems through a
# 16-node application job.
#
# Launched with srun directly, one task per GPU, no torchrun.  The nested
# srun-then-torchrun pattern puts elastic agents between Slurm and the workers;
# when anything fails there, every agent takes SIGTERM at once and the log
# becomes a wall of "SignalException: signal 15" with the real cause elsewhere.
#SBATCH --job-name        cellfate-multinode
#SBATCH --nodes           16
#SBATCH --ntasks-per-node 4
#SBATCH --gpus            64
#SBATCH --time            06:00:00
#SBATCH --output          slog/%x-%j.out
set -uo pipefail

SCRATCH=${PWD}
RESULTS="${SCRATCH}/results/step3_multinode/${SLURM_JOB_ID}"
mkdir -p "${RESULTS}" "${SCRATCH}/checkpoints" slog

source "${SCRATCH}/venv/bin/activate"
module load brics/nccl        # site NCCL with the libfabric plugin; see 03a

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=4
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export CELLFATE_DIST_TIMEOUT=600   # fail fast on a fabric problem, do not hang to the wall clock

# Bins are the unit of ownership, so one ensemble can use at most n_bins ranks.
# Beyond that, run independent replicas: 64 GPUs with 32 bins means 16 replicas
# of 4 ranks each.  The spread across replicas is also a better error bar than
# the blocked standard error within a single run.
export CELLFATE_REPLICAS=${CELLFATE_REPLICAS:-16}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "nodes=${SLURM_NNODES} gpus=${SLURM_GPUS} job=${SLURM_JOB_ID}"

srun -N "${SLURM_NNODES}" \
     --gpus="${SLURM_GPUS}" \
     --mpi=pmi2 \
     --ntasks-per-node="${SLURM_NTASKS_PER_NODE}" \
     bash -c 'export WORLD_SIZE=$SLURM_GPUS; export RANK=$PMI_RANK; export LOCAL_RANK=$SLURM_LOCALID; \
              cellfate run ${SCRATCH}/configs/th17_treg.yaml \
                --replicas ${CELLFATE_REPLICAS:-16} \
                --override we.checkpoint_path='"${SCRATCH}"'/checkpoints/th17_mn_${SLURM_JOB_ID} \
                --out '"${RESULTS}"'/th17_treg_multinode.json'

echo "exit code: $?  Results in ${RESULTS}"
