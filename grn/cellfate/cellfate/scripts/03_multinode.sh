#!/bin/bash
# Step 3a: multi-node, ONE shared walker population.
#
#   sbatch --nodes=16 scripts/03_multinode.sh     # 64 GPUs
#
# Use this layout when a single cytokine point needs more walkers than one node
# holds, or when bin coverage rather than trajectory count is the binding
# constraint.  Communication is one variable-length all-to-all of walker
# payloads per generation, scaling with bin count rather than walker count.
#SBATCH --job-name=cellfate-multinode
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail

SCRATCH=/lus/lfs1aip2/scratch/u6pl/mat611.u6pl/grn/2-grn-gh200
RESULTS="${SCRATCH}/results/step3_multinode/${SLURM_JOB_ID}"
mkdir -p "${RESULTS}" "${SCRATCH}/checkpoints"

module load cuda/12.6
# shellcheck disable=SC1091
source "${SCRATCH}/venv/bin/activate"

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
# Slingshot tunables.  Confirm the interface name with `ip -o link` on a compute
# node; hsn0 is the usual name but sites differ, and an unset value silently
# falls back to a slow path rather than failing.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
export NCCL_IB_DISABLE=0
export FI_CXI_DEFAULT_CQ_SIZE=131072

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# One task per node launches torchrun for that node's four GPUs.  This is more
# reliable on Cray systems than letting torchrun handle the rendezvous itself.
srun --ntasks-per-node=1 --gpus-per-node=4 bash -c '
  torchrun \
    --nnodes="${SLURM_NNODES}" \
    --node_rank="${SLURM_NODEID}" \
    --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m cellfate.cli run configs/th17_treg.yaml \
      --override we.checkpoint_path="'"${SCRATCH}"'/checkpoints/th17_mn_${SLURM_JOB_ID}" \
      --out "'"${RESULTS}"'/th17_treg_multinode.json"
'

echo "Step 3a complete.  Results in ${RESULTS}"
