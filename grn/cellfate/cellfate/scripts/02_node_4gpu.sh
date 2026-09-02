#!/bin/bash
# Step 2 of the ladder: four GPUs, one node, one shared walker population.
#
#   sbatch scripts/02_node_4gpu.sh
#
# Set --partition, --account and --qos for your site before submitting.
#SBATCH --job-name=cellfate-4gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=slog/%x-%j.out

set -euo pipefail

SCRATCH=/lus/lfs1aip2/scratch/u6pl/mat611.u6pl/grn/2-grn-gh200
RESULTS="${SCRATCH}/results/step2_4gpu"
mkdir -p "${RESULTS}" "${SCRATCH}/checkpoints"

module load cuda/12.6
# shellcheck disable=SC1091
source "${SCRATCH}/venv/bin/activate"

export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Aggregate throughput first: this is the strong-scaling numerator for step 3.
torchrun --standalone --nproc_per_node=4 -m cellfate.cli bench \
    --model th17_treg --scale 3.0 --walkers 1000000 --steps 200 \
    --json "${RESULTS}/bench_4gpu.json"

# Then the correctness claim that matters at this step: resampling is exact
# across the whole node, so the rate must match the single-GPU value and total
# weight must remain 1.0.
torchrun --standalone --nproc_per_node=4 -m cellfate.cli run \
    configs/toggle_validation.yaml --out "${RESULTS}/toggle_validation_4gpu.json"

torchrun --standalone --nproc_per_node=4 -m cellfate.cli run \
    configs/th17_treg.yaml \
    --override we.checkpoint_path="${SCRATCH}/checkpoints/th17_4gpu" \
    --out "${RESULTS}/th17_treg_4gpu.json"

echo "Step 2 complete.  Compare MFPT against ${SCRATCH}/results/step1_1gpu/toggle_validation.json"
