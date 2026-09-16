#!/bin/bash

#SBATCH --job-name        ddp16
#SBATCH --nodes           2
#SBATCH --ntasks-per-node 4           # one task per GPU
#SBATCH --gpus            8
#SBATCH --time            05:00:00
#SBATCH --output          slog/%j.out

source /lus/lfs1aip2/scratch/u6pl/mat611.u6pl/torch_dis/torch_env/bin/activate
module load brics/nccl

#export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
#export MASTER_PORT=29500
#export NCCL_DEBUG=INFO
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

# Run the function with srun
srun -N -${SLURM_NNODES} \
    --gpus=${SLURM_GPUS} \
    --mpi=pmi2 \
    --ntasks-per-node=${SLURM_NTASKS_PER_NODE} \
    bash -c 'export WORLD_SIZE=$SLURM_GPUS; export RANK=$PMI_RANK; export LOCAL_RANK=$SLURM_LOCALID; python3 3-train.py'
