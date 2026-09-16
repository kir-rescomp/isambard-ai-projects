#!/bin/bash

#SBATCH --job-name        ddp16
#SBATCH --nodes           1
#SBATCH --ntasks-per-node 1           # one task per GPU
#SBATCH --gpus-per-node   4
#SBATCH --cpus-per-task   288            # GH200: 72 Grace cores/node ÷ 4
#SBATCH --time            01:30:00
#SBATCH --output          slog/%j.out

source ~/project/software/virtual_env/torch_py312/bin/activate

#export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
#export MASTER_PORT=29500
#export NCCL_DEBUG=INFO
#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

srun bash -c 'torchrun \
  --nnodes=4 \
  --nproc_per_node=4 \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  train.py'
