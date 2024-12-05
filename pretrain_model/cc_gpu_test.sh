#!/bin/bash
#SBATCH --account=def-mbfeng
#SBATCH --nodes=2
#SBATCH --mem=4gb
#SBATCH --tasks-per-node=3
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:3
#SBATCH --time=0:30:0
#SBATCH --output=torch_test_2nodes.out

export NCCL_BLOCKING_WAIT=1
export MASTER_ADDR=$(hostname)

srun python -u test_cc.py