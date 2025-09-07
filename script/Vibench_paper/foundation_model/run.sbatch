#!/bin/bash
#SBATCH --job-name=multitask_test
#SBATCH --partition=gpu
#SBATCH --gpus=v100:4  
## a100
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/multitask_%x_%j.out
#SBATCH --error=logs/multitask_%x_%j.err
#SBATCH --chdir=/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/
## #SBATCH --account=pi_xxxx         # 如需指定计费账户，取消注释

echo "=========================================="
echo "Multi-Task PHM Foundation Model - A100"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=========================================="

# Load modules and environment
module reset
module load miniconda
conda activate P  # 使用您的环境名称

./script/Vibench_paper/foundation_model/run_multitask_a100.sh