#!/bin/bash
#SBATCH -J unified_patchtst
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH -t 12:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o logs/slurm-%j.out

module reset
module load miniconda
conda activate YOUR_ENV

cd /home/lq/LQcode/2_project/PHMBench/PHM-Vibench-metric

python main.py \
    --pipeline Pipeline_04_unified_metric \
    --config script/unified_metric/configs/unified_experiments.yaml \
    --notes "PatchTST baseline - unified metric learning"