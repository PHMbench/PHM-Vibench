#!/bin/bash
#SBATCH -J patchtst_l6
#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH -t 12:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -o logs/slurm-%j.out

module reset
module load miniconda
conda activate P

cd /home/lq/LQcode/2_project/PHMBench/PHM-Vibench-metric

mkdir -p logs

python main.py \
    --pipeline Pipeline_04_unified_metric \
    --config script/unified_metric/configs/unified_experiments_grace.yaml \
    --model.num_layers 6 \
    --notes "PatchTST num_layers=6 ablation - unified metric learning" \
2>&1 | tee "logs/${SLURM_JOB_ID}.log"
