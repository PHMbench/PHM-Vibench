#!/bin/bash
#SBATCH -J prompt_no_prompt
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
    --model.use_prompt false \
    --task.prompt_weight 0.0 \
    --notes "Ablation: disable prompt fusion" \
2>&1 | tee "logs/${SLURM_JOB_ID}.log"
