#!/bin/bash
#SBATCH --job-name=multitask_fno
#SBATCH --partition=gpu
#SBATCH --gpus=a5000:1             # Single A5000 GPU optimized for FNO
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00            # 24 hours for full training
#SBATCH --output=logs/fno_%x_%j.out
#SBATCH --error=logs/fno_%x_%j.err
#SBATCH --chdir=/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/

echo "=========================================="
echo "Multi-Task PHM Foundation Model - B_09_FNO"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=========================================="

# Load modules and environment
module reset
module load miniconda
conda activate P  # Use your conda environment

# Create logs directory if it doesn't exist
mkdir -p logs

# Display system information
echo "System Information:"
echo "=================="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

echo "Running B_09_FNO experiment..."
echo "Config: multitask_B_09_FNO.yaml"
echo "Expected runtime: ~17 hours for 50 epochs"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run the FNO experiment
python main_LQ.py \
    --config_path script/Vibench_paper/foundation_model/multitask_B_09_FNO.yaml \
    --notes "B_09_FNO on Grace A5000 - Job $SLURM_JOB_ID" \
    2>&1 | tee logs/fno_output_${SLURM_JOB_ID}.log

# Check exit status
EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo "Experiment Summary"
echo "=========================================="
echo "Model: B_09_FNO"
echo "Duration: ${HOURS}h ${MINUTES}m"
echo "Exit Code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: ✅ SUCCESS"
    echo ""
    echo "Results saved in: results/multitask_B_09_FNO/"
    echo "Log file: logs/fno_output_${SLURM_JOB_ID}.log"
    
    # Optional: Report key metrics if available
    if [ -f results/multitask_B_09_FNO/metrics.json ]; then
        echo ""
        echo "Key Metrics:"
        python -c "import json; data=json.load(open('results/multitask_B_09_FNO/metrics.json')); print(f\"  Accuracy: {data.get('test_acc', 'N/A')}\"); print(f\"  F1 Score: {data.get('test_f1', 'N/A')}\")" 2>/dev/null || echo "  Unable to parse metrics"
    fi
else
    echo "Status: ❌ FAILED"
    echo ""
    echo "Check error log: logs/fno_${SLURM_JOB_ID}.err"
    echo "Full output: logs/fno_output_${SLURM_JOB_ID}.log"
fi

echo ""
echo "GPU Memory Usage at End:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo ""
echo "Completed at: $(date)"
echo "=========================================="