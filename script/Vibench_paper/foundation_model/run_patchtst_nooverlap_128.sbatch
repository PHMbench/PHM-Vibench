#!/bin/bash
#SBATCH --job-name=patchtst_nooverlap_128
#SBATCH --partition=gpu
#SBATCH --gpus=v100:1              # V100 GPU for PatchTST with 128 windows
#SBATCH --cpus-per-gpu=12          # Increased CPUs for better throughput
#SBATCH --mem=48G                  # Increased memory for PatchTST with 128 windows
#SBATCH --time=36:00:00            # 36 hours for PatchTST with 128 windows
#SBATCH --output=logs/patchtst_nooverlap_128_%x_%j.out
#SBATCH --error=logs/patchtst_nooverlap_128_%x_%j.err
#SBATCH --chdir=/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/

echo "=========================================="
echo "🚀 Multi-Task PHM Foundation Model - B_08_PatchTST (No Overlap, 128 windows)"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo ""
echo "Configuration:"
echo "  - Window Size: 4096"
echo "  - Stride: 4096 (no overlap)"
echo "  - Strategy: Sequential sampling"
echo "  - Datasets: [1, 2, 5, 6, 13, 19]"
echo "  - Batch Size: 96"
echo "  - Num Windows: 128"
echo "=========================================="

# Load modules and environment
module reset
module load miniconda
conda activate P  # Use your conda environment

# Create logs directory if it doesn't exist
mkdir -p logs

# Display system information
echo ""
echo "System Information:"
echo "=================="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

echo "Running B_08_PatchTST No-Overlap (128 windows) experiment..."
echo "Config: multitask_B_08_PatchTST_nooverlap.yaml"
echo "Expected runtime: ~30 hours (aggressive 128 windows)"
echo ""

# Record start time
START_TIME=$(date +%s)

# Run the PatchTST experiment
python main_LQ.py \
    --config_path script/Vibench_paper/foundation_model/multitask_B_08_PatchTST_nooverlap.yaml \
    --notes "B_08_PatchTST No-Overlap Sequential 128 windows on Grace V100 - Job $SLURM_JOB_ID" \
    2>&1 | tee logs/patchtst_nooverlap_128_output_${SLURM_JOB_ID}.log

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
echo "Model: B_08_PatchTST (No Overlap, 128 windows)"
echo "Duration: ${HOURS}h ${MINUTES}m"
echo "Exit Code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: ✅ SUCCESS"
    echo ""
    echo "Results saved in: results/multitask_B_08_PatchTST_nooverlap/"
    echo "Log file: logs/patchtst_nooverlap_128_output_${SLURM_JOB_ID}.log"
    
    # Optional: Report key metrics if available
    if [ -f results/multitask_B_08_PatchTST_nooverlap/metrics.json ]; then
        echo ""
        echo "Key Metrics:"
        python -c "import json; data=json.load(open('results/multitask_B_08_PatchTST_nooverlap/metrics.json')); print(f\"  Accuracy: {data.get('test_acc', 'N/A')}\"); print(f\"  F1 Score: {data.get('test_f1', 'N/A')}\"); print(f\"  Loss: {data.get('test_loss', 'N/A')}\")" 2>/dev/null || echo "  Unable to parse metrics"
    fi
    
    echo ""
    echo "Configuration Used:"
    echo "  - stride=4096 (no overlap windows)"
    echo "  - window_sampling_strategy=sequential"
    echo "  - batch_size=96, num_window=128"
    echo "  - Aggressive setting: uses max available windows per dataset"
else
    echo "Status: ❌ FAILED"
    echo ""
    echo "Check error log: logs/patchtst_nooverlap_128_${SLURM_JOB_ID}.err"
    echo "Full output: logs/patchtst_nooverlap_128_output_${SLURM_JOB_ID}.log"
    echo ""
    echo "Troubleshooting Tips:"
    echo "  - Check dataset availability in metadata_6_11.xlsx"
    echo "  - Verify CUDA/PyTorch compatibility"
    echo "  - Check if num_window=128 causes OOM (reduce batch_size if needed)"
    echo "  - Dataset 2 has only 8 windows available (may cause issues)"
    echo "  - PatchTST may require more memory than DLinear"
fi

echo ""
echo "GPU Memory Usage at End:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo ""
echo "Completed at: $(date)"
echo "=========================================="