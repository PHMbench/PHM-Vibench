#!/bin/bash
#SBATCH --job-name=multitask_test
#SBATCH --partition=gpu
#SBATCH --gpus=v100:1              # V100 is sufficient for quick test
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G                  # Reduced memory for test
#SBATCH --time=01:00:00            # 1 hour for quick test
#SBATCH --output=logs/test_%x_%j.out
#SBATCH --error=logs/test_%x_%j.err
#SBATCH --chdir=/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/

echo "=========================================="
echo "Multi-Task PHM Foundation Model - Quick Test"
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

echo "Running quick 1-epoch test..."
echo "Config: multitask_test_1epoch.yaml"
echo ""

# Run the test experiment
python main_LQ.py \
    --config_path script/Vibench_paper/foundation_model/multitask_test_1epoch.yaml \
    --notes "Quick 1-epoch test on Grace - Job $SLURM_JOB_ID" \
    2>&1 | tee logs/test_output_${SLURM_JOB_ID}.log

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test completed successfully!"
    echo "Check logs/test_output_${SLURM_JOB_ID}.log for details"
    echo ""
    echo "If successful, you can proceed with full experiments:"
    echo "  sbatch run_dlinear.sbatch"
    echo "  sbatch run_fno.sbatch"
    echo "  sbatch run_timesnet.sbatch"
    echo "  sbatch run_patchtst.sbatch"
else
    echo ""
    echo "❌ Test failed! Check logs for errors."
    echo "Debug log: logs/test_output_${SLURM_JOB_ID}.log"
fi

echo ""
echo "Test completed at: $(date)"
echo "=========================================="