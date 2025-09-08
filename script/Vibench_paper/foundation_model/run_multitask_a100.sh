
#!/bin/bash
#SBATCH --job-name=multitask_a100
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/multitask_a100_%j.out
#SBATCH --error=results/multitask_a100_%j.err

# Multi-Task Foundation Model A100 Experiments
# Tests all backbone models with memory optimizations
# Author: PHM-Vibench Team
# Date: 2025-09-07

# Set environment variables
# export CUDA_VISIBLE_DEVICES=0
# export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}"

# Create results directory
EXPERIMENT_DIR="script/Vibench_paper/foundation_model"
RESULTS_DIR="results/multitask_a100_${SLURM_JOB_ID}"
mkdir -p $RESULTS_DIR/logs

# Initialize log file
MAIN_LOG="$RESULTS_DIR/experiment_summary.log"
echo "Multi-Task A100 Experiments Started: $(date)" > $MAIN_LOG
echo "SLURM Job ID: $SLURM_JOB_ID" >> $MAIN_LOG
echo "Node: $SLURMD_NODENAME" >> $MAIN_LOG
echo "========================================" >> $MAIN_LOG

# Display system information
echo "System Information:"
echo "=================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Function to run single experiment
run_experiment() {
    local model_name=$1
    local config_file=$2
    local start_time=$(date +%s)
    
    echo "=========================================="
    echo "Running Multi-Task with $model_name..."
    echo "Config: $config_file"
    echo "Started at: $(date)"
    echo "=========================================="
    
    # Log experiment start
    echo "Experiment: $model_name - Started: $(date)" >> $MAIN_LOG
    
    # Run the experiment
    python main_LQ.py --config_path "$EXPERIMENT_DIR/$config_file" \
        --notes "Multi-task A100 experiment with $model_name - Job $SLURM_JOB_ID" \
        2>&1 | tee "$RESULTS_DIR/$model_name.log"
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Log results
    if [ $exit_code -eq 0 ]; then
        echo "✅ $model_name completed successfully (${duration}s)"
        echo "   Log: $RESULTS_DIR/$model_name.log"
        echo "Experiment: $model_name - Completed: $(date) - Duration: ${duration}s - Status: SUCCESS" >> $MAIN_LOG
        return 0
    else
        echo "❌ $model_name failed with exit code $exit_code (${duration}s)"
        echo "   Check log: $RESULTS_DIR/$model_name.log"
        echo "Experiment: $model_name - Failed: $(date) - Duration: ${duration}s - Status: FAILED (exit $exit_code)" >> $MAIN_LOG
        return 1
    fi
    echo ""
}

# Track experiment results
declare -A results
total_start_time=$(date +%s)

# Run experiments sequentially
echo "Starting A100 experiments..."
echo ""

# Experiment 1: B_04_Dlinear (快速测试)
run_experiment "B_04_Dlinear" "multitask_B_04_Dlinear.yaml"
results["B_04_Dlinear"]=$?

# 根据第一个实验结果决定是否继续
if [ ${results["B_04_Dlinear"]} -eq 0 ]; then
    echo "First experiment successful, continuing with remaining models..."
    
    # Experiment 2: B_06_TimesNet
    run_experiment "B_06_TimesNet" "multitask_B_06_TimesNet.yaml"
    results["B_06_TimesNet"]=$?
    
    # Experiment 3: B_08_PatchTST
    run_experiment "B_08_PatchTST" "multitask_B_08_PatchTST.yaml"
    results["B_08_PatchTST"]=$?
    
    # Experiment 4: B_09_FNO
    run_experiment "B_09_FNO" "multitask_B_09_FNO.yaml"
    results["B_09_FNO"]=$?
else
    echo "First experiment failed, skipping remaining experiments"
    results["B_06_TimesNet"]=99  # Mark as skipped
    results["B_08_PatchTST"]=99
    results["B_09_FNO"]=99
fi

# Calculate total time
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

# Generate summary report
echo "=========================================="
echo "EXPERIMENT SUMMARY"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Total Duration: ${total_duration}s ($(($total_duration / 60))min $(($total_duration % 60))s)"
echo ""
echo "Results:"
echo "--------"

successful=0
failed=0
skipped=0

for model in "B_04_Dlinear" "B_06_TimesNet" "B_08_PatchTST" "B_09_FNO"; do
    if [ ${results[$model]} -eq 0 ]; then
        echo "✅ $model - SUCCESS"
        ((successful++))
    elif [ ${results[$model]} -eq 99 ]; then
        echo "⏭️  $model - SKIPPED"
        ((skipped++))
    else
        echo "❌ $model - FAILED"
        ((failed++))
    fi
done

echo ""
echo "Summary: $successful successful, $failed failed, $skipped skipped out of 4 experiments"

# Log final summary
echo "" >> $MAIN_LOG
echo "FINAL SUMMARY - Completed: $(date)" >> $MAIN_LOG
echo "Total Duration: ${total_duration}s" >> $MAIN_LOG
echo "Successful: $successful, Failed: $failed, Skipped: $skipped" >> $MAIN_LOG

# Final status and cleanup
if [ $failed -eq 0 ] && [ $successful -gt 0 ]; then
    echo ""
    echo "🎉 Experiments completed successfully on A100!"
    echo ""
    echo "Results saved in: $RESULTS_DIR"
    echo "Summary log: $MAIN_LOG"
    echo ""
    echo "GPU utilization summary:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
    exit 0
else
    echo ""
    echo "⚠️  Some experiments failed. Check logs for details."
    echo ""
    echo "Debug info:"
    echo "- Results directory: $RESULTS_DIR"
    echo "- Summary log: $MAIN_LOG"
    echo "- Individual logs: $RESULTS_DIR/*.log"
    exit 1
fi