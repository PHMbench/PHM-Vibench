#!/bin/bash
# Multi-Task PHM Foundation Model Experiments
# Runs all 4 backbone models with multi-task learning
# Author: PHM-Vibench Team
# Date: 2025-08-29

echo "=========================================="
echo "Multi-Task PHM Foundation Model Experiments"
echo "Testing 4 Models: B_04_Dlinear, B_06_TimesNet, B_08_PatchTST, B_09_FNO"
echo "Test Mode: 1 epoch each, wandb dryrun"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/home/lq/LQcode/2_project/PHMBench/PHM-Vibench"

# Experiment directory
EXPERIMENT_DIR="script/Vibench_paper/foundation_model"
RESULTS_DIR="results/multitask_experiments"

# Create results directory
mkdir -p $RESULTS_DIR

# Initialize log file
MAIN_LOG="$RESULTS_DIR/experiment_summary.log"
echo "Multi-Task Experiments Started: $(date)" > $MAIN_LOG
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
        --notes "Multi-task experiment with $model_name" \
        2>&1 | tee "$RESULTS_DIR/$model_name.log"
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Log results
    if [ $exit_code -eq 0 ]; then
        echo "✅ $model_name completed successfully (${duration}s)"
        echo "   Results: $RESULTS_DIR/$model_name/"
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
echo "Starting experiments..."
echo ""

# Experiment 1: B_04_Dlinear
run_experiment "B_04_Dlinear" "multitask_B_04_Dlinear.yaml"
results["B_04_Dlinear"]=$?

# Experiment 2: B_06_TimesNet
run_experiment "B_06_TimesNet" "multitask_B_06_TimesNet.yaml"
results["B_06_TimesNet"]=$?

# Experiment 3: B_08_PatchTST
run_experiment "B_08_PatchTST" "multitask_B_08_PatchTST.yaml"
results["B_08_PatchTST"]=$?

# Experiment 4: B_09_FNO
run_experiment "B_09_FNO" "multitask_B_09_FNO.yaml"
results["B_09_FNO"]=$?

# Calculate total time
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

# Generate summary report
echo "=========================================="
echo "EXPERIMENT SUMMARY"
echo "=========================================="
echo "Total Duration: ${total_duration}s ($(($total_duration / 60))min $(($total_duration % 60))s)"
echo ""
echo "Results:"
echo "--------"

successful=0
failed=0

for model in "B_04_Dlinear" "B_06_TimesNet" "B_08_PatchTST" "B_09_FNO"; do
    if [ ${results[$model]} -eq 0 ]; then
        echo "✅ $model - SUCCESS"
        ((successful++))
    else
        echo "❌ $model - FAILED"
        ((failed++))
    fi
done

echo ""
echo "Summary: $successful successful, $failed failed out of 4 experiments"

# Log final summary
echo "" >> $MAIN_LOG
echo "FINAL SUMMARY - Completed: $(date)" >> $MAIN_LOG
echo "Total Duration: ${total_duration}s" >> $MAIN_LOG
echo "Successful: $successful, Failed: $failed" >> $MAIN_LOG

if [ $failed -eq 0 ]; then
    echo ""
    echo "🎉 All experiments completed successfully!"
    echo ""
    echo "Next steps:"
    echo "- Review results in: $RESULTS_DIR/"
    echo "- Check individual logs for detailed metrics"
    echo "- For production runs, update configs (wandb: True, num_epochs: 50-150)"
    exit 0
else
    echo ""
    echo "⚠️  Some experiments failed. Check logs for details."
    echo ""
    echo "Debug steps:"
    echo "- Check failed experiment logs in: $RESULTS_DIR/"
    echo "- Verify GPU memory and dependencies"
    echo "- Run test_multitask.sh for quick diagnosis"
    exit 1
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in: $RESULTS_DIR"
echo "Summary log: $MAIN_LOG"
echo "=========================================="