#!/bin/bash
# Local test version of the A100 multi-task script
# Use this for testing on interactive nodes or local machines

echo "=========================================="
echo "Multi-Task PHM Foundation Model - A100 (Local Test)"
echo "Started: $(date)"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/home/lq/LQcode/2_project/PHMBench/PHM-Vibench"

# Experiment directory
EXPERIMENT_DIR="script/Vibench_paper/foundation_model"
RESULTS_DIR="results/multitask_a100_local_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

# Initialize log file
MAIN_LOG="$RESULTS_DIR/experiment_summary.log"
echo "Multi-Task A100 Local Test Started: $(date)" > $MAIN_LOG
echo "========================================" >> $MAIN_LOG

# Display system information
echo "System Information:"
echo "=================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 2>/dev/null || echo 'GPU info not available')"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not available')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'CUDA info not available')"
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
    
    # Check if config file exists
    if [ ! -f "$EXPERIMENT_DIR/$config_file" ]; then
        echo "❌ Config file not found: $EXPERIMENT_DIR/$config_file"
        echo "Config file missing: $config_file" >> $MAIN_LOG
        return 1
    fi
    
    # Log experiment start
    echo "Experiment: $model_name - Started: $(date)" >> $MAIN_LOG
    
    # Run the experiment
    python main_LQ.py --config_path "$EXPERIMENT_DIR/$config_file" \
        --notes "Multi-task A100 local test with $model_name" \
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

echo "Starting local A100 test experiments..."
echo ""

# Only test B_04_Dlinear for quick validation
echo "Running single model test (B_04_Dlinear)..."
run_experiment "B_04_Dlinear" "multitask_B_04_Dlinear.yaml"
results["B_04_Dlinear"]=$?

# Calculate total time
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

# Generate summary report
echo "=========================================="
echo "LOCAL TEST SUMMARY"
echo "=========================================="
echo "Total Duration: ${total_duration}s ($(($total_duration / 60))min $(($total_duration % 60))s)"
echo ""

if [ ${results["B_04_Dlinear"]} -eq 0 ]; then
    echo "✅ B_04_Dlinear - SUCCESS"
    echo ""
    echo "🎉 Local test passed! Ready for SLURM submission."
    echo ""
    echo "To submit to A100 cluster, run:"
    echo "sbatch script/Vibench_paper/foundation_model/run_multitask_a100.sh"
    echo ""
    echo "Results saved in: $RESULTS_DIR"
    echo "Summary log: $MAIN_LOG"
    exit 0
else
    echo "❌ B_04_Dlinear - FAILED"
    echo ""
    echo "⚠️  Local test failed. Please fix issues before SLURM submission."
    echo ""
    echo "Debug info:"
    echo "- Results directory: $RESULTS_DIR"
    echo "- Summary log: $MAIN_LOG"
    echo "- Test log: $RESULTS_DIR/B_04_Dlinear.log"
    exit 1
fi