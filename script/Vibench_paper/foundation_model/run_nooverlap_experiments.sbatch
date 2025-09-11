#!/bin/bash
#SBATCH --job-name=foundation_nooverlap
#SBATCH --partition=gpu
#SBATCH --gpus=v100:1              # V100 GPU sufficient for no-overlap training
#SBATCH --cpus-per-gpu=12          # Increased CPUs for better throughput
#SBATCH --mem=32G                  # Reduced from 48G (no-overlap needs less memory)
#SBATCH --time=72:00:00            # 3 days for all 4 models (conservative estimate)
#SBATCH --output=logs/nooverlap_%x_%j.out
#SBATCH --error=logs/nooverlap_%x_%j.err
#SBATCH --chdir=/vast/palmer/home.grace/ql334/LQ/PHM-Vibench/

echo "================================================================================"
echo "🚀 Multi-Task PHM Foundation Models - No Overlap Windows Training"
echo "================================================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "Training Configuration:"
echo "  - Window Size: 4096"
echo "  - Stride: 4096 (no overlap)"
echo "  - Strategy: Sequential sampling"
echo "  - Datasets: [1, 2, 5, 6, 13, 19]"
echo "  - Models: DLinear, TimesNet, PatchTST, FNO"
echo "================================================================================"

# Load modules and environment
module reset
module load miniconda
conda activate P  # Use your conda environment

# Create logs directory if it doesn't exist
mkdir -p logs/nooverlap

# Display system information
echo ""
echo "System Information:"
echo "=================="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# Define models and their expected runtimes
declare -A models
models[DLinear]="multitask_B_04_Dlinear_nooverlap.yaml"
models[TimesNet]="multitask_B_06_TimesNet_nooverlap.yaml"
models[PatchTST]="multitask_B_08_PatchTST_nooverlap.yaml"
models[FNO]="multitask_B_09_FNO_nooverlap.yaml"

declare -A expected_time
expected_time[DLinear]="6 hours"
expected_time[TimesNet]="18 hours"
expected_time[PatchTST]="18 hours"
expected_time[FNO]="24 hours"

# Total runtime tracking
TOTAL_START_TIME=$(date +%s)
RESULTS_SUMMARY=""

# Run each model sequentially
for model in DLinear TimesNet PatchTST FNO; do
    config_file=${models[$model]}
    expected=${expected_time[$model]}
    
    echo ""
    echo "🔄 Starting $model Training"
    echo "============================================="
    echo "Config: $config_file"
    echo "Expected runtime: ~$expected"
    echo "Started: $(date)"
    echo ""
    
    # Record start time for this model
    MODEL_START_TIME=$(date +%s)
    
    # Run the experiment
    python main_LQ.py \
        --config_path script/Vibench_paper/foundation_model/$config_file \
        --notes "$model No-Overlap Sequential on Grace V100 - Job $SLURM_JOB_ID" \
        2>&1 | tee logs/nooverlap/${model,,}_output_${SLURM_JOB_ID}.log
    
    # Check exit status
    EXIT_CODE=$?
    MODEL_END_TIME=$(date +%s)
    MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
    MODEL_HOURS=$((MODEL_DURATION / 3600))
    MODEL_MINUTES=$(((MODEL_DURATION % 3600) / 60))
    
    echo ""
    echo "============================================="
    echo "$model Training Summary"
    echo "============================================="
    echo "Duration: ${MODEL_HOURS}h ${MODEL_MINUTES}m"
    echo "Exit Code: $EXIT_CODE"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Status: ✅ SUCCESS"
        RESULTS_SUMMARY="$RESULTS_SUMMARY\n✅ $model: ${MODEL_HOURS}h ${MODEL_MINUTES}m - SUCCESS"
        
        # Check for results
        result_dir="results/multitask_B_*${model}*_nooverlap"
        if [ -d $result_dir ]; then
            echo "Results saved in: $result_dir"
        fi
        
        # Optional: Report key metrics if available
        metrics_file=$(find results/ -name "metrics.json" -path "*${model}*nooverlap*" | head -1)
        if [ -f "$metrics_file" ]; then
            echo ""
            echo "Key Metrics:"
            python -c "
import json
try:
    with open('$metrics_file') as f:
        data = json.load(f)
    print(f'  Accuracy: {data.get(\"test_acc\", \"N/A\")}')
    print(f'  F1 Score: {data.get(\"test_f1\", \"N/A\")}')
    print(f'  Loss: {data.get(\"test_loss\", \"N/A\")}')
except:
    print('  Unable to parse metrics')
" 2>/dev/null || echo "  Unable to parse metrics"
        fi
    else
        echo "Status: ❌ FAILED"
        RESULTS_SUMMARY="$RESULTS_SUMMARY\n❌ $model: ${MODEL_HOURS}h ${MODEL_MINUTES}m - FAILED (Code: $EXIT_CODE)"
        echo ""
        echo "Check error log: logs/nooverlap/${model,,}_${SLURM_JOB_ID}.err"
        echo "Full output: logs/nooverlap/${model,,}_output_${SLURM_JOB_ID}.log"
    fi
    
    # Display current GPU memory usage
    echo ""
    echo "GPU Memory Usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    
    echo ""
    echo "Completed: $(date)"
    echo ""
    
    # Brief pause between models
    sleep 10
done

# Final summary
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "================================================================================"
echo "🎯 FINAL EXPERIMENT SUMMARY"
echo "================================================================================"
echo "Total Runtime: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m"
echo "Completed: $(date)"
echo ""
echo "Model Results:"
echo -e "$RESULTS_SUMMARY"
echo ""
echo "Configuration Summary:"
echo "  - Window Strategy: Sequential, No Overlap (stride=4096)"
echo "  - Memory Usage: Optimized for V100-16GB"
echo "  - Datasets Processed: 6 (1, 2, 5, 6, 13, 19)"
echo "  - Total Samples: 9,589"
echo ""

# Generate final report
cat > logs/nooverlap/experiment_summary_${SLURM_JOB_ID}.txt << EOF
PHM-Vibench Foundation Models - No Overlap Training Summary
==========================================================

Job Information:
- SLURM Job ID: $SLURM_JOB_ID
- Node: $SLURMD_NODENAME
- Total Runtime: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m
- Completed: $(date)

Configuration:
- Window Size: 4096
- Stride: 4096 (no overlap)
- Sampling Strategy: Sequential
- Datasets: [1, 2, 5, 6, 13, 19]
- GPU: V100-16GB
- Memory: 32GB

Results:
$(echo -e "$RESULTS_SUMMARY")

Log Files:
$(ls -la logs/nooverlap/*_${SLURM_JOB_ID}.*)
EOF

echo "📊 Experiment summary saved: logs/nooverlap/experiment_summary_${SLURM_JOB_ID}.txt"
echo ""
echo "================================================================================"
echo "🏁 All experiments completed!"
echo "================================================================================"