#!/bin/bash
# PHM-Vibench Unified Metric Learning - Stage 2: Fine-tuning Only
# Dataset-specific fine-tuning using pretrained universal models
# Expected Duration: ~10 hours (5 datasets × 5 seeds × ~24 minutes each)
# Author: PHM-Vibench Team
# Date: 2025-09-16

echo "=========================================="
echo "🚀 PHM-Vibench Unified Metric Learning"
echo "Stage 2: Dataset-Specific Fine-tuning"
echo "=========================================="
echo "📊 Datasets: CWRU, XJTU, THU, Ottawa, JNU (individually)"
echo "🎲 Seeds: 5 seeds per dataset"
echo "⏱️  Expected Duration: ~10 hours"
echo "🎯 Goal: High accuracy with pretrained models"
echo "💾 Requires: Completed pretraining checkpoints"
echo "=========================================="

# ===========================================
# Configuration and Environment Setup
# ===========================================

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration paths
SCRIPT_DIR="script/unified_metric"
CONFIG_FILE="$SCRIPT_DIR/configs/unified_experiments.yaml"
RESULTS_DIR="results/unified_metric_learning_finetuning"
PRETRAINING_DIR="results/unified_metric_learning_pretraining"
TEMP_CONFIG="/tmp/unified_finetuning_config.yaml"

# Check if running from project root
if [ ! -f "main.py" ]; then
    echo "❌ Error: Must run from PHM-Vibench project root directory"
    echo "💡 Tip: cd to PHM-Vibench root and run: bash $0"
    exit 1
fi

# Check configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Check for pretraining checkpoints
if [ ! -d "$PRETRAINING_DIR" ]; then
    echo "❌ Error: Pretraining results not found: $PRETRAINING_DIR"
    echo "💡 Please run pretraining first:"
    echo "   bash script/unified_metric/run_unified_pretraining.sh"
    exit 1
fi

# Count available checkpoints
checkpoint_count=$(find $PRETRAINING_DIR -name "*.ckpt" 2>/dev/null | wc -l)
if [ $checkpoint_count -eq 0 ]; then
    echo "❌ Error: No pretraining checkpoints found in $PRETRAINING_DIR"
    echo "💡 Please run pretraining first:"
    echo "   bash script/unified_metric/run_unified_pretraining.sh"
    exit 1
fi

echo "✅ Found $checkpoint_count pretraining checkpoint(s)"

# Create results directory
mkdir -p $RESULTS_DIR
MAIN_LOG="$RESULTS_DIR/finetuning_$(date +%Y%m%d_%H%M%S).log"

# Initialize log file
echo "🚀 Unified Metric Learning Fine-tuning Stage - Started: $(date)" > $MAIN_LOG
echo "==========================================================" >> $MAIN_LOG
echo "Configuration: $CONFIG_FILE" >> $MAIN_LOG
echo "Pretraining Results: $PRETRAINING_DIR" >> $MAIN_LOG
echo "Results Directory: $RESULTS_DIR" >> $MAIN_LOG
echo "Available Checkpoints: $checkpoint_count" >> $MAIN_LOG
echo "" >> $MAIN_LOG

# ===========================================
# System Information
# ===========================================

echo "📋 System Information:"
echo "======================"
echo "🖥️  Hostname: $(hostname)"
echo "🐍 Python: $(python --version 2>&1)"
echo "🔥 PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    echo "💾 GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
    echo "🔋 GPU Utilization: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)%"
else
    echo "🎮 GPU: Not available (will use CPU - much slower)"
fi
echo "📅 Start Time: $(date)"
echo ""

# Log system information
echo "System Information - $(date)" >> $MAIN_LOG
echo "Hostname: $(hostname)" >> $MAIN_LOG
echo "Python: $(python --version 2>&1)" >> $MAIN_LOG
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)" >> $MAIN_LOG
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB" >> $MAIN_LOG
fi
echo "" >> $MAIN_LOG

# ===========================================
# Create Fine-tuning-Only Configuration
# ===========================================

echo "⚙️  Creating fine-tuning-only configuration..."

# Use Python to create modified config
python3 -c "
import yaml
import sys

# Load base config
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Modify for fine-tuning only
if 'execution' not in config:
    config['execution'] = {}
config['execution']['mode'] = 'finetuning'

# Update output directory
if 'environment' in config:
    config['environment']['output_dir'] = '$RESULTS_DIR'

# Disable pretraining stage
if 'training' in config and 'stage_1_pretraining' in config['training']:
    config['training']['stage_1_pretraining']['enabled'] = False

# Set pretraining checkpoint directory
if 'training' in config and 'stage_2_finetuning' in config['training']:
    config['training']['stage_2_finetuning']['pretraining_dir'] = '$PRETRAINING_DIR'

# Save modified config
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

print('✅ Fine-tuning configuration created')
"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create fine-tuning configuration"
    exit 1
fi

# ===========================================
# Pretraining Checkpoint Validation
# ===========================================

echo "🔍 Validating pretraining checkpoints..."

echo "📁 Available checkpoints:"
find $PRETRAINING_DIR -name "*.ckpt" 2>/dev/null | head -5 | while read ckpt; do
    echo "   💾 $(basename $ckpt)"
done

if [ $checkpoint_count -gt 5 ]; then
    echo "   ... and $(($checkpoint_count - 5)) more"
fi

echo ""

# ===========================================
# Main Fine-tuning Execution
# ===========================================

start_time=$(date +%s)

echo "🚀 Starting Dataset-Specific Fine-tuning Stage"
echo "=============================================="
echo "🎯 Mode: Fine-tuning Only"
echo "📄 Config: $TEMP_CONFIG"
echo "📁 Output: $RESULTS_DIR"
echo "💾 Pretrained Models: $PRETRAINING_DIR"
echo "⏱️  Started: $(date)"
echo ""
echo "📊 Fine-tuning Details:"
echo "======================"
echo "• Source: Pretrained universal models"
echo "• Target: 5 datasets (CWRU, XJTU, THU, Ottawa, JNU)"
echo "• Seeds: 5 random seeds per dataset"
echo "• Expected experiments: 25 fine-tuning runs (5×5)"
echo "• Strategy: Transfer learning with checkpoint loading"
echo "• Expected improvement: >10% over zero-shot"
echo ""

# Log fine-tuning start
echo "FINE-TUNING EXECUTION START - $(date)" >> $MAIN_LOG
echo "Mode: Fine-tuning Only" >> $MAIN_LOG
echo "Config: $TEMP_CONFIG" >> $MAIN_LOG
echo "Pretraining Checkpoints: $checkpoint_count" >> $MAIN_LOG
echo "" >> $MAIN_LOG

# Run the fine-tuning stage using PHM-Vibench main.py
echo "🏃 Executing: python main.py --pipeline Pipeline_04_unified_metric --config $TEMP_CONFIG"
python main.py \
    --pipeline Pipeline_04_unified_metric \
    --config $TEMP_CONFIG \
    --notes "Dataset-specific fine-tuning stage - $(date)" \
    2>&1 | tee -a $MAIN_LOG

# Capture exit code
exit_code=$?
end_time=$(date +%s)
total_duration=$((end_time - start_time))

# Cleanup temporary config
rm -f $TEMP_CONFIG

# ===========================================
# Results Analysis and Summary
# ===========================================

echo ""
echo "=========================================="
echo "📊 FINE-TUNING EXECUTION SUMMARY"
echo "=========================================="
echo "⏱️  Total Duration: ${total_duration}s ($(($total_duration / 3600))h $(($total_duration % 3600 / 60))m $(($total_duration % 60))s)"
echo "📅 Completed: $(date)"

# Log summary
echo "" >> $MAIN_LOG
echo "FINE-TUNING EXECUTION SUMMARY - $(date)" >> $MAIN_LOG
echo "Total Duration: ${total_duration}s" >> $MAIN_LOG
echo "Exit Code: $exit_code" >> $MAIN_LOG

if [ $exit_code -eq 0 ]; then
    echo "✅ Status: SUCCESS"
    echo "" >> $MAIN_LOG
    echo "Status: SUCCESS" >> $MAIN_LOG

    echo ""
    echo "🎉 Fine-tuning stage completed successfully!"
    echo ""
    echo "📊 Expected Outputs:"
    echo "==================="
    echo "• 25 fine-tuned model checkpoints (5 datasets × 5 seeds)"
    echo "• Training logs and metrics for each dataset"
    echo "• Performance improvements over zero-shot baseline"
    echo ""
    echo "📈 Expected Results:"
    echo "==================="
    echo "• Fine-tuned accuracy: >95% (target: 94.7%)"
    echo "• Improvement over zero-shot: >10%"
    echo "• Statistical significance: p < 0.001"
    echo ""
    echo "📋 Next Steps:"
    echo "=============="
    echo "1. 📊 Analyze results:"
    echo "   python script/unified_metric/analysis/collect_results.py --mode analyze"
    echo ""
    echo "2. 🎨 Generate figures:"
    echo "   python script/unified_metric/analysis/paper_visualization.py --demo"
    echo ""
    echo "3. 📄 Create comparison tables:"
    echo "   python script/unified_metric/pipeline/sota_comparison.py --methods all"
    echo ""
    echo "4. 📋 Generate publication report:"
    echo "   python script/unified_metric/analysis/collect_results.py --mode publication"
    echo ""
    echo "🔍 Results Summary:"
    echo "=================="
    if [ -d "$RESULTS_DIR" ]; then
        echo "📁 Results directory: $RESULTS_DIR"
        metrics_files=$(find $RESULTS_DIR -name "metrics.json" 2>/dev/null | wc -l)
        echo "📈 Completed experiments: $metrics_files/25 expected"

        if [ $metrics_files -gt 0 ]; then
            echo "💾 Results found for fine-tuning experiments"

            # Show directory structure
            echo "📁 Results structure:"
            if command -v tree &> /dev/null; then
                tree $RESULTS_DIR -L 3 2>/dev/null | head -15
            else
                ls -la $RESULTS_DIR/ 2>/dev/null | head -10
            fi
        fi
    fi

else
    echo "❌ Status: FAILED (Exit code: $exit_code)"
    echo "" >> $MAIN_LOG
    echo "Status: FAILED (Exit code: $exit_code)" >> $MAIN_LOG

    echo ""
    echo "💥 Fine-tuning stage failed!"
    echo ""
    echo "🔍 Debugging Steps:"
    echo "=================="
    echo "1. 📝 Check main log: $MAIN_LOG"
    echo "2. 📁 Check experiment logs in: $RESULTS_DIR/logs/"
    echo "3. 🔍 Look for error messages in the output above"
    echo "4. 💾 Verify pretraining checkpoints:"
    echo "   ls -la $PRETRAINING_DIR/"
    echo "5. 🧪 Run quick test: bash script/unified_metric/test_unified_1epoch.sh"
    echo ""
    echo "🆘 Common Issues:"
    echo "================"
    echo "• Checkpoint loading errors (check pretraining paths)"
    echo "• GPU out of memory (reduce batch_size in config)"
    echo "• Missing pretraining results (run pretraining first)"
    echo "• Model architecture mismatch (check model config)"
fi

# ===========================================
# Performance Analysis
# ===========================================

if [ $exit_code -eq 0 ] && [ -d "$RESULTS_DIR" ]; then
    echo ""
    echo "🔍 Quick Performance Analysis:"
    echo "============================="

    # Count successful experiments by dataset
    for dataset in CWRU XJTU THU Ottawa JNU; do
        dataset_results=$(find $RESULTS_DIR -path "*${dataset}*" -name "metrics.json" 2>/dev/null | wc -l)
        echo "📊 $dataset: $dataset_results/5 experiments completed"
    done
fi

# ===========================================
# Resource Usage Summary
# ===========================================

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🔋 Final GPU Status:"
    echo "==================="
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -1 | while read mem_used mem_total gpu_util; do
        echo "💾 GPU Memory: ${mem_used}MB / ${mem_total}MB used"
        echo "🔋 GPU Utilization: ${gpu_util}%"
    done
fi

echo ""
echo "=========================================="
echo "🏁 Fine-tuning Stage Completed"
echo "=========================================="
echo "📝 Main Log: $MAIN_LOG"
echo "📁 Results: $RESULTS_DIR"
echo "💾 Pretrained Models: $PRETRAINING_DIR"
echo "⏱️  Duration: $(($total_duration / 3600))h $(($total_duration % 3600 / 60))m $(($total_duration % 60))s"
echo "📅 Completed: $(date)"
echo "=========================================="

# Final log entry
echo "" >> $MAIN_LOG
echo "FINE-TUNING STAGE COMPLETED - $(date)" >> $MAIN_LOG
echo "Final Status: $([ $exit_code -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" >> $MAIN_LOG
echo "=========================================" >> $MAIN_LOG

exit $exit_code