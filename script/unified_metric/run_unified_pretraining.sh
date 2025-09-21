#!/bin/bash
# PHM-Vibench Unified Metric Learning - Stage 1: Pretraining Only
# Unified pretraining on all 5 industrial datasets simultaneously
# Expected Duration: ~12 hours
# Author: PHM-Vibench Team
# Date: 2025-09-16

echo "=========================================="
echo "🚀 PHM-Vibench Unified Metric Learning"
echo "Stage 1: Unified Pretraining"
echo "=========================================="
echo "📊 Training on: CWRU, XJTU, THU, Ottawa, JNU (simultaneously)"
echo "🎲 Seeds: 5 different random seeds"
echo "⏱️  Expected Duration: ~12 hours"
echo "🎯 Goal: Learn universal representations"
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
RESULTS_DIR="results/unified_metric_learning_pretraining"
TEMP_CONFIG="/tmp/unified_pretraining_config.yaml"

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

# Create results directory
mkdir -p $RESULTS_DIR
MAIN_LOG="$RESULTS_DIR/pretraining_$(date +%Y%m%d_%H%M%S).log"

# Initialize log file
echo "🚀 Unified Metric Learning Pretraining Stage - Started: $(date)" > $MAIN_LOG
echo "==========================================================" >> $MAIN_LOG
echo "Configuration: $CONFIG_FILE" >> $MAIN_LOG
echo "Results Directory: $RESULTS_DIR" >> $MAIN_LOG
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
# Create Pretraining-Only Configuration
# ===========================================

echo "⚙️  Creating pretraining-only configuration..."

# Use Python to create modified config
python3 -c "
import yaml
import sys

# Load base config
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)

# Modify for pretraining only
if 'execution' not in config:
    config['execution'] = {}
config['execution']['mode'] = 'pretraining'

# Update output directory
if 'environment' in config:
    config['environment']['output_dir'] = '$RESULTS_DIR'

# Disable fine-tuning stage
if 'training' in config and 'stage_2_finetuning' in config['training']:
    config['training']['stage_2_finetuning']['enabled'] = False

# Save modified config
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, indent=2)

print('✅ Pretraining configuration created')
"

if [ $? -ne 0 ]; then
    echo "❌ Failed to create pretraining configuration"
    exit 1
fi

# ===========================================
# Pre-flight Health Check
# ===========================================

echo "🔍 Running pre-flight health check..."
python script/unified_metric/pipeline/quick_validate.py --mode health_check --config $CONFIG_FILE 2>&1 | tee -a $MAIN_LOG

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Health check failed! Please fix issues before running pretraining."
    echo "💡 Tips:"
    echo "   - Check data directory path in config"
    echo "   - Ensure sufficient GPU memory (>8GB recommended)"
    echo "   - Verify all dependencies installed"
    rm -f $TEMP_CONFIG
    exit 1
fi

echo "✅ Health check passed!"
echo ""

# ===========================================
# Main Pretraining Execution
# ===========================================

start_time=$(date +%s)

echo "🚀 Starting Unified Pretraining Stage"
echo "====================================="
echo "🎯 Mode: Pretraining Only"
echo "📄 Config: $TEMP_CONFIG"
echo "📁 Output: $RESULTS_DIR"
echo "⏱️  Started: $(date)"
echo ""
echo "📊 Training Details:"
echo "==================="
echo "• Datasets: All 5 datasets simultaneously"
echo "• Model: ISFM with HSE prompt guidance"
echo "• Seeds: 5 different random seeds"
echo "• Expected experiments: 5 pretraining runs"
echo "• GPU Memory: Monitor with 'nvidia-smi' in another terminal"
echo ""

# Log pretraining start
echo "PRETRAINING EXECUTION START - $(date)" >> $MAIN_LOG
echo "Mode: Pretraining Only" >> $MAIN_LOG
echo "Config: $TEMP_CONFIG" >> $MAIN_LOG
echo "" >> $MAIN_LOG

# Run the pretraining stage using PHM-Vibench main.py
echo "🏃 Executing: python main.py --pipeline Pipeline_04_unified_metric --config $TEMP_CONFIG"
python main.py \
    --pipeline Pipeline_04_unified_metric \
    --config $TEMP_CONFIG \
    --notes "Unified pretraining stage - $(date)" \
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
echo "📊 PRETRAINING EXECUTION SUMMARY"
echo "=========================================="
echo "⏱️  Total Duration: ${total_duration}s ($(($total_duration / 3600))h $(($total_duration % 3600 / 60))m $(($total_duration % 60))s)"
echo "📅 Completed: $(date)"

# Log summary
echo "" >> $MAIN_LOG
echo "PRETRAINING EXECUTION SUMMARY - $(date)" >> $MAIN_LOG
echo "Total Duration: ${total_duration}s" >> $MAIN_LOG
echo "Exit Code: $exit_code" >> $MAIN_LOG

if [ $exit_code -eq 0 ]; then
    echo "✅ Status: SUCCESS"
    echo "" >> $MAIN_LOG
    echo "Status: SUCCESS" >> $MAIN_LOG

    echo ""
    echo "🎉 Pretraining stage completed successfully!"
    echo ""
    echo "📊 Expected Outputs:"
    echo "==================="
    echo "• 5 pretrained model checkpoints"
    echo "• Training logs and metrics"
    echo "• Universal representations learned"
    echo ""
    echo "📋 Next Steps:"
    echo "=============="
    echo "1. 🎯 Run zero-shot evaluation:"
    echo "   python main.py --pipeline Pipeline_04_unified_metric --config configs/unified_experiments.yaml"
    echo "   (Set execution.mode to 'zero_shot_eval' in config)"
    echo ""
    echo "2. 🚀 Run fine-tuning stage:"
    echo "   bash script/unified_metric/run_unified_finetuning.sh"
    echo ""
    echo "3. 📊 Or run complete remaining pipeline:"
    echo "   bash script/unified_metric/run_unified_complete.sh"
    echo ""
    echo "🔍 Checkpoint Locations:"
    echo "========================"
    if [ -d "$RESULTS_DIR" ]; then
        echo "📁 Results directory: $RESULTS_DIR"
        checkpoint_count=$(find $RESULTS_DIR -name "*.ckpt" 2>/dev/null | wc -l)
        echo "💾 Checkpoints found: $checkpoint_count"

        if [ $checkpoint_count -gt 0 ]; then
            echo "📝 Checkpoint paths:"
            find $RESULTS_DIR -name "*.ckpt" 2>/dev/null | head -5
        fi
    fi

else
    echo "❌ Status: FAILED (Exit code: $exit_code)"
    echo "" >> $MAIN_LOG
    echo "Status: FAILED (Exit code: $exit_code)" >> $MAIN_LOG

    echo ""
    echo "💥 Pretraining stage failed!"
    echo ""
    echo "🔍 Debugging Steps:"
    echo "=================="
    echo "1. 📝 Check main log: $MAIN_LOG"
    echo "2. 📁 Check experiment logs in: $RESULTS_DIR/logs/"
    echo "3. 🔍 Look for error messages in the output above"
    echo "4. 🧪 Run quick test: bash script/unified_metric/test_unified_1epoch.sh"
    echo "5. ✅ Run health check again: python script/unified_metric/pipeline/quick_validate.py --mode health_check"
    echo ""
    echo "🆘 Common Issues:"
    echo "================"
    echo "• GPU out of memory (reduce batch_size in config)"
    echo "• CUDA errors (check GPU drivers)"
    echo "• Data loading issues (verify data_dir path)"
    echo "• Dependency conflicts (check environment)"
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
echo "🏁 Pretraining Stage Completed"
echo "=========================================="
echo "📝 Main Log: $MAIN_LOG"
echo "📁 Results: $RESULTS_DIR"
echo "⏱️  Duration: $(($total_duration / 3600))h $(($total_duration % 3600 / 60))m $(($total_duration % 60))s"
echo "📅 Completed: $(date)"
echo "=========================================="

# Final log entry
echo "" >> $MAIN_LOG
echo "PRETRAINING STAGE COMPLETED - $(date)" >> $MAIN_LOG
echo "Final Status: $([ $exit_code -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')" >> $MAIN_LOG
echo "=========================================" >> $MAIN_LOG

exit $exit_code