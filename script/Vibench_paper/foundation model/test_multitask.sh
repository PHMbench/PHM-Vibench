#!/bin/bash
# Multi-Task Foundation Model - Quick Test Script
# Tests only B_04_Dlinear for rapid validation
# Author: PHM-Vibench Team
# Date: 2025-08-29

echo "=========================================="
echo "Multi-Task Foundation Model - Quick Test"
echo "Test Mode: B_04_Dlinear only, 1 epoch"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/home/lq/LQcode/2_project/PHMBench/PHM-Vibench"

# Test directory
EXPERIMENT_DIR="script/Vibench_paper/foundation model"
TEST_RESULTS_DIR="results/test_multitask"

# Create test results directory
mkdir -p $TEST_RESULTS_DIR

# Display system information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo ""
echo "Python Environment:"
python --version
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"

echo ""
echo "=========================================="
echo "Testing Multi-Task with B_04_Dlinear..."
echo "Config: $EXPERIMENT_DIR/multitask_B_04_Dlinear.yaml"
echo "=========================================="

# Run the test
start_time=$(date +%s)

python main.py --config_path "$EXPERIMENT_DIR/multitask_B_04_Dlinear.yaml" \
    --notes "Quick test of multi-task functionality" \
    2>&1 | tee "$TEST_RESULTS_DIR/B_04_Dlinear_test.log"

exit_code=$?
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ Test PASSED! Multi-task functionality working."
    echo "   Duration: ${duration}s"
    echo "   Ready for full experiments."
    
    # Show log summary
    echo ""
    echo "Log Summary:"
    echo "============"
    tail -n 10 "$TEST_RESULTS_DIR/B_04_Dlinear_test.log"
    
else
    echo "❌ Test FAILED! Exit code: $exit_code"
    echo "   Duration: ${duration}s"
    echo "   Please check configuration and dependencies."
    
    # Show error information
    echo ""
    echo "Error Summary:"
    echo "=============="
    tail -n 20 "$TEST_RESULTS_DIR/B_04_Dlinear_test.log"
    
    exit 1
fi

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "Results saved in: $TEST_RESULTS_DIR"
echo "Log file: $TEST_RESULTS_DIR/B_04_Dlinear_test.log"
echo "=========================================="

# Display next steps
echo ""
echo "Next Steps:"
echo "==========="
echo "1. Review test results in: $TEST_RESULTS_DIR"
echo "2. If successful, run full experiments with:"
echo "   bash script/Vibench_paper/foundation\\ model/run_multitask_experiments.sh"
echo "3. Monitor GPU memory usage for batch size optimization"
echo ""