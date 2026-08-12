#!/bin/bash

# Environment Setup Script for LLM-Enhanced Fault Diagnosis Experiments

set -e  # Exit on any error

echo "🚀 Setting up environment for LLM-Enhanced Fault Diagnosis Experiments"
echo "================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    print_status "Python version $PYTHON_VERSION is compatible (required: >= $REQUIRED_VERSION)"
else
    print_error "Python version $PYTHON_VERSION is not compatible (required: >= $REQUIRED_VERSION)"
    exit 1
fi

# Check if we're in the correct directory
print_status "Checking current directory..."
if [[ ! -f "../../README.md" ]] && [[ ! -f "../../../README.md" ]]; then
    print_warning "Not in the expected directory structure"
    print_status "Current directory: $(pwd)"
    print_status "Expected to be in Paper/LLM_Explainable_FD_Toolkit/experiments/scripts/"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p ../results/baseline_results
mkdir -p ../results/llm_results
mkdir -p ../results/comparison_analysis
mkdir -p ../results/figures
mkdir -p ../logs
mkdir -p ../cache
mkdir -p ../data/processed
mkdir -p ../data/synthetic

print_status "Directory structure created successfully."

# Check for CUDA availability
print_status "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | cut -d':' -f2 | xargs)
        print_status "CUDA detected: $CUDA_VERSION"

        # Check GPU memory
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        print_status "GPU Memory: ${GPU_MEMORY} MB"
    else
        print_warning "nvidia-smi available but no GPU detected"
    fi
else
    print_warning "nvidia-smi not found. CUDA may not be available."
fi

# Check PyTorch installation
print_status "Checking PyTorch installation..."
if python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    print_status "PyTorch is properly installed"
else
    print_error "PyTorch is not installed or not accessible"
    exit 1
fi

# Check environment variables
print_status "Checking environment variables..."

# LLM API Keys (optional for basic setup)
if [[ -n "$OPENAI_API_KEY" ]]; then
    print_status "OpenAI API key is set"
else
    print_warning "OpenAI API key is not set (OPENAI_API_KEY)"
    print_warning "LLM functionality will use fallback mode"
fi

if [[ -n "$ANTHROPIC_API_KEY" ]]; then
    print_status "Anthropic API key is set"
else
    print_warning "Anthropic API key is not set (ANTHROPIC_API_KEY)"
fi

# Data directory check
print_status "Checking data directories..."
DATA_BASE_DIR="/home/user/data/PHMbenchdata/PHM-Vibench"

if [[ -d "$DATA_BASE_DIR" ]]; then
    print_status "Data directory found: $DATA_BASE_DIR"

    # Check for specific datasets
    if [[ -d "$DATA_BASE_DIR/THU_006" ]]; then
        print_status "THU_006 dataset available"
    else
        print_warning "THU_006 dataset not found"
    fi

    if [[ -d "$DATA_BASE_DIR/THU_018" ]]; then
        print_status "THU_018 dataset available"
    else
        print_warning "THU_018 dataset not found"
    fi
else
    print_warning "Data directory not found: $DATA_BASE_DIR"
    print_status "Will use synthetic data for testing"
fi

# Install dependencies if needed
print_status "Checking Python dependencies..."
if [[ -f "../../code/requirements.txt" ]]; then
    cd ../../code
    if python3 -m pip install -r requirements.txt --quiet; then
        print_status "Dependencies installed successfully"
    else
        print_warning "Failed to install some dependencies"
        print_warning "Please manually check: pip install -r code/requirements.txt"
    fi
    cd ../experiments/scripts
else
    print_warning "requirements.txt not found"
fi

# Check disk space
print_status "Checking available disk space..."
DISK_SPACE=$(df -h . | tail -1 | awk '{print $4}')
print_status "Available disk space: $DISK_SPACE"

if [[ ${DISK_SPACE%G} -lt 5 ]]; then
    print_warning "Low disk space (< 5GB). Consider cleaning up."
fi

# Create log directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="../logs/env_setup_$TIMESTAMP"
mkdir -p "$LOG_DIR"

print_status "Log directory created: $LOG_DIR"

# System information
print_status "System Information:"
echo "  - OS: $(uname -s) $(uname -r)"
echo "  - CPU: $(nproc) cores"
echo "  - Memory: $(free -h | awk '/^Mem:/ {print $7}') available"
echo "  - Python: $PYTHON_VERSION"
echo "  - Current time: $(date)"

# Test basic imports
print_status "Testing basic Python imports..."
cd ../../code

python3 -c "
import sys
print('Python path:', sys.path[0])

try:
    import torch
    print('✓ PyTorch:', torch.__version__)
except ImportError as e:
    print('✗ PyTorch import failed:', e)

try:
    import numpy
    print('✓ NumPy:', numpy.__version__)
except ImportError as e:
    print('✗ NumPy import failed:', e)

try:
    import matplotlib
    print('✓ Matplotlib:', matplotlib.__version__)
except ImportError as e:
    print('✗ Matplotlib import failed:', e)

try:
    import scipy
    print('✓ SciPy:', scipy.__version__)
except ImportError as e:
    print('✗ SciPy import failed:', e)

print('Basic imports test completed.')
"

cd ../experiments/scripts

# Create environment configuration file
print_status "Creating environment configuration..."
cat > ../config/.env_setup << EOF
# Environment Configuration
# Generated by environment setup script

export EXPERIMENT_START_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
export EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"
export PYTHONPATH=$(pwd)/../../code:$PYTHONPATH

# Data paths
export DATA_BASE_DIR="/home/user/data/PHMbenchdata/PHM-Vibench"
export RESULTS_BASE_DIR="$(pwd)/../results"
export LOG_BASE_DIR="$(pwd)/../logs"

# Model paths
export MODEL_CACHE_DIR="$(pwd)/../cache/models"
export EXPLANATION_CACHE_DIR="$(pwd)/../cache/explanations"

# GPU settings
export CUDA_VISIBLE_DEVICES=0

# LLM settings (if available)
# export OPENAI_API_KEY="your-api-key-here"
# export ANTHROPIC_API_KEY="your-api-key-here"

# Performance settings
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=4

EOF

print_status "Environment configuration saved to ../config/.env_setup"

# Create test script
print_status "Creating environment test script..."
cat > ../scripts/test_environment.py << 'EOF'
#!/usr/bin/env python3
"""
Environment Test Script
"""

import sys
import os
import torch
import numpy as np

print("Environment Test")
print("=" * 30)

# Test basic functionality
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# Test PyTorch
try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
except Exception as e:
    print(f"PyTorch test failed: {e}")

# Test basic array operations
try:
    x = torch.randn(1, 1024, 1)
    print(f"Test tensor shape: {x.shape}")
    print(f"Test tensor mean: {x.mean().item():.4f}")
except Exception as e:
    print(f"Tensor operation test failed: {e}")

print("Environment test completed successfully!")
EOF

chmod +x ../scripts/test_environment.py

print_status "Environment test script created"

# Final status check
print_status "Environment setup completed!"
echo "================================================================="

echo ""
print_status "Next steps:"
echo "1. Run environment test: python3 ../scripts/test_environment.py"
echo "2. Execute experiments: python3 ../scripts/02_data_preparation.py"
echo "3. Check configuration: cat ../configs/base_config.yaml"

echo ""
print_warning "Important notes:"
echo "- Ensure you have sufficient GPU memory for model training"
echo "- Configure LLM API keys for full functionality"
echo "- Monitor disk space during experiments"
echo "- Check log files for any errors"

echo ""
print_status "Setup log saved to: $LOG_DIR"
print_status "Environment completed at: $(date)"