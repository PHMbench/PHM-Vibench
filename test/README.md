# PHM-Vibench Model Factory Testing Suite

This directory contains a comprehensive testing suite for the PHM-Vibench Model Factory, ensuring all implemented models work correctly across different scenarios and configurations.

## 📋 Test Structure

```
test/
├── README.md                    # This file
├── conftest.py                  # Pytest configuration and fixtures
├── test_model_factory.py        # Core model tests
├── test_integration.py          # Integration and workflow tests
├── test_performance.py          # Performance benchmarks
└── test_utils.py               # Utility function tests
```

## 🧪 Test Categories

### 1. **Unit Tests** (`test_model_factory.py`)
Tests individual model components and functionality:

- **Model Creation**: Verify all models can be instantiated
- **Forward Pass**: Test forward propagation with various input shapes
- **Output Validation**: Ensure outputs have correct shapes and no NaN/Inf values
- **Parameter Counting**: Validate reasonable parameter counts

**Model Categories Tested**:
- MLP Models (5 models): ResNetMLP, MLPMixer, gMLP, DenseNetMLP, Dlinear
- Neural Operators (5 models): FNO, DeepONet, NeuralODE, GraphNO, WaveletNO
- Transformer Models (5 models): Informer, Autoformer, PatchTST, Linformer, ConvTransformer
- RNN Models (5 models): AttentionLSTM, ConvLSTM, ResidualRNN, AttentionGRU, TransformerRNN
- CNN Models (5 models): ResNet1D, TCN, AttentionCNN, MobileNet1D, MultiScaleCNN
- ISFM Models (5 models): ContrastiveSSL, MaskedAutoencoder, MultiModalFM, SignalLanguageFM, TemporalDynamicsSSL

### 2. **Integration Tests** (`test_integration.py`)
Tests end-to-end workflows and model interactions:

- **Training Workflows**: Complete training loops with synthetic data
- **Model Persistence**: Save/load functionality
- **Device Compatibility**: CPU/GPU transfer
- **Batch Size Scaling**: Different batch sizes
- **Sequence Length Compatibility**: Variable input lengths
- **Error Handling**: Invalid inputs and edge cases

### 3. **Performance Tests** (`test_performance.py`)
Benchmarks model performance characteristics:

- **Inference Speed**: Time per batch for different models
- **Training Speed**: Training time benchmarks
- **Memory Usage**: RAM and GPU memory consumption
- **Scaling Analysis**: Performance vs. sequence length/batch size
- **Accuracy Validation**: Performance on synthetic datasets

### 4. **Utility Tests** (`test_utils.py`)
Tests helper functions and utilities:

- **Parameter Utilities**: Counting and analysis functions
- **Data Validation**: Input shape and type checking
- **Configuration Validation**: Parameter validation
- **Model Compatibility**: TorchScript, ONNX, mixed precision
- **Error Recovery**: Gradient clipping, checkpointing

## 🚀 Running Tests

### Quick Start

```bash
# Install testing dependencies
pip install -r requirements-test.txt

# Run smoke tests (quick validation)
python run_tests.py --smoke

# Run all unit tests
python run_tests.py --unit

# Run integration tests
python run_tests.py --integration
```

### Test Runner Options

The `run_tests.py` script provides multiple testing modes:

```bash
# Quick validation
python run_tests.py --smoke

# Core functionality tests
python run_tests.py --unit

# End-to-end workflow tests
python run_tests.py --integration

# Performance benchmarks
python run_tests.py --performance

# GPU-specific tests (requires CUDA)
python run_tests.py --gpu

# Coverage analysis
python run_tests.py --coverage

# Test specific model category
python run_tests.py --model mlp
python run_tests.py --model transformer
python run_tests.py --model isfm

# Complete test suite
python run_tests.py --all
```

### Direct Pytest Usage

```bash
# Run all tests
pytest test/

# Run specific test file
pytest test/test_model_factory.py

# Run specific test class
pytest test/test_model_factory.py::TestMLPModels

# Run specific test method
pytest test/test_model_factory.py::TestMLPModels::test_mlp_model_creation

# Run with coverage
pytest test/ --cov=src --cov-report=html

# Run performance tests only
pytest test/ -m performance

# Skip slow tests
pytest test/ -m "not slow"

# Parallel execution
pytest test/ -n auto
```

## 📊 Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.slow`: Long-running tests (performance benchmarks)
- `@pytest.mark.gpu`: Tests requiring CUDA/GPU
- `@pytest.mark.integration`: Integration and workflow tests
- `@pytest.mark.performance`: Performance benchmarking tests
- `@pytest.mark.unit`: Unit tests (default)
- `@pytest.mark.smoke`: Quick validation tests

## 🔧 Configuration

### Pytest Configuration (`pytest.ini`)
- Test discovery patterns
- Output formatting
- Coverage settings
- Marker definitions
- Warning filters

### Test Fixtures (`conftest.py`)
- **Device Selection**: Automatic CPU/GPU detection
- **Sample Data**: Synthetic datasets for testing
- **Model Configurations**: Pre-defined model setups
- **Test Helpers**: Utility functions for testing

## 📈 Performance Benchmarks

The performance tests provide comprehensive benchmarks:

### Speed Benchmarks
- Inference time per batch
- Training time per epoch
- Scaling with sequence length
- Scaling with batch size

### Memory Benchmarks
- Model size in MB
- Peak memory usage during inference
- Memory scaling characteristics
- Memory leak detection

### Accuracy Benchmarks
- Classification accuracy on synthetic data
- Convergence speed during training
- Robustness to different input patterns

## 🎯 Test Coverage

The test suite aims for comprehensive coverage:

- **Model Coverage**: All 30+ implemented models
- **Functionality Coverage**: Creation, forward pass, training
- **Input Coverage**: Various shapes, types, edge cases
- **Error Coverage**: Invalid inputs, edge cases
- **Platform Coverage**: CPU/GPU, different OS

### Coverage Reports

```bash
# Generate HTML coverage report
pytest test/ --cov=src --cov-report=html

# View coverage in terminal
pytest test/ --cov=src --cov-report=term-missing

# Coverage with minimum threshold
pytest test/ --cov=src --cov-fail-under=80
```

## 🔍 Debugging Tests

### Verbose Output
```bash
# Detailed test output
pytest test/ -v -s

# Show local variables on failure
pytest test/ --tb=long

# Stop on first failure
pytest test/ -x

# Run last failed tests
pytest test/ --lf
```

### Performance Debugging
```bash
# Profile test execution time
pytest test/ --durations=10

# Memory profiling
pytest test/ --profile

# Benchmark specific tests
pytest test/test_performance.py -s
```

## 🚨 Continuous Integration

The test suite integrates with GitHub Actions for automated testing:

- **Multi-platform**: Ubuntu, Windows, macOS
- **Multi-version**: Python 3.8, 3.9, 3.10, 3.11
- **Parallel execution**: Multiple test jobs
- **Coverage reporting**: Codecov integration
- **Performance monitoring**: Benchmark tracking

### CI Configuration (`.github/workflows/test.yml`)
- Automated testing on push/PR
- Nightly performance benchmarks
- Security scanning
- Code quality checks

## 📝 Writing New Tests

### Test Template

```python
class TestNewModel:
    """Test suite for NewModel."""
    
    @pytest.fixture
    def model_config(self):
        """Model configuration for testing."""
        return Namespace(
            model_name='NewModel',
            input_dim=3,
            hidden_dim=64,
            num_classes=4
        )
    
    def test_model_creation(self, model_config):
        """Test model can be created."""
        from src.model_factory.Category.NewModel import Model
        model = Model(model_config)
        assert model is not None
    
    def test_forward_pass(self, model_config):
        """Test forward pass."""
        from src.model_factory.Category.NewModel import Model
        model = Model(model_config)
        
        x = torch.randn(4, 32, 3)
        output = model(x)
        
        assert output.shape == (4, 4)
        assert not torch.isnan(output).any()
```

### Best Practices

1. **Use Fixtures**: Reuse common configurations and data
2. **Parametrize Tests**: Test multiple configurations efficiently
3. **Clear Assertions**: Specific error messages
4. **Isolated Tests**: Each test should be independent
5. **Performance Awareness**: Mark slow tests appropriately

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **CUDA Errors**: Skip GPU tests if CUDA unavailable
3. **Memory Issues**: Reduce batch sizes for testing
4. **Timeout Issues**: Increase timeout for slow tests

### Environment Setup

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytest; print(f'Pytest: {pytest.__version__}')"
```

## 📞 Support

For testing-related issues:

1. Check this documentation
2. Review test output and error messages
3. Run tests with verbose output (`-v -s`)
4. Check GitHub Issues for known problems
5. Contact the development team

## 🎉 Contributing

When adding new models or features:

1. **Add Tests**: Include comprehensive tests for new functionality
2. **Update Documentation**: Update this README if needed
3. **Run Full Suite**: Ensure all tests pass before submitting
4. **Performance Check**: Add performance tests for new models
5. **CI Validation**: Verify tests pass in CI environment

The testing suite ensures the PHM-Vibench Model Factory maintains high quality and reliability across all supported models and use cases.
