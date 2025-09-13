"""
Pytest configuration and fixtures for PHM-Vibench Model Factory tests.

This module provides shared fixtures and configuration for all tests.
"""

import pytest
import torch
import numpy as np
import warnings
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("\n" + "="*50)
    print("PHM-Vibench Model Factory Test Suite")
    print("="*50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print("="*50)


@pytest.fixture
def device():
    """Get the best available device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data."""
    batch_size = 8
    seq_len = 64
    input_dim = 3
    num_classes = 4
    
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randint(0, num_classes, (batch_size,))
    
    return x, y


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    batch_size = 8
    seq_len = 64
    input_dim = 3
    output_dim = 3
    pred_len = 24
    
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randn(batch_size, pred_len, output_dim)
    
    return x, y


@pytest.fixture
def sample_multimodal_data():
    """Generate sample multi-modal data."""
    batch_size = 8
    seq_len = 64
    
    data = {
        'vibration': torch.randn(batch_size, seq_len, 3),
        'acoustic': torch.randn(batch_size, seq_len, 1),
        'thermal': torch.randn(batch_size, 2)
    }
    labels = torch.randint(0, 4, (batch_size,))
    
    return data, labels


@pytest.fixture
def basic_model_configs():
    """Basic model configurations for testing."""
    from argparse import Namespace
    
    return {
        'mlp': Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            num_classes=4
        ),
        'rnn': Namespace(
            model_name='AttentionLSTM',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            num_classes=4
        ),
        'cnn': Namespace(
            model_name='ResNet1D',
            input_dim=3,
            block_type='basic',
            layers=[1, 1],
            dropout=0.1,
            num_classes=4
        ),
        'transformer': Namespace(
            model_name='PatchTST',
            input_dim=3,
            d_model=32,
            n_heads=4,
            e_layers=2,
            patch_len=8,
            stride=4,
            seq_len=64,
            pred_len=24,
            dropout=0.1
        )
    }


@pytest.fixture
def performance_model_configs():
    """Model configurations for performance testing."""
    from argparse import Namespace
    
    return {
        'small': Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        ),
        'medium': Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=128,
            num_layers=4,
            num_classes=4
        ),
        'large': Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=256,
            num_layers=8,
            num_classes=4
        )
    }


@pytest.fixture
def synthetic_dataset():
    """Generate a synthetic dataset for training tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_samples = 200
    seq_len = 64
    input_dim = 3
    num_classes = 4
    
    data = []
    labels = []
    
    for class_id in range(num_classes):
        for _ in range(num_samples // num_classes):
            # Generate class-specific patterns
            t = np.linspace(0, 1, seq_len)
            freq = 10 + class_id * 5  # Different frequencies for each class
            
            signal = np.sin(2 * np.pi * freq * t)
            noise = 0.1 * np.random.randn(seq_len)
            
            # Create multi-channel data
            multi_channel = np.stack([
                signal + noise,
                signal + 0.1 * np.random.randn(seq_len),
                0.8 * signal + 0.05 * np.random.randn(seq_len)
            ], axis=1)
            
            data.append(multi_channel)
            labels.append(class_id)
    
    X = torch.FloatTensor(data)
    y = torch.LongTensor(labels)
    
    return X, y


@pytest.fixture
def temp_model_path(tmp_path):
    """Provide a temporary path for saving models."""
    return tmp_path / "test_model.pth"


class ModelTestHelper:
    """Helper class for model testing."""
    
    @staticmethod
    def count_parameters(model):
        """Count model parameters."""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def count_trainable_parameters(model):
        """Count trainable model parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size_mb(model):
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024 / 1024
    
    @staticmethod
    def test_forward_backward(model, x, y=None):
        """Test forward and backward pass."""
        # Forward pass
        output = model(x)
        
        if y is not None:
            # Test backward pass
            if len(y.shape) == 1:  # Classification
                criterion = torch.nn.CrossEntropyLoss()
            else:  # Regression
                criterion = torch.nn.MSELoss()
            
            loss = criterion(output, y)
            loss.backward()
            
            # Check gradients exist
            has_gradients = any(p.grad is not None for p in model.parameters())
            return output, loss, has_gradients
        
        return output, None, None
    
    @staticmethod
    def check_output_shape(output, expected_shape):
        """Check if output has expected shape."""
        return output.shape == expected_shape
    
    @staticmethod
    def check_no_nan_inf(tensor):
        """Check tensor has no NaN or Inf values."""
        return torch.isfinite(tensor).all()


@pytest.fixture
def model_test_helper():
    """Provide model test helper."""
    return ModelTestHelper()


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "cuda" in item.nodeid or "gpu" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.nodeid or "workflow" in item.nodeid:
            item.add_marker(pytest.mark.integration)


def pytest_runtest_setup(item):
    """Setup for each test."""
    # Skip GPU tests if CUDA is not available
    if "gpu" in [mark.name for mark in item.iter_markers()]:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")


def pytest_runtest_teardown(item):
    """Teardown for each test."""
    # Clear CUDA cache after each test if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
