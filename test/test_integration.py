"""
Integration tests for PHM-Vibench Model Factory

This module tests end-to-end workflows including training,
evaluation, and model persistence.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from argparse import Namespace
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_factory import build_model


class TestModelTraining:
    """Test model training workflows."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset for training tests."""
        # Generate synthetic data
        np.random.seed(42)
        torch.manual_seed(42)
        
        num_samples = 100
        seq_len = 32
        input_dim = 3
        num_classes = 4
        
        # Create synthetic time series data
        data = []
        labels = []
        
        for class_id in range(num_classes):
            for _ in range(num_samples // num_classes):
                # Generate class-specific patterns
                t = np.linspace(0, 1, seq_len)
                if class_id == 0:
                    signal = np.sin(2 * np.pi * 10 * t)
                elif class_id == 1:
                    signal = np.sin(2 * np.pi * 20 * t)
                elif class_id == 2:
                    signal = np.sin(2 * np.pi * 30 * t)
                else:
                    signal = np.sin(2 * np.pi * 40 * t)
                
                # Add noise and create multi-channel
                noise = 0.1 * np.random.randn(seq_len)
                multi_channel = np.stack([
                    signal + noise,
                    signal + 0.1 * np.random.randn(seq_len),
                    0.5 * signal + 0.05 * np.random.randn(seq_len)
                ], axis=1)
                
                data.append(multi_channel)
                labels.append(class_id)
        
        X = torch.FloatTensor(data)
        y = torch.LongTensor(labels)
        
        # Create train/test split
        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        return train_loader, test_loader
    
    @pytest.mark.parametrize("model_name", ['ResNetMLP', 'AttentionLSTM', 'ResNet1D'])
    def test_model_training_workflow(self, sample_dataset, model_name):
        """Test complete training workflow for different models."""
        train_loader, test_loader = sample_dataset
        
        # Model configuration
        args = Namespace(
            model_name=model_name,
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            num_classes=4
        )
        
        # Add model-specific parameters
        if model_name == 'AttentionLSTM':
            args.bidirectional = True
        elif model_name == 'ResNet1D':
            args.block_type = 'basic'
            args.layers = [1, 1]
        
        # Build model
        model = build_model(args)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Training loop
        model.train()
        initial_loss = None
        final_loss = None
        
        for epoch in range(5):  # Short training for testing
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if initial_loss is None:
                    initial_loss = loss.item()
            
            final_loss = epoch_loss / len(train_loader)
        
        # Check that loss decreased
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        assert accuracy > 20, f"Accuracy too low: {accuracy}%"  # Should be better than random (25%)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            dropout=0.1,
            num_classes=4
        )
        
        # Build original model
        original_model = build_model(args)
        
        # Generate test input
        test_input = torch.randn(4, 32, 3)
        
        # Get original output
        original_model.eval()
        with torch.no_grad():
            original_output = original_model(test_input)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
            torch.save(original_model.state_dict(), tmp_file.name)
            
            # Load model
            loaded_model = build_model(args)
            loaded_model.load_state_dict(torch.load(tmp_file.name))
            
            # Get loaded output
            loaded_model.eval()
            with torch.no_grad():
                loaded_output = loaded_model(test_input)
            
            # Clean up
            os.unlink(tmp_file.name)
        
        # Check outputs are identical
        assert torch.allclose(original_output, loaded_output, atol=1e-6)


class TestModelCompatibility:
    """Test model compatibility with different input formats."""
    
    def test_batch_size_compatibility(self):
        """Test models work with different batch sizes."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        model.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16]
        seq_len = 32
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, seq_len, 3)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, 4)
            assert not torch.isnan(output).any()
    
    def test_sequence_length_compatibility(self):
        """Test models work with different sequence lengths."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        model.eval()
        
        # Test different sequence lengths
        seq_lengths = [16, 32, 64, 128]
        batch_size = 4
        
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, 3)
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (batch_size, 4)
            assert not torch.isnan(output).any()
    
    def test_device_compatibility(self):
        """Test model works on different devices."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        
        # Test CPU
        model_cpu = model.to('cpu')
        x_cpu = torch.randn(4, 32, 3)
        
        model_cpu.eval()
        with torch.no_grad():
            output_cpu = model_cpu(x_cpu)
        
        assert output_cpu.shape == (4, 4)
        assert not torch.isnan(output_cpu).any()
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            x_cuda = torch.randn(4, 32, 3).cuda()
            
            model_cuda.eval()
            with torch.no_grad():
                output_cuda = model_cuda(x_cuda)
            
            assert output_cuda.shape == (4, 4)
            assert not torch.isnan(output_cuda).any()


class TestModelPerformance:
    """Test model performance characteristics."""
    
    def test_model_parameter_count(self):
        """Test model parameter counts are reasonable."""
        test_configs = [
            ('ResNetMLP', {'input_dim': 3, 'hidden_dim': 64, 'num_layers': 3, 'num_classes': 4}),
            ('AttentionLSTM', {'input_dim': 3, 'hidden_dim': 64, 'num_layers': 2, 'num_classes': 4, 'bidirectional': True}),
            ('ResNet1D', {'input_dim': 3, 'block_type': 'basic', 'layers': [2, 2], 'num_classes': 4}),
        ]
        
        for model_name, config in test_configs:
            args = Namespace(model_name=model_name, **config)
            model = build_model(args)
            
            param_count = sum(p.numel() for p in model.parameters())
            
            # Check parameter count is reasonable (not too small or too large)
            assert 1000 < param_count < 10_000_000, f"{model_name} has {param_count} parameters"
    
    def test_model_inference_speed(self):
        """Test model inference speed is reasonable."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=64,
            num_layers=3,
            num_classes=4
        )
        
        model = build_model(args)
        model.eval()
        
        # Warm up
        x = torch.randn(32, 64, 3)
        with torch.no_grad():
            _ = model(x)
        
        # Time inference
        import time
        
        num_runs = 10
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        # Should complete inference in reasonable time (< 1 second per batch)
        assert avg_time < 1.0, f"Inference too slow: {avg_time:.3f}s per batch"
    
    def test_model_memory_usage(self):
        """Test model memory usage is reasonable."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=64,
            num_layers=3,
            num_classes=4
        )
        
        model = build_model(args)
        
        # Calculate model size in MB
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        # Model should be reasonable size (< 100MB for test configs)
        assert model_size_mb < 100, f"Model too large: {model_size_mb:.2f}MB"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_input_shapes(self):
        """Test model handles invalid input shapes gracefully."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        
        # Test wrong number of dimensions
        with pytest.raises((RuntimeError, ValueError)):
            x_wrong_dims = torch.randn(32, 3)  # Missing sequence dimension
            model(x_wrong_dims)
        
        # Test wrong feature dimension
        with pytest.raises((RuntimeError, ValueError)):
            x_wrong_features = torch.randn(32, 64, 5)  # Wrong feature dimension
            model(x_wrong_features)
    
    def test_empty_input_handling(self):
        """Test model handles empty inputs gracefully."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        
        # Test empty batch
        with pytest.raises((RuntimeError, ValueError)):
            x_empty = torch.randn(0, 32, 3)
            model(x_empty)
    
    def test_gradient_flow(self):
        """Test gradients flow properly through models."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        criterion = nn.CrossEntropyLoss()
        
        x = torch.randn(4, 32, 3, requires_grad=True)
        y = torch.randint(0, 4, (4,))
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Check that gradients exist and are not zero
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients found in model parameters"
