"""
Performance benchmarking tests for PHM-Vibench Model Factory

This module provides comprehensive performance testing including
speed, memory usage, and accuracy benchmarks.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import psutil
import os
from argparse import Namespace
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_factory import build_model


class TestModelSpeed:
    """Test model inference and training speed."""
    
    @pytest.fixture
    def benchmark_data(self):
        """Generate benchmark data for speed tests."""
        batch_sizes = [1, 8, 32, 64]
        seq_lengths = [64, 128, 256, 512]
        input_dim = 3
        
        data = {}
        for batch_size in batch_sizes:
            data[batch_size] = {}
            for seq_len in seq_lengths:
                data[batch_size][seq_len] = torch.randn(batch_size, seq_len, input_dim)
        
        return data
    
    @pytest.mark.parametrize("model_name", ['ResNetMLP', 'AttentionLSTM', 'ResNet1D', 'PatchTST'])
    def test_inference_speed(self, benchmark_data, model_name):
        """Benchmark inference speed for different models."""
        # Model configuration
        config = {
            'ResNetMLP': Namespace(
                model_name='ResNetMLP',
                input_dim=3,
                hidden_dim=128,
                num_layers=4,
                num_classes=4
            ),
            'AttentionLSTM': Namespace(
                model_name='AttentionLSTM',
                input_dim=3,
                hidden_dim=128,
                num_layers=2,
                bidirectional=True,
                num_classes=4
            ),
            'ResNet1D': Namespace(
                model_name='ResNet1D',
                input_dim=3,
                block_type='basic',
                layers=[2, 2, 2],
                num_classes=4
            ),
            'PatchTST': Namespace(
                model_name='PatchTST',
                input_dim=3,
                d_model=128,
                n_heads=8,
                e_layers=3,
                patch_len=16,
                stride=8,
                seq_len=256,
                pred_len=64
            )
        }
        
        args = config[model_name]
        model = build_model(args)
        model.eval()
        
        # Warm up
        x_warmup = torch.randn(8, 64, 3)
        with torch.no_grad():
            _ = model(x_warmup)
        
        # Benchmark different batch sizes and sequence lengths
        results = {}
        
        for batch_size in [8, 32]:
            for seq_len in [64, 128, 256]:
                if seq_len in benchmark_data[batch_size]:
                    x = benchmark_data[batch_size][seq_len]
                    
                    # Time inference
                    num_runs = 10
                    start_time = time.time()
                    
                    with torch.no_grad():
                        for _ in range(num_runs):
                            _ = model(x)
                    
                    end_time = time.time()
                    avg_time = (end_time - start_time) / num_runs
                    
                    results[f"batch_{batch_size}_seq_{seq_len}"] = avg_time
                    
                    # Assert reasonable performance (< 1 second per batch)
                    assert avg_time < 1.0, f"{model_name} too slow: {avg_time:.3f}s for batch_size={batch_size}, seq_len={seq_len}"
        
        print(f"\n{model_name} inference times:")
        for key, time_val in results.items():
            print(f"  {key}: {time_val:.3f}s")
    
    def test_training_speed(self):
        """Test training speed for a representative model."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=64,
            num_layers=3,
            num_classes=4
        )
        
        model = build_model(args)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Generate training data
        batch_size = 32
        seq_len = 64
        num_batches = 10
        
        data = []
        targets = []
        for _ in range(num_batches):
            x = torch.randn(batch_size, seq_len, 3)
            y = torch.randint(0, 4, (batch_size,))
            data.append(x)
            targets.append(y)
        
        # Time training
        model.train()
        start_time = time.time()
        
        for x, y in zip(data, targets):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        total_time = end_time - start_time
        time_per_batch = total_time / num_batches
        
        print(f"\nTraining speed: {time_per_batch:.3f}s per batch")
        
        # Assert reasonable training speed
        assert time_per_batch < 2.0, f"Training too slow: {time_per_batch:.3f}s per batch"


class TestModelMemory:
    """Test model memory usage."""
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @pytest.mark.parametrize("model_name", ['ResNetMLP', 'AttentionLSTM', 'ResNet1D'])
    def test_memory_usage(self, model_name):
        """Test memory usage for different models."""
        # Model configurations
        configs = {
            'ResNetMLP': Namespace(
                model_name='ResNetMLP',
                input_dim=3,
                hidden_dim=256,
                num_layers=6,
                num_classes=4
            ),
            'AttentionLSTM': Namespace(
                model_name='AttentionLSTM',
                input_dim=3,
                hidden_dim=256,
                num_layers=3,
                bidirectional=True,
                num_classes=4
            ),
            'ResNet1D': Namespace(
                model_name='ResNet1D',
                input_dim=3,
                block_type='basic',
                layers=[3, 4, 6, 3],
                num_classes=4
            )
        }
        
        args = configs[model_name]
        
        # Measure memory before model creation
        memory_before = self.get_memory_usage()
        
        # Create model
        model = build_model(args)
        
        # Measure memory after model creation
        memory_after = self.get_memory_usage()
        model_memory = memory_after - memory_before
        
        # Test inference memory
        x = torch.randn(32, 128, 3)
        memory_before_inference = self.get_memory_usage()
        
        with torch.no_grad():
            _ = model(x)
        
        memory_after_inference = self.get_memory_usage()
        inference_memory = memory_after_inference - memory_before_inference
        
        print(f"\n{model_name} memory usage:")
        print(f"  Model: {model_memory:.2f} MB")
        print(f"  Inference: {inference_memory:.2f} MB")
        
        # Assert reasonable memory usage
        assert model_memory < 500, f"{model_name} uses too much memory: {model_memory:.2f} MB"
        assert inference_memory < 1000, f"{model_name} inference uses too much memory: {inference_memory:.2f} MB"
    
    def test_memory_leak(self):
        """Test for memory leaks during repeated inference."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=64,
            num_layers=3,
            num_classes=4
        )
        
        model = build_model(args)
        model.eval()
        
        x = torch.randn(16, 64, 3)
        
        # Measure initial memory
        initial_memory = self.get_memory_usage()
        
        # Run many inferences
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        
        # Measure final memory
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory increase after 100 inferences: {memory_increase:.2f} MB")
        
        # Should not have significant memory increase (< 50 MB)
        assert memory_increase < 50, f"Potential memory leak: {memory_increase:.2f} MB increase"


class TestModelAccuracy:
    """Test model accuracy on synthetic datasets."""
    
    def generate_synthetic_classification_data(self, num_samples=1000, seq_len=64, 
                                             input_dim=3, num_classes=4):
        """Generate synthetic classification dataset."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        data = []
        labels = []
        
        for class_id in range(num_classes):
            for _ in range(num_samples // num_classes):
                t = np.linspace(0, 1, seq_len)
                
                # Class-specific frequency patterns
                freq = 10 + class_id * 10  # 10, 20, 30, 40 Hz
                signal = np.sin(2 * np.pi * freq * t)
                
                # Add noise and create multi-channel
                noise = 0.1 * np.random.randn(seq_len)
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
    
    @pytest.mark.parametrize("model_name", ['ResNetMLP', 'AttentionLSTM', 'ResNet1D'])
    def test_classification_accuracy(self, model_name):
        """Test classification accuracy on synthetic data."""
        # Generate synthetic data
        X, y = self.generate_synthetic_classification_data(
            num_samples=400, seq_len=64, input_dim=3, num_classes=4
        )
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Model configuration
        configs = {
            'ResNetMLP': Namespace(
                model_name='ResNetMLP',
                input_dim=3,
                hidden_dim=128,
                num_layers=4,
                num_classes=4
            ),
            'AttentionLSTM': Namespace(
                model_name='AttentionLSTM',
                input_dim=3,
                hidden_dim=128,
                num_layers=2,
                bidirectional=True,
                num_classes=4
            ),
            'ResNet1D': Namespace(
                model_name='ResNet1D',
                input_dim=3,
                block_type='basic',
                layers=[2, 2, 2],
                num_classes=4
            )
        }
        
        args = configs[model_name]
        model = build_model(args)
        
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        for epoch in range(20):  # Quick training
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
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
        print(f"\n{model_name} accuracy: {accuracy:.2f}%")
        
        # Should achieve better than random accuracy (25% for 4 classes)
        # With clear synthetic patterns, should achieve > 70%
        assert accuracy > 70, f"{model_name} accuracy too low: {accuracy:.2f}%"


class TestModelScaling:
    """Test model scaling characteristics."""
    
    @pytest.mark.parametrize("model_name", ['ResNetMLP', 'PatchTST', 'FNO'])
    def test_sequence_length_scaling(self, model_name):
        """Test how models scale with sequence length."""
        # Model configurations
        configs = {
            'ResNetMLP': Namespace(
                model_name='ResNetMLP',
                input_dim=3,
                hidden_dim=64,
                num_layers=3,
                num_classes=4
            ),
            'PatchTST': Namespace(
                model_name='PatchTST',
                input_dim=3,
                d_model=64,
                n_heads=4,
                e_layers=2,
                patch_len=16,
                stride=8,
                seq_len=512,
                pred_len=64
            ),
            'FNO': Namespace(
                model_name='FNO',
                input_dim=3,
                output_dim=3,
                modes=16,
                width=32,
                num_layers=2
            )
        }
        
        args = configs[model_name]
        model = build_model(args)
        model.eval()
        
        # Test different sequence lengths
        seq_lengths = [64, 128, 256, 512]
        batch_size = 8
        
        times = []
        
        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, 3)
            
            # Warm up
            with torch.no_grad():
                _ = model(x)
            
            # Time inference
            start_time = time.time()
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 5
            times.append(avg_time)
            
            print(f"{model_name} - seq_len {seq_len}: {avg_time:.3f}s")
        
        # Check scaling characteristics
        # Time should not grow too quickly with sequence length
        time_ratio = times[-1] / times[0]  # Ratio of longest to shortest
        
        # For most models, 8x sequence length should not take more than 16x time
        assert time_ratio < 16, f"{model_name} scales poorly: {time_ratio:.2f}x time increase for 8x sequence length"
    
    def test_batch_size_scaling(self):
        """Test how models scale with batch size."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=64,
            num_layers=3,
            num_classes=4
        )
        
        model = build_model(args)
        model.eval()
        
        # Test different batch sizes
        batch_sizes = [1, 4, 16, 64]
        seq_len = 64
        
        times = []
        throughputs = []
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, seq_len, 3)
            
            # Warm up
            with torch.no_grad():
                _ = model(x)
            
            # Time inference
            num_runs = 10
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(x)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_runs
            throughput = batch_size / avg_time
            
            times.append(avg_time)
            throughputs.append(throughput)
            
            print(f"Batch size {batch_size}: {avg_time:.3f}s, {throughput:.1f} samples/s")
        
        # Throughput should generally increase with batch size (up to a point)
        assert throughputs[-1] > throughputs[0], "Throughput should increase with batch size"
