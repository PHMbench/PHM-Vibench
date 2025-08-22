"""
Comprehensive Testing Framework for PHM-Vibench Research Code

This module provides a complete testing infrastructure for validating research code
quality, mathematical correctness, and scientific reproducibility. It includes:

- Unit tests for individual components
- Property-based tests for mathematical invariants
- Integration tests for complete workflows
- Performance benchmarks
- Reproducibility validation tests

The framework ensures that research code meets publication standards and
can be reliably used for scientific experiments.
"""

from __future__ import annotations

import math
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
import pytorch_lightning as pl
from pathlib import Path


class BaseTestCase(ABC):
    """
    Abstract base class for all test cases in PHM-Vibench.

    Provides common functionality for testing research components including:
    - Reproducible test setup
    - Common assertion methods
    - Performance benchmarking utilities
    - Mathematical property validation
    """

    def setup_method(self, method):
        """Setup method called before each test."""
        # Set deterministic seeds for reproducible tests
        torch.manual_seed(42)
        np.random.seed(42)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def assert_tensor_shape(self, tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: str = "tensor"):
        """Assert that tensor has expected shape."""
        actual_shape = tuple(tensor.shape)
        assert actual_shape == expected_shape, (
            f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"
        )

    def assert_tensor_finite(self, tensor: torch.Tensor, name: str = "tensor"):
        """Assert that tensor contains only finite values."""
        assert torch.isfinite(tensor).all(), f"{name} contains non-finite values (NaN or Inf)"

    def assert_tensor_range(self, tensor: torch.Tensor, min_val: float, max_val: float, name: str = "tensor"):
        """Assert that tensor values are within expected range."""
        assert tensor.min() >= min_val, f"{name} minimum value {tensor.min()} < {min_val}"
        assert tensor.max() <= max_val, f"{name} maximum value {tensor.max()} > {max_val}"

    def assert_approximately_equal(self, a: float, b: float, tolerance: float = 1e-6, name: str = "values"):
        """Assert that two values are approximately equal."""
        assert abs(a - b) <= tolerance, f"{name} not approximately equal: {a} vs {b} (tolerance: {tolerance})"

    def benchmark_function(self, func: Callable, *args, num_runs: int = 10, **kwargs) -> Dict[str, float]:
        """Benchmark a function and return timing statistics."""
        times = []

        for _ in range(num_runs):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'total_time': np.sum(times)
        }


class ModelTestCase(BaseTestCase):
    """
    Test case class for neural network models.

    Provides specialized testing methods for validating model implementations
    including forward pass correctness, gradient flow, and mathematical properties.
    """

    def assert_model_output_shape(self, model: nn.Module, input_tensor: torch.Tensor, expected_output_shape: Tuple[int, ...]):
        """Test that model produces expected output shape."""
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        self.assert_tensor_shape(output, expected_output_shape, "model output")
        self.assert_tensor_finite(output, "model output")

    def assert_gradient_flow(self, model: nn.Module, input_tensor: torch.Tensor, loss_fn: Optional[Callable] = None):
        """Test that gradients flow properly through the model."""
        model.train()
        input_tensor.requires_grad_(True)

        # Forward pass
        output = model(input_tensor)

        # Compute loss (use sum if no loss function provided)
        if loss_fn is None:
            loss = output.sum()
        else:
            # Create dummy target for loss computation
            if output.dim() == 2:  # Classification
                target = torch.randint(0, output.size(1), (output.size(0),))
            else:  # Regression
                target = torch.randn_like(output)
            loss = loss_fn(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist and are finite
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"
                self.assert_tensor_finite(param.grad, f"gradient for {name}")

    def assert_model_determinism(self, model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 3):
        """Test that model produces deterministic outputs."""
        model.eval()
        outputs = []

        for _ in range(num_runs):
            torch.manual_seed(42)  # Reset seed for each run
            with torch.no_grad():
                output = model(input_tensor.clone())
            outputs.append(output)

        # Check that all outputs are identical
        for i in range(1, num_runs):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6), (
                f"Model outputs are not deterministic: run 0 vs run {i}"
            )

    def assert_model_parameter_count(self, model: nn.Module, expected_range: Tuple[int, int]):
        """Test that model has reasonable number of parameters."""
        param_count = sum(p.numel() for p in model.parameters())
        min_params, max_params = expected_range

        assert min_params <= param_count <= max_params, (
            f"Model parameter count {param_count} not in expected range [{min_params}, {max_params}]"
        )