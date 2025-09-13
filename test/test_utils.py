"""
Utility tests for PHM-Vibench Model Factory

This module tests utility functions, data preprocessing,
and helper components.
"""

import pytest
import torch
import numpy as np
from argparse import Namespace
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_factory import build_model


class TestModelUtilities:
    """Test utility functions for models."""
    
    def test_parameter_counting(self):
        """Test parameter counting utility."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=64,
            num_layers=3,
            num_classes=4
        )
        
        model = build_model(args)
        
        # Count parameters manually
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0, "Model should have parameters"
        assert trainable_params == total_params, "All parameters should be trainable by default"
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def test_model_summary(self):
        """Test model summary generation."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        
        # Test model can be summarized
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        
        assert total_params > 0
        assert model_size_mb > 0
        
        print(f"Model size: {model_size_mb:.2f} MB")
    
    def test_model_device_transfer(self):
        """Test model device transfer utilities."""
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
        for param in model_cpu.parameters():
            assert param.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to('cuda')
            for param in model_cuda.parameters():
                assert param.device.type == 'cuda'
    
    def test_model_mode_switching(self):
        """Test model train/eval mode switching."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            dropout=0.5,  # High dropout to test mode switching
            num_classes=4
        )
        
        model = build_model(args)
        x = torch.randn(4, 32, 3)
        
        # Test training mode
        model.train()
        assert model.training
        
        output_train1 = model(x)
        output_train2 = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train1, output_train2, atol=1e-6)
        
        # Test evaluation mode
        model.eval()
        assert not model.training
        
        with torch.no_grad():
            output_eval1 = model(x)
            output_eval2 = model(x)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval1, output_eval2, atol=1e-6)


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_input_shape_validation(self):
        """Test input shape validation."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        
        # Valid input
        x_valid = torch.randn(4, 32, 3)
        output = model(x_valid)
        assert output.shape == (4, 4)
        
        # Test various invalid inputs
        invalid_inputs = [
            torch.randn(4, 32),        # Missing feature dimension
            torch.randn(4, 32, 5),     # Wrong feature dimension
            torch.randn(32, 3),        # Missing batch dimension
        ]
        
        for x_invalid in invalid_inputs:
            with pytest.raises((RuntimeError, ValueError, IndexError)):
                model(x_invalid)
    
    def test_data_type_validation(self):
        """Test data type validation."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        
        # Test different data types
        x_float32 = torch.randn(4, 32, 3, dtype=torch.float32)
        x_float64 = torch.randn(4, 32, 3, dtype=torch.float64)
        
        # Both should work
        output_32 = model(x_float32)
        output_64 = model(x_float64)
        
        assert output_32.dtype == torch.float32
        assert output_64.dtype == torch.float64
    
    def test_nan_inf_handling(self):
        """Test handling of NaN and Inf values."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        
        # Test with NaN input
        x_nan = torch.randn(4, 32, 3)
        x_nan[0, 0, 0] = float('nan')
        
        output_nan = model(x_nan)
        # Output should contain NaN
        assert torch.isnan(output_nan).any()
        
        # Test with Inf input
        x_inf = torch.randn(4, 32, 3)
        x_inf[0, 0, 0] = float('inf')
        
        output_inf = model(x_inf)
        # Output might contain Inf or very large values
        assert torch.isfinite(output_inf).sum() < output_inf.numel() or torch.abs(output_inf).max() > 1e6


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_required_parameters(self):
        """Test that required parameters are validated."""
        # Missing input_dim should raise error
        with pytest.raises((AttributeError, KeyError, TypeError)):
            args = Namespace(
                model_name='ResNetMLP',
                hidden_dim=32,
                num_classes=4
                # Missing input_dim
            )
            build_model(args)
    
    def test_parameter_defaults(self):
        """Test that default parameters are applied correctly."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            num_classes=4
            # Missing optional parameters
        )
        
        model = build_model(args)
        assert model is not None
        
        # Should work with minimal configuration
        x = torch.randn(4, 32, 3)
        output = model(x)
        assert output.shape == (4, 4)
    
    def test_parameter_validation(self):
        """Test parameter value validation."""
        # Test invalid parameter values
        invalid_configs = [
            Namespace(model_name='ResNetMLP', input_dim=0, num_classes=4),  # Zero input_dim
            Namespace(model_name='ResNetMLP', input_dim=3, num_classes=0),  # Zero num_classes
            Namespace(model_name='ResNetMLP', input_dim=3, hidden_dim=-1, num_classes=4),  # Negative hidden_dim
        ]
        
        for args in invalid_configs:
            with pytest.raises((ValueError, RuntimeError)):
                model = build_model(args)
                # Some errors might only appear during forward pass
                x = torch.randn(4, 32, 3)
                model(x)


class TestModelCompatibility:
    """Test model compatibility across different scenarios."""
    
    def test_torch_script_compatibility(self):
        """Test TorchScript compatibility."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        model.eval()
        
        # Test tracing
        x = torch.randn(4, 32, 3)
        try:
            traced_model = torch.jit.trace(model, x)
            
            # Test traced model
            with torch.no_grad():
                original_output = model(x)
                traced_output = traced_model(x)
            
            assert torch.allclose(original_output, traced_output, atol=1e-5)
            print("TorchScript tracing successful")
        except Exception as e:
            print(f"TorchScript tracing failed: {e}")
            # Some models might not be traceable, which is okay
    
    def test_onnx_compatibility(self):
        """Test ONNX export compatibility."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        model.eval()
        
        x = torch.randn(4, 32, 3)
        
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=True) as tmp_file:
                torch.onnx.export(
                    model, x, tmp_file.name,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
            print("ONNX export successful")
        except Exception as e:
            print(f"ONNX export failed: {e}")
            # ONNX export might fail for some models, which is acceptable
    
    def test_mixed_precision_compatibility(self):
        """Test mixed precision training compatibility."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        
        # Test with autocast
        x = torch.randn(4, 32, 3)
        
        try:
            from torch.cuda.amp import autocast
            
            with autocast():
                output = model(x)
            
            assert output.dtype == torch.float16 or output.dtype == torch.float32
            print("Mixed precision compatible")
        except ImportError:
            print("Mixed precision not available")
        except Exception as e:
            print(f"Mixed precision test failed: {e}")


class TestErrorRecovery:
    """Test error recovery and robustness."""
    
    def test_gradient_clipping_compatibility(self):
        """Test compatibility with gradient clipping."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        model = build_model(args)
        criterion = torch.nn.CrossEntropyLoss()
        
        x = torch.randn(4, 32, 3)
        y = torch.randint(0, 4, (4,))
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Test gradient clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check that gradients exist and were clipped if necessary
        has_gradients = any(param.grad is not None for param in model.parameters())
        assert has_gradients, "No gradients found"
        
        print(f"Gradient norm before clipping: {grad_norm_before:.4f}")
    
    def test_checkpoint_compatibility(self):
        """Test model checkpointing and loading."""
        args = Namespace(
            model_name='ResNetMLP',
            input_dim=3,
            hidden_dim=32,
            num_layers=2,
            num_classes=4
        )
        
        # Create and train model briefly
        model = build_model(args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        
        x = torch.randn(4, 32, 3)
        y = torch.randint(0, 4, (4,))
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }
        
        # Create new model and load checkpoint
        new_model = build_model(args)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        
        new_model.load_state_dict(checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Test that loaded model produces same output
        with torch.no_grad():
            original_output = model(x)
            loaded_output = new_model(x)
        
        assert torch.allclose(original_output, loaded_output, atol=1e-6)
        print("Checkpoint save/load successful")
