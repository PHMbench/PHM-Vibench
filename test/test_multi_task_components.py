"""
Comprehensive unit tests for multi-task PHM foundation model components.

This test suite validates the functionality of the multi-task head module
and related components without requiring the full PyTorch Lightning environment.

Author: PHM-Vibench Team
Date: 2025-08-18
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace
import unittest

# Import the multi-task head
from src.model_factory.ISFM.task_head.multi_task_head import MultiTaskHead


class TestMultiTaskHead(unittest.TestCase):
    """Test cases for the MultiTaskHead module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.args = Namespace(
            output_dim=512,
            hidden_dim=256,
            dropout=0.1,
            num_classes={'system1': 5, 'system2': 3, 'system3': 7},
            rul_max_value=1000.0,
            use_batch_norm=True,
            activation='relu'
        )
        self.model = MultiTaskHead(self.args)
        self.batch_size = 16
        self.seq_len = 128
        self.feature_dim = 512
    
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, MultiTaskHead)
        self.assertEqual(self.model.input_dim, 512)
        self.assertEqual(self.model.hidden_dim, 256)
        self.assertEqual(len(self.model.fault_classification_heads), 3)
        
        # Check that all classification heads exist
        for system_id in ['system1', 'system2', 'system3']:
            self.assertIn(system_id, self.model.fault_classification_heads)
    
    def test_forward_3d_input(self):
        """Test forward pass with 3D input (B, L, C)."""
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        
        # Test classification
        cls_output = self.model(x, system_id='system1', task_id='classification')
        self.assertEqual(cls_output.shape, (self.batch_size, 5))
        
        # Test RUL prediction
        rul_output = self.model(x, task_id='rul_prediction')
        self.assertEqual(rul_output.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(rul_output >= 0))  # RUL should be non-negative
        
        # Test anomaly detection
        anomaly_output = self.model(x, task_id='anomaly_detection')
        self.assertEqual(anomaly_output.shape, (self.batch_size, 1))
    
    def test_forward_2d_input(self):
        """Test forward pass with 2D input (B, C)."""
        x = torch.randn(self.batch_size, self.feature_dim)
        
        cls_output = self.model(x, system_id='system2', task_id='classification')
        self.assertEqual(cls_output.shape, (self.batch_size, 3))
    
    def test_forward_all_tasks(self):
        """Test forward pass for all tasks simultaneously."""
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        
        all_outputs = self.model(x, system_id='system1', task_id='all')
        
        self.assertIsInstance(all_outputs, dict)
        self.assertIn('classification', all_outputs)
        self.assertIn('rul_prediction', all_outputs)
        self.assertIn('anomaly_detection', all_outputs)
        
        # Check shapes
        self.assertEqual(all_outputs['classification'].shape, (self.batch_size, 5))
        self.assertEqual(all_outputs['rul_prediction'].shape, (self.batch_size, 1))
        self.assertEqual(all_outputs['anomaly_detection'].shape, (self.batch_size, 1))
    
    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        
        features = self.model(x, return_feature=True)
        self.assertEqual(features.shape, (self.batch_size, 256))
    
    def test_invalid_system_id(self):
        """Test error handling for invalid system ID."""
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        
        with self.assertRaises(ValueError):
            self.model(x, system_id='invalid_system', task_id='classification')
    
    def test_invalid_task_id(self):
        """Test error handling for invalid task ID."""
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        
        with self.assertRaises(ValueError):
            self.model(x, task_id='invalid_task')
    
    def test_missing_system_id_for_classification(self):
        """Test error handling when system_id is missing for classification."""
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        
        with self.assertRaises(ValueError):
            self.model(x, task_id='classification')
    
    def test_invalid_input_dimensions(self):
        """Test error handling for invalid input dimensions."""
        # Test 1D input (should fail)
        x_1d = torch.randn(self.feature_dim)
        with self.assertRaises(ValueError):
            self.model(x_1d, task_id='rul_prediction')
        
        # Test 4D input (should fail)
        x_4d = torch.randn(self.batch_size, self.seq_len, self.feature_dim, 10)
        with self.assertRaises(ValueError):
            self.model(x_4d, task_id='rul_prediction')
    
    def test_different_activations(self):
        """Test model with different activation functions."""
        activations = ['relu', 'gelu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
        
        for activation in activations:
            args = Namespace(
                output_dim=512,
                hidden_dim=256,
                dropout=0.1,
                num_classes={'system1': 5},
                rul_max_value=1000.0,
                use_batch_norm=True,
                activation=activation
            )
            
            model = MultiTaskHead(args)
            x = torch.randn(4, 128, 512)
            
            # Test that forward pass works
            output = model(x, system_id='system1', task_id='classification')
            self.assertEqual(output.shape, (4, 5))
    
    def test_batch_norm_disabled(self):
        """Test model without batch normalization."""
        args = Namespace(
            output_dim=512,
            hidden_dim=256,
            dropout=0.1,
            num_classes={'system1': 5},
            rul_max_value=1000.0,
            use_batch_norm=False,
            activation='relu'
        )
        
        model = MultiTaskHead(args)
        x = torch.randn(4, 128, 512)
        
        output = model(x, system_id='system1', task_id='classification')
        self.assertEqual(output.shape, (4, 5))
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model."""
        x = torch.randn(4, 128, 512, requires_grad=True)
        
        # Test classification gradient flow
        cls_output = self.model(x, system_id='system1', task_id='classification')
        loss = cls_output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))
        
        # Reset gradients
        x.grad = None
        
        # Test RUL prediction gradient flow
        rul_output = self.model(x, task_id='rul_prediction')
        loss = rul_output.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.any(x.grad != 0))


class TestMultiTaskIntegration(unittest.TestCase):
    """Integration tests for multi-task components."""
    
    def test_model_parameter_count(self):
        """Test that the model has a reasonable number of parameters."""
        args = Namespace(
            output_dim=512,
            hidden_dim=256,
            dropout=0.1,
            num_classes={'system1': 5, 'system2': 3},
            rul_max_value=1000.0,
            use_batch_norm=True,
            activation='relu'
        )
        
        model = MultiTaskHead(args)
        param_count = sum(p.numel() for p in model.parameters())
        
        # Should have a reasonable number of parameters (not too few, not too many)
        self.assertGreater(param_count, 100000)  # At least 100k parameters
        self.assertLess(param_count, 10000000)   # Less than 10M parameters
    
    def test_model_device_compatibility(self):
        """Test model compatibility with different devices."""
        args = Namespace(
            output_dim=512,
            hidden_dim=256,
            dropout=0.1,
            num_classes={'system1': 5},
            rul_max_value=1000.0,
            use_batch_norm=True,
            activation='relu'
        )
        
        model = MultiTaskHead(args)
        
        # Test CPU
        x_cpu = torch.randn(4, 128, 512)
        output_cpu = model(x_cpu, system_id='system1', task_id='classification')
        self.assertEqual(output_cpu.device.type, 'cpu')
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x_cpu.cuda()
            output_cuda = model_cuda(x_cuda, system_id='system1', task_id='classification')
            self.assertEqual(output_cuda.device.type, 'cuda')


def run_tests():
    """Run all tests."""
    print("Running Multi-Task PHM Foundation Model Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMultiTaskHead))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiTaskIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
