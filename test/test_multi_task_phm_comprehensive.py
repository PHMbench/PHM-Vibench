"""
Comprehensive unit tests for multi_task_phm.py module.

Tests all critical functionality including:
- Task initialization and configuration
- Loss computation for all task types
- Batch metadata processing
- Metric computation
- Error handling and edge cases

Author: PHM-Vibench Team
Date: 2025-09-08
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import torch
import torch.nn as nn
from argparse import Namespace
from unittest.mock import patch, MagicMock
import numpy as np

from src.task_factory.task.In_distribution.multi_task_phm import task as MultiTaskPHM


class MockMultiTaskOutput:
    """Mock output from multi-task model with proper dimensions."""
    def __init__(self, batch_size=4, seq_len=128, num_classes=4):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # Classification logits
        self.classification_logits = torch.randn(batch_size, num_classes)
        
        # Anomaly detection logits (binary) - with extra dimension to test dimension fix
        self.anomaly_detection_logits = torch.randn(batch_size, 1)
        
        # Signal prediction logits
        self.signal_prediction_logits = torch.randn(batch_size, seq_len, 2)
        
        # RUL prediction logits
        self.rul_prediction_logits = torch.randn(batch_size, 1)


class MockNetwork(torch.nn.Module):
    """Mock network for testing."""
    def __init__(self, return_dict=True, missing_tasks=None):
        super().__init__()
        self.return_dict = return_dict
        self.missing_tasks = missing_tasks or []
        
    def forward(self, x, file_id=None, task_id=None):
        B, L, C = x.shape if len(x.shape) == 3 else (x.shape[0], 128, 2)
        mock_output = MockMultiTaskOutput(B, L, 4)
        
        if self.return_dict:
            # Return as dictionary
            output_dict = {
                'classification': mock_output.classification_logits,
                'anomaly_detection': mock_output.anomaly_detection_logits,
                'signal_prediction': mock_output.signal_prediction_logits,
                'rul_prediction': mock_output.rul_prediction_logits
            }
            # Remove missing tasks to test error handling
            for task in self.missing_tasks:
                output_dict.pop(task, None)
            return output_dict
        else:
            # Return as object with attributes
            return mock_output


class TestMultiTaskPHMComprehensive(unittest.TestCase):
    """Comprehensive test cases for multi_task_phm module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.args_data = Namespace()
        self.args_model = Namespace()
        self.args_task = Namespace(
            enabled_tasks=['classification', 'anomaly_detection', 'signal_prediction', 'rul_prediction'],
            task_weights={'classification': 1.0, 'anomaly_detection': 0.6, 'signal_prediction': 0.7, 'rul_prediction': 0.8},
            optimizer='adamw',
            lr=0.001,
            weight_decay=0.01
        )
        self.args_trainer = Namespace(gpus=False)
        self.args_environment = Namespace()
        
        # Mock metadata with different scenarios
        self.metadata = {
            1: {'Name': 'test_dataset', 'RUL_label': 500.0, 'Label': 2, 'Dataset_id': 1},
            2: {'Name': 'test_dataset', 'RUL_label': 750.0, 'Label': 1}, # Missing Dataset_id
            3: {'Name': 'test_dataset', 'RUL_label': None, 'Label': 0}, # None RUL
            4: {'Name': 'test_dataset', 'Label': 3}, # Missing RUL_label entirely
            5: {'Name': 'test_dataset', 'RUL_label': np.nan, 'Label': 1}, # NaN RUL
        }
        
        self.network = MockNetwork()
        self.task_module = MultiTaskPHM(
            self.network,
            self.args_data,
            self.args_model,
            self.args_task,
            self.args_trainer,
            self.args_environment,
            self.metadata
        )

    # ========== Task Initialization Tests ==========
    
    def test_task_initialization_all_enabled(self):
        """Test initialization with all tasks enabled."""
        self.assertEqual(len(self.task_module.enabled_tasks), 4)
        self.assertIn('classification', self.task_module.enabled_tasks)
        self.assertIn('anomaly_detection', self.task_module.enabled_tasks)
        self.assertIn('signal_prediction', self.task_module.enabled_tasks)
        self.assertIn('rul_prediction', self.task_module.enabled_tasks)
    
    def test_task_initialization_partial_enabled(self):
        """Test initialization with only some tasks enabled."""
        limited_args = Namespace(
            enabled_tasks=['classification', 'rul_prediction'],
            task_weights={'classification': 1.0, 'rul_prediction': 0.8},
            optimizer='adam',
            lr=0.001
        )
        
        limited_task_module = MultiTaskPHM(
            self.network, self.args_data, self.args_model, limited_args,
            self.args_trainer, self.args_environment, self.metadata
        )
        
        self.assertEqual(len(limited_task_module.enabled_tasks), 2)
        self.assertIn('classification', limited_task_module.enabled_tasks)
        self.assertIn('rul_prediction', limited_task_module.enabled_tasks)
    
    def test_task_weight_initialization(self):
        """Test task weight initialization."""
        self.assertEqual(self.task_module.task_weights['classification'], 1.0)
        self.assertEqual(self.task_module.task_weights['anomaly_detection'], 0.6)
        self.assertEqual(self.task_module.task_weights['signal_prediction'], 0.7)
        self.assertEqual(self.task_module.task_weights['rul_prediction'], 0.8)
    
    def test_loss_function_initialization(self):
        """Test that loss functions are properly initialized."""
        self.assertIn('classification', self.task_module.task_loss_fns)
        self.assertIn('anomaly_detection', self.task_module.task_loss_fns)
        self.assertIn('signal_prediction', self.task_module.task_loss_fns)
        self.assertIn('rul_prediction', self.task_module.task_loss_fns)
    
    def test_metric_initialization(self):
        """Test that metrics are properly initialized with correct types."""
        # Test classification metrics
        self.assertIn('classification', self.task_module.task_metrics)
        for stage in ['train', 'val', 'test']:
            self.assertIn(stage, self.task_module.task_metrics['classification'])
    
    # ========== Loss Computation Tests ==========
    
    def test_classification_loss_computation(self):
        """Test classification loss computation."""
        task_output = torch.randn(4, 4)  # [batch_size, num_classes]
        targets = torch.tensor([0, 1, 2, 3])
        
        loss = self.task_module._compute_task_loss('classification', task_output, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() >= 0)
    
    def test_anomaly_detection_loss_computation(self):
        """Test anomaly detection loss computation with dimension fix."""
        # Test with extra dimension (common scenario that was causing issues)
        task_output = torch.randn(4, 1)  # [batch_size, 1]
        targets = torch.tensor([0.0, 1.0, 1.0, 0.0])  # Binary targets
        
        loss = self.task_module._compute_task_loss('anomaly_detection', task_output, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() >= 0)
    
    def test_anomaly_detection_dimension_fix(self):
        """Test that BCE loss dimension fix works properly."""
        # Test various input dimensions
        test_cases = [
            (torch.randn(4, 1), torch.tensor([0.0, 1.0, 1.0, 0.0])),  # Extra dimension
            (torch.randn(4), torch.tensor([0.0, 1.0, 1.0, 0.0])),     # Correct dimension
        ]
        
        for task_output, targets in test_cases:
            with self.subTest(output_shape=task_output.shape):
                loss = self.task_module._compute_task_loss('anomaly_detection', task_output, targets)
                self.assertIsInstance(loss, torch.Tensor)
                self.assertTrue(loss.item() >= 0)
    
    def test_signal_prediction_loss_computation(self):
        """Test signal prediction loss computation."""
        task_output = torch.randn(4, 128, 2)
        x = torch.randn(4, 128, 2)  # Use input as reconstruction target
        
        loss = self.task_module._compute_task_loss('signal_prediction', task_output, None, x)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() >= 0)
    
    def test_rul_prediction_loss_computation(self):
        """Test RUL prediction loss computation."""
        task_output = torch.randn(4, 1)
        targets = torch.tensor([500.0, 750.0, 300.0, 900.0])
        
        loss = self.task_module._compute_task_loss('rul_prediction', task_output, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTrue(loss.item() >= 0)

    # ========== Batch Metadata Processing Tests ==========
    
    def test_batch_metadata_processing_normal(self):
        """Test normal batch metadata processing."""
        y = torch.tensor([0, 1, 2, 3])
        file_ids = [1, 2, 3, 4]
        
        y_dict = self.task_module._build_task_labels_batch(y, file_ids)
        
        self.assertIn('classification', y_dict)
        self.assertIn('rul_prediction', y_dict)
        torch.testing.assert_close(y_dict['classification'], y)
    
    def test_batch_metadata_missing_rul(self):
        """Test batch processing with missing RUL values."""
        y = torch.tensor([0, 1, 2, 3])
        file_ids = [1, 3, 4, 5]  # File 3,4,5 have missing/invalid RUL
        
        with patch('builtins.print'):  # Suppress warnings
            y_dict = self.task_module._build_task_labels_batch(y, file_ids)
        
        self.assertIn('rul_prediction', y_dict)
        rul_values = y_dict['rul_prediction']
        
        # First value should be real (500.0), others should be default (1000.0)
        self.assertAlmostEqual(rul_values[0].item(), 500.0, places=1)
        self.assertAlmostEqual(rul_values[1].item(), 1000.0, places=1)  # None RUL
        self.assertAlmostEqual(rul_values[2].item(), 1000.0, places=1)  # Missing RUL_label
        self.assertAlmostEqual(rul_values[3].item(), 1000.0, places=1)  # NaN RUL
    
    def test_batch_metadata_tensor_file_ids(self):
        """Test batch processing with tensor file_ids."""
        y = torch.tensor([0, 1, 2])
        file_ids = torch.tensor([1, 2, 3])
        
        y_dict = self.task_module._build_task_labels_batch(y, file_ids)
        
        self.assertIn('classification', y_dict)
        self.assertIn('rul_prediction', y_dict)
    
    def test_batch_metadata_single_file_id(self):
        """Test batch processing when single file_id is expanded."""
        y = torch.tensor([0, 1, 2])
        file_ids = [1]  # Single file_id for batch
        
        y_dict = self.task_module._build_task_labels_batch(y, file_ids)
        
        # Should expand single file_id to match batch size
        expected_rul = torch.tensor([500.0, 500.0, 500.0])
        torch.testing.assert_close(y_dict['rul_prediction'], expected_rul)

    # ========== Task Output Validation Tests ==========
    
    def test_task_output_validation_missing_task(self):
        """Test handling when model output is missing tasks."""
        network_missing_tasks = MockNetwork(missing_tasks=['anomaly_detection'])
        
        task_module = MultiTaskPHM(
            network_missing_tasks, self.args_data, self.args_model, self.args_task,
            self.args_trainer, self.args_environment, self.metadata
        )
        
        # Create mock batch
        batch = {
            'x': torch.randn(2, 128, 2),
            'y': torch.tensor([0, 1]),
            'file_id': torch.tensor([1, 2])
        }
        
        with patch('builtins.print'):  # Suppress warnings
            # Should not crash, should skip missing task
            loss = task_module._shared_step(batch, 0, mode='train')
            self.assertIsInstance(loss, torch.Tensor)
    
    def test_task_output_validation_non_tensor_output(self):
        """Test handling when task output is not a tensor."""
        # Create network that returns invalid output
        class InvalidOutputNetwork(nn.Module):
            def forward(self, x, file_id=None, task_id=None):
                return {
                    'classification': torch.randn(x.shape[0], 4),
                    'anomaly_detection': "invalid_output",  # Non-tensor
                    'signal_prediction': torch.randn(x.shape[0], 128, 2),
                    'rul_prediction': torch.randn(x.shape[0], 1)
                }
        
        invalid_network = InvalidOutputNetwork()
        task_module = MultiTaskPHM(
            invalid_network, self.args_data, self.args_model, self.args_task,
            self.args_trainer, self.args_environment, self.metadata
        )
        
        batch = {
            'x': torch.randn(2, 128, 2),
            'y': torch.tensor([0, 1]),
            'file_id': torch.tensor([1, 2])
        }
        
        with patch('builtins.print'):  # Suppress warnings
            loss = task_module._shared_step(batch, 0, mode='train')
            self.assertIsInstance(loss, torch.Tensor)

    # ========== Metric Computation Tests ==========
    
    def test_classification_metrics_computation(self):
        """Test classification metrics computation."""
        task_output = torch.randn(4, 4)
        targets = torch.tensor([0, 1, 2, 3])
        
        metrics = self.task_module._compute_task_metrics('classification', task_output, targets, 'train')
        
        # Should have all classification metrics
        expected_metrics = ['classification_train_acc', 'classification_train_f1', 
                           'classification_train_precision', 'classification_train_recall']
        
        for metric_name in expected_metrics:
            self.assertIn(metric_name, metrics)
    
    def test_anomaly_detection_metrics_computation(self):
        """Test anomaly detection metrics computation."""
        task_output = torch.randn(4, 1)
        targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
        
        metrics = self.task_module._compute_task_metrics('anomaly_detection', task_output, targets, 'val')
        
        # Should have all anomaly detection metrics
        expected_metrics = ['anomaly_detection_val_acc', 'anomaly_detection_val_f1', 
                           'anomaly_detection_val_precision', 'anomaly_detection_val_recall',
                           'anomaly_detection_val_auroc']
        
        for metric_name in expected_metrics:
            self.assertIn(metric_name, metrics)

    # ========== Error Handling Tests ==========
    
    def test_error_handling_in_shared_step(self):
        """Test error handling during shared step execution."""
        # Create network that raises exception
        class ErrorNetwork(nn.Module):
            def forward(self, x, file_id=None, task_id=None):
                raise RuntimeError("Simulated network error")
        
        error_network = ErrorNetwork()
        task_module = MultiTaskPHM(
            error_network, self.args_data, self.args_model, self.args_task,
            self.args_trainer, self.args_environment, self.metadata
        )
        
        batch = {
            'x': torch.randn(2, 128, 2),
            'y': torch.tensor([0, 1]),
            'file_id': torch.tensor([1, 2])
        }
        
        with patch('builtins.print'):  # Suppress error prints
            loss = task_module._shared_step(batch, 0, mode='train')
            # Should return zero loss when all tasks fail
            self.assertEqual(loss.item(), 0.0)
    
    def test_loss_computation_error_handling(self):
        """Test error handling in loss computation."""
        # Test with incompatible shapes to trigger error
        task_output = torch.randn(4, 3)  # Wrong number of classes
        targets = torch.tensor([0, 1, 2, 5])  # Class 5 doesn't exist
        
        with patch('builtins.print'):  # Suppress warnings
            # Should handle the error gracefully
            try:
                loss = self.task_module._compute_task_loss('classification', task_output, targets)
                # If no exception, loss should still be valid
                if loss is not None:
                    self.assertIsInstance(loss, torch.Tensor)
            except Exception:
                # Exception is acceptable in this error scenario
                pass

    # ========== Edge Cases ==========
    
    def test_empty_enabled_tasks(self):
        """Test behavior with empty enabled tasks."""
        empty_args = Namespace(
            enabled_tasks=[],
            task_weights={},
            optimizer='adam',
            lr=0.001
        )
        
        task_module = MultiTaskPHM(
            self.network, self.args_data, self.args_model, empty_args,
            self.args_trainer, self.args_environment, self.metadata
        )
        
        self.assertEqual(len(task_module.enabled_tasks), 0)
    
    def test_single_sample_batch(self):
        """Test with single sample batch."""
        batch = {
            'x': torch.randn(1, 128, 2),
            'y': torch.tensor([0]),
            'file_id': torch.tensor([1])
        }
        
        loss = self.task_module._shared_step(batch, 0, mode='train')
        self.assertIsInstance(loss, torch.Tensor)
    
    def test_large_batch_processing(self):
        """Test with larger batch size."""
        batch_size = 32
        batch = {
            'x': torch.randn(batch_size, 128, 2),
            'y': torch.randint(0, 4, (batch_size,)),
            'file_id': torch.randint(1, 6, (batch_size,))  # Random file_ids from available metadata
        }
        
        with patch('builtins.print'):  # Suppress potential warnings
            loss = self.task_module._shared_step(batch, 0, mode='train')
            self.assertIsInstance(loss, torch.Tensor)


if __name__ == '__main__':
    unittest.main(verbosity=2)