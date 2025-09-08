"""
Integration tests for task-specific metrics computation in multi-task PHM training.

Tests the bug fixes implemented for computing and logging task-specific metrics
for all four task types: classification, anomaly detection, signal prediction,
and RUL prediction.

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

from src.task_factory.task.In_distribution.multi_task_phm import task as MultiTaskPHM


class MockMultiTaskOutput:
    """Mock output from multi-task model."""
    def __init__(self, batch_size=4, seq_len=128, num_classes=4):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # Classification logits (reduced to 4 classes to match metadata)
        self.classification_logits = torch.randn(batch_size, num_classes)
        
        # Anomaly detection logits (binary)
        self.anomaly_detection_logits = torch.randn(batch_size, 1)
        
        # Signal prediction logits
        self.signal_prediction_logits = torch.randn(batch_size, seq_len, 2)
        
        # RUL prediction logits
        self.rul_prediction_logits = torch.randn(batch_size, 1)


class MockNetwork(torch.nn.Module):
    """Mock network for testing."""
    def forward(self, x, file_id=None, task_id=None):
        B, L, C = x.shape
        return MockMultiTaskOutput(B, L, 4)


class TestTaskSpecificMetrics(unittest.TestCase):
    """Test cases for task-specific metrics computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.args_data = Namespace()
        self.args_model = Namespace()
        self.args_task = Namespace(
            enabled_tasks=['classification', 'anomaly_detection', 'signal_prediction', 'rul_prediction'],
            task_weights={'classification': 1.0, 'anomaly_detection': 0.6, 'signal_prediction': 0.7, 'rul_prediction': 0.8},
            optimizer='adam',
            lr=0.001
        )
        self.args_trainer = Namespace(gpus=False)
        self.args_environment = Namespace()
        
        # Mock metadata with proper labels
        self.metadata = {
            1: {'Name': 'test_dataset', 'RUL_label': 500.0, 'Label': 2},
            2: {'Name': 'test_dataset', 'RUL_label': 750.0, 'Label': 1}, 
            3: {'Name': 'test_dataset', 'RUL_label': 300.0, 'Label': 0},
            4: {'Name': 'test_dataset', 'RUL_label': 900.0, 'Label': 3},
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
    
    def test_classification_metrics_computation(self):
        """Test that classification metrics are computed correctly."""
        # Create mock outputs
        mock_output = MockMultiTaskOutput(batch_size=4)
        
        # Create target labels (0-3 to match 4 classes in mock)
        targets = torch.tensor([0, 1, 2, 3])
        
        # Test metrics computation with extracted task output
        task_output = mock_output.classification_logits
        metrics = self.task_module._compute_task_metrics('classification', task_output, targets, 'train')
        
        # Should have classification metrics
        expected_metrics = ['classification_train_acc', 'classification_train_f1', 
                           'classification_train_precision', 'classification_train_recall']
        
        for metric_name in expected_metrics:
            self.assertIn(metric_name, metrics)
            metric_value = metrics[metric_name]
            self.assertIsInstance(metric_value, torch.Tensor)
            # Accuracy, F1, precision, recall should be between 0 and 1
            if 'acc' in metric_name or 'f1' in metric_name or 'precision' in metric_name or 'recall' in metric_name:
                self.assertTrue(0 <= metric_value <= 1, f"{metric_name} should be between 0 and 1, got {metric_value}")
    
    def test_anomaly_detection_metrics_computation(self):
        """Test that anomaly detection metrics are computed correctly."""
        # Create mock outputs  
        mock_output = MockMultiTaskOutput(batch_size=4)
        
        # Binary targets (0=normal, 1=anomaly)
        targets = torch.tensor([0.0, 1.0, 1.0, 0.0])
        
        # Test metrics computation with extracted task output
        task_output = mock_output.anomaly_detection_logits
        metrics = self.task_module._compute_task_metrics('anomaly_detection', task_output, targets, 'val')
        
        # Should have anomaly detection metrics
        expected_metrics = ['anomaly_detection_val_acc', 'anomaly_detection_val_f1', 
                           'anomaly_detection_val_precision', 'anomaly_detection_val_recall',
                           'anomaly_detection_val_auroc']
        
        for metric_name in expected_metrics:
            self.assertIn(metric_name, metrics)
            metric_value = metrics[metric_name]
            self.assertIsInstance(metric_value, torch.Tensor)
            # All metrics should be between 0 and 1
            self.assertTrue(0 <= metric_value <= 1, f"{metric_name} should be between 0 and 1, got {metric_value}")
    
    def test_signal_prediction_metrics_computation(self):
        """Test that signal prediction metrics are computed correctly."""
        # Create mock outputs
        mock_output = MockMultiTaskOutput(batch_size=4, seq_len=128)
        
        # Regression targets (same shape as predictions)
        targets = torch.randn(4, 128, 2)
        
        # Test metrics computation with extracted task output
        task_output = mock_output.signal_prediction_logits
        metrics = self.task_module._compute_task_metrics('signal_prediction', task_output, targets, 'test')
        
        # Should have regression metrics
        expected_metrics = ['signal_prediction_test_mse', 'signal_prediction_test_mae', 'signal_prediction_test_r2']
        
        for metric_name in expected_metrics:
            self.assertIn(metric_name, metrics)
            metric_value = metrics[metric_name]
            self.assertIsInstance(metric_value, torch.Tensor)
            # MSE and MAE should be non-negative, R2 can be negative
            if 'mse' in metric_name or 'mae' in metric_name:
                self.assertTrue(metric_value >= 0, f"{metric_name} should be non-negative, got {metric_value}")
    
    def test_rul_prediction_metrics_computation(self):
        """Test that RUL prediction metrics are computed correctly."""
        # Create mock outputs
        mock_output = MockMultiTaskOutput(batch_size=4)
        
        # RUL targets (scalar values)
        targets = torch.tensor([500.0, 750.0, 300.0, 900.0])
        
        # Test metrics computation with extracted task output
        task_output = mock_output.rul_prediction_logits
        metrics = self.task_module._compute_task_metrics('rul_prediction', task_output, targets, 'val')
        
        # Should have regression metrics including MAPE
        expected_metrics = ['rul_prediction_val_mse', 'rul_prediction_val_mae', 
                           'rul_prediction_val_r2', 'rul_prediction_val_mape']
        
        for metric_name in expected_metrics:
            self.assertIn(metric_name, metrics)
            metric_value = metrics[metric_name]
            self.assertIsInstance(metric_value, torch.Tensor)
            # MSE, MAE, MAPE should be non-negative
            if any(reg_metric in metric_name for reg_metric in ['mse', 'mae', 'mape']):
                self.assertTrue(metric_value >= 0, f"{metric_name} should be non-negative, got {metric_value}")
    
    def test_unsupported_task_metrics(self):
        """Test that unsupported tasks return empty metrics."""
        mock_output = MockMultiTaskOutput(batch_size=4)
        targets = torch.tensor([1, 2, 0, 1])
        
        # Test with unsupported task name
        metrics = self.task_module._compute_task_metrics('unsupported_task', mock_output, targets, 'train')
        
        # Should return empty dict
        self.assertEqual(len(metrics), 0)
    
    def test_missing_task_in_metrics_dict(self):
        """Test behavior when task not in metrics dictionary."""
        # Create task module with limited tasks
        limited_args = Namespace(
            enabled_tasks=['classification'],  # Only classification
            task_weights={'classification': 1.0},
            optimizer='adam',
            lr=0.001
        )
        
        limited_task_module = MultiTaskPHM(
            self.network,
            self.args_data,
            self.args_model,
            limited_args,
            self.args_trainer,
            self.args_environment,
            self.metadata
        )
        
        mock_output = MockMultiTaskOutput(batch_size=4)
        targets = torch.tensor([500.0, 750.0, 300.0, 900.0])
        
        # Try to compute metrics for task not in enabled_tasks
        metrics = limited_task_module._compute_task_metrics('rul_prediction', mock_output, targets, 'train')
        
        # Should return empty dict since rul_prediction not in task_metrics
        self.assertEqual(len(metrics), 0)
    
    def test_metric_computation_error_handling(self):
        """Test that metric computation errors are handled gracefully."""
        mock_output = MockMultiTaskOutput(batch_size=4)
        
        # Create mismatched targets that might cause errors
        targets = torch.tensor([1, 2])  # Wrong batch size
        
        # Should not raise exception, but might return fewer metrics
        with patch('builtins.print') as mock_print:
            metrics = self.task_module._compute_task_metrics('classification', mock_output, targets, 'train')
        
        # Should either return empty dict or partial metrics, but not crash
        self.assertIsInstance(metrics, dict)
        
        # If warnings were printed, they should mention failed metrics
        if mock_print.called:
            args, _ = mock_print.call_args
            self.assertIn('Failed to compute', args[0])
    
    def test_dimension_mismatch_handling(self):
        """Test handling of dimension mismatches between predictions and targets."""
        mock_output = MockMultiTaskOutput(batch_size=4)
        
        # Test RUL prediction with extra dimension
        mock_output.rul_prediction_logits = torch.randn(4, 1, 1)  # Extra dimension
        targets = torch.tensor([500.0, 750.0, 300.0, 900.0])
        
        # Should handle dimension squeeze correctly
        metrics = self.task_module._compute_task_metrics('rul_prediction', mock_output, targets, 'train')
        
        # Should still compute metrics successfully
        self.assertGreater(len(metrics), 0)
        self.assertIn('rul_prediction_train_mse', metrics)
    
    def test_all_stages_metrics_computation(self):
        """Test that metrics computation works for all stages."""
        mock_output = MockMultiTaskOutput(batch_size=2)
        targets = torch.tensor([0, 1])
        
        for stage in ['train', 'val', 'test']:
            metrics = self.task_module._compute_task_metrics('classification', mock_output, targets, stage)
            
            # All metrics should be prefixed with the stage
            for metric_name in metrics.keys():
                self.assertIn(f'classification_{stage}', metric_name)
    
    def test_batch_size_consistency(self):
        """Test that metrics work with different batch sizes."""
        for batch_size in [1, 8, 16, 32]:
            mock_output = MockMultiTaskOutput(batch_size=batch_size)
            targets = torch.randint(0, 4, (batch_size,))  # Random class targets (0-3)
            
            metrics = self.task_module._compute_task_metrics('classification', mock_output, targets, 'train')
            
            # Should successfully compute metrics for any reasonable batch size
            self.assertGreater(len(metrics), 0)
            self.assertIn('classification_train_acc', metrics)
    
    def test_edge_case_single_class_targets(self):
        """Test metrics computation with single class targets."""
        mock_output = MockMultiTaskOutput(batch_size=4)
        
        # All targets are the same class
        targets = torch.tensor([1, 1, 1, 1])
        
        # Should still compute metrics (though some might be undefined like F1)
        with patch('builtins.print'):  # Suppress potential warnings
            metrics = self.task_module._compute_task_metrics('classification', mock_output, targets, 'train')
        
        # Should return some metrics
        self.assertIsInstance(metrics, dict)
    
    def test_zero_targets_regression(self):
        """Test regression metrics with zero targets."""
        mock_output = MockMultiTaskOutput(batch_size=4)
        
        # Zero RUL targets
        targets = torch.tensor([0.0, 0.0, 0.0, 0.0])
        
        # Should compute metrics successfully
        metrics = self.task_module._compute_task_metrics('rul_prediction', mock_output, targets, 'train')
        
        # Should have metrics computed
        self.assertGreater(len(metrics), 0)
        
        # MAPE might be problematic with zero targets, but should not crash
        if 'rul_prediction_train_mape' in metrics:
            mape_value = metrics['rul_prediction_train_mape']
            # MAPE with zero targets might be inf or very large, but should be a tensor
            self.assertIsInstance(mape_value, torch.Tensor)


if __name__ == '__main__':
    unittest.main()