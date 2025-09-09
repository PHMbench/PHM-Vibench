"""
Unit tests for regression metrics extensions in the metrics system.

Tests the bug fixes implemented for extending the metrics system
to support regression tasks (MSE, MAE, R2, MAPE) alongside 
classification metrics.

Author: PHM-Vibench Team  
Date: 2025-09-08
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import torch
import torch.nn as nn
from unittest.mock import patch

from src.task_factory.Components.metrics import get_metrics
import torchmetrics


class TestRegressionMetrics(unittest.TestCase):
    """Test cases for regression metrics extensions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock metadata for testing
        self.metadata_classification = {
            1: {'Name': 'dataset1', 'Label': 0},
            2: {'Name': 'dataset1', 'Label': 1},
            3: {'Name': 'dataset1', 'Label': 2},
        }
        
        self.metadata_mixed = {
            1: {'Name': 'dataset1', 'Label': 0},
            2: {'Name': 'dataset1', 'Label': 1}, 
            3: {'Name': 'dataset1', 'Label': 2},
            4: {'Name': 'dataset2', 'Label': 0},  # Second dataset
        }
    
    def test_classification_metrics_only(self):
        """Test that classification metrics still work correctly."""
        metric_names = ['acc', 'f1', 'precision', 'recall']
        metrics = get_metrics(metric_names, self.metadata_classification)
        
        self.assertIn('dataset1', metrics)
        dataset_metrics = metrics['dataset1']
        
        # Check all stages and metrics are present
        expected_keys = []
        for stage in ['train', 'val', 'test']:
            for metric in metric_names:
                expected_keys.append(f"{stage}_{metric}")
        
        for key in expected_keys:
            self.assertIn(key, dataset_metrics)
            
        # Check that metrics are torchmetrics instances
        self.assertTrue(hasattr(dataset_metrics['train_acc'], '__call__'))  # Callable metric
        self.assertTrue(hasattr(dataset_metrics['train_f1'], '__call__'))
        self.assertTrue(hasattr(dataset_metrics['train_precision'], '__call__'))
        self.assertTrue(hasattr(dataset_metrics['train_recall'], '__call__'))
    
    def test_regression_metrics_only(self):
        """Test that regression metrics work correctly."""
        metric_names = ['mse', 'mae', 'r2', 'mape']
        metrics = get_metrics(metric_names, self.metadata_classification)
        
        self.assertIn('dataset1', metrics)
        dataset_metrics = metrics['dataset1']
        
        # Check all stages and metrics are present
        expected_keys = []
        for stage in ['train', 'val', 'test']:
            for metric in metric_names:
                expected_keys.append(f"{stage}_{metric}")
        
        for key in expected_keys:
            self.assertIn(key, dataset_metrics)
            
        # Check that metrics are torchmetrics instances
        self.assertIsInstance(dataset_metrics['train_mse'], torchmetrics.MeanSquaredError)
        self.assertIsInstance(dataset_metrics['train_mae'], torchmetrics.MeanAbsoluteError)
        self.assertIsInstance(dataset_metrics['train_r2'], torchmetrics.R2Score)
        self.assertIsInstance(dataset_metrics['train_mape'], torchmetrics.MeanAbsolutePercentageError)
    
    def test_mixed_metrics(self):
        """Test that classification and regression metrics work together."""
        metric_names = ['acc', 'f1', 'mse', 'mae', 'r2']
        metrics = get_metrics(metric_names, self.metadata_classification)
        
        self.assertIn('dataset1', metrics)
        dataset_metrics = metrics['dataset1']
        
        # Check classification metrics are callable
        self.assertTrue(hasattr(dataset_metrics['train_acc'], '__call__'))
        self.assertTrue(hasattr(dataset_metrics['train_f1'], '__call__'))
        
        # Check regression metrics are callable
        self.assertTrue(hasattr(dataset_metrics['train_mse'], '__call__'))
        self.assertTrue(hasattr(dataset_metrics['train_mae'], '__call__'))
        self.assertTrue(hasattr(dataset_metrics['train_r2'], '__call__'))
    
    def test_regression_metrics_no_parameters(self):
        """Test that regression metrics don't require task/num_classes parameters."""
        metric_names = ['mse', 'mae', 'r2', 'mape']
        
        # Should not raise any errors about missing parameters
        try:
            metrics = get_metrics(metric_names, self.metadata_classification)
            # Test that we can instantiate the metrics
            mse_metric = metrics['dataset1']['train_mse']
            mae_metric = metrics['dataset1']['train_mae']
            
            # Test basic functionality
            preds = torch.tensor([1.0, 2.0, 3.0])
            targets = torch.tensor([1.1, 2.2, 2.9])
            
            mse_value = mse_metric(preds, targets)
            mae_value = mae_metric(preds, targets)
            
            self.assertIsInstance(mse_value, torch.Tensor)
            self.assertIsInstance(mae_value, torch.Tensor)
            self.assertTrue(mse_value >= 0)
            self.assertTrue(mae_value >= 0)
            
        except Exception as e:
            self.fail(f"Regression metrics should not require parameters: {e}")
    
    def test_classification_metrics_with_parameters(self):
        """Test that classification metrics still get required parameters."""
        metric_names = ['acc', 'f1']
        metrics = get_metrics(metric_names, self.metadata_classification)
        
        # Classification metrics should work with integer inputs
        acc_metric = metrics['dataset1']['train_acc']
        f1_metric = metrics['dataset1']['train_f1']
        
        # Test with multiclass predictions
        preds = torch.tensor([0, 1, 2, 1])
        targets = torch.tensor([0, 1, 2, 2])
        
        acc_value = acc_metric(preds, targets)
        f1_value = f1_metric(preds, targets)
        
        self.assertIsInstance(acc_value, torch.Tensor)
        self.assertIsInstance(f1_value, torch.Tensor)
        self.assertTrue(0 <= acc_value <= 1)
        self.assertTrue(0 <= f1_value <= 1)
    
    def test_auroc_metrics(self):
        """Test AUROC metrics for binary classification."""
        metric_names = ['auroc']
        metrics = get_metrics(metric_names, self.metadata_classification)
        
        auroc_metric = metrics['dataset1']['train_auroc']
        self.assertTrue(hasattr(auroc_metric, '__call__'))  # Callable metric
        
        # Test with multiclass probabilities (3 classes from metadata)  
        # Shape: [batch_size, num_classes]
        preds = torch.tensor([[0.7, 0.2, 0.1],
                             [0.1, 0.8, 0.1], 
                             [0.2, 0.1, 0.7],
                             [0.6, 0.3, 0.1]])
        targets = torch.tensor([0, 1, 2, 0])
        
        auroc_value = auroc_metric(preds, targets)
        self.assertIsInstance(auroc_value, torch.Tensor)
        self.assertTrue(0 <= auroc_value <= 1)
    
    def test_multiple_datasets(self):
        """Test that metrics work correctly with multiple datasets."""
        metric_names = ['acc', 'mse', 'mae']
        metrics = get_metrics(metric_names, self.metadata_mixed)
        
        self.assertIn('dataset1', metrics)
        self.assertIn('dataset2', metrics)
        
        # Both datasets should have the same metrics
        for dataset_name in ['dataset1', 'dataset2']:
            dataset_metrics = metrics[dataset_name]
            for stage in ['train', 'val', 'test']:
                for metric in metric_names:
                    key = f"{stage}_{metric}"
                    self.assertIn(key, dataset_metrics)
                    
        # Verify metrics are callable for both datasets
        self.assertTrue(hasattr(metrics['dataset1']['train_acc'], '__call__'))
        self.assertTrue(hasattr(metrics['dataset1']['train_mse'], '__call__'))
        self.assertTrue(hasattr(metrics['dataset2']['train_acc'], '__call__'))
        self.assertTrue(hasattr(metrics['dataset2']['train_mse'], '__call__'))
    
    def test_unsupported_metrics(self):
        """Test handling of unsupported metric names."""
        metric_names = ['acc', 'unsupported_metric', 'mse']
        
        with patch('builtins.print') as mock_print:
            metrics = get_metrics(metric_names, self.metadata_classification)
        
        # Should print warning for unsupported metric
        mock_print.assert_called()
        args, _ = mock_print.call_args
        self.assertIn('unsupported_metric', args[0])
        
        # Supported metrics should still work
        self.assertIn('dataset1', metrics)
        dataset_metrics = metrics['dataset1']
        self.assertIn('train_acc', dataset_metrics)
        self.assertIn('train_mse', dataset_metrics)
        
        # Unsupported metric should not be present
        self.assertNotIn('train_unsupported_metric', dataset_metrics)
    
    def test_empty_metadata(self):
        """Test behavior with empty metadata."""
        empty_metadata = {}
        metric_names = ['acc', 'mse']
        
        # Should not raise an error, but return empty metrics dict
        metrics = get_metrics(metric_names, empty_metadata)
        self.assertEqual(len(metrics), 0)
    
    def test_binary_vs_multiclass_handling(self):
        """Test that binary vs multiclass classification is handled correctly."""
        # Binary classification case (only 1 class max)
        binary_metadata = {
            1: {'Name': 'binary_dataset', 'Label': 0},
            2: {'Name': 'binary_dataset', 'Label': 1},
        }
        
        # Multiclass classification case (more than 2 classes)
        multiclass_metadata = {
            1: {'Name': 'multiclass_dataset', 'Label': 0},
            2: {'Name': 'multiclass_dataset', 'Label': 1},
            3: {'Name': 'multiclass_dataset', 'Label': 2},
            4: {'Name': 'multiclass_dataset', 'Label': 3},
        }
        
        metric_names = ['acc', 'f1']
        
        binary_metrics = get_metrics(metric_names, binary_metadata)
        multiclass_metrics = get_metrics(metric_names, multiclass_metadata)
        
        # Both should work without errors
        self.assertIn('binary_dataset', binary_metrics)
        self.assertIn('multiclass_dataset', multiclass_metrics)
        
        # Both should have the same structure
        for dataset_name, metrics_dict in [('binary_dataset', binary_metrics), 
                                          ('multiclass_dataset', multiclass_metrics)]:
            self.assertIn('train_acc', metrics_dict[dataset_name])
            self.assertIn('train_f1', metrics_dict[dataset_name])


if __name__ == '__main__':
    unittest.main()