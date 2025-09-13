"""
Unit tests for RUL label validation in multi-task PHM training.

Tests the bug fixes implemented for handling missing/invalid RUL labels
that were causing NaN losses during training.

Author: PHM-Vibench Team  
Date: 2025-09-08
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import torch
import numpy as np
from argparse import Namespace
from unittest.mock import patch, MagicMock
from io import StringIO

from src.task_factory.task.In_distribution.multi_task_phm import task as MultiTaskPHM


class MockNetwork(torch.nn.Module):
    """Mock network for testing."""
    def forward(self, x, file_id=None, task_id=None):
        B, L, C = x.shape
        # Mock outputs for different task heads
        mock_output = MagicMock()
        mock_output.classification_logits = torch.randn(B, 5)
        mock_output.anomaly_detection_logits = torch.randn(B, 1)
        mock_output.signal_prediction_logits = torch.randn(B, L, C)
        mock_output.rul_prediction_logits = torch.randn(B, 1)
        return mock_output


class TestRULValidation(unittest.TestCase):
    """Test cases for RUL label validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.args_data = Namespace()
        self.args_model = Namespace()
        self.args_task = Namespace(
            enabled_tasks=['classification', 'rul_prediction'],
            task_weights={'classification': 1.0, 'rul_prediction': 0.8},
            optimizer='adam',
            lr=0.001
        )
        self.args_trainer = Namespace(gpus=False)
        self.args_environment = Namespace()
        
        # Mock metadata with different RUL scenarios
        self.metadata = {
            1: {'Name': 'good_rul', 'RUL_label': 500.0, 'Label': 2},
            2: {'Name': 'missing_rul', 'Label': 1},  # No RUL_label key
            3: {'Name': 'none_rul', 'RUL_label': None, 'Label': 0},
            4: {'Name': 'nan_rul', 'RUL_label': float('nan'), 'Label': 3},
            5: {'Name': 'zero_rul', 'RUL_label': 0.0, 'Label': 2},
            6: {'Name': 'invalid_type', 'RUL_label': 'invalid', 'Label': 1},
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
    
    def test_valid_rul_label(self):
        """Test that valid RUL labels are used correctly."""
        y = torch.tensor([1])  # classification label
        metadata = {'RUL_label': 500.0}
        
        y_dict = self.task_module._build_task_labels(y, metadata, file_id=1)
        
        self.assertIn('rul_prediction', y_dict)
        rul_tensor = y_dict['rul_prediction']
        self.assertEqual(rul_tensor.shape, torch.Size([1]))
        self.assertEqual(rul_tensor.item(), 500.0)
        self.assertFalse(torch.isnan(rul_tensor))
    
    def test_missing_rul_label(self):
        """Test that missing RUL labels use default value."""
        y = torch.tensor([1])
        metadata = {}  # No RUL_label key
        
        with patch('builtins.print') as mock_print:
            y_dict = self.task_module._build_task_labels(y, metadata, file_id=2)
        
        self.assertIn('rul_prediction', y_dict)
        rul_tensor = y_dict['rul_prediction']
        self.assertEqual(rul_tensor.shape, torch.Size([1]))
        self.assertEqual(rul_tensor.item(), 1000.0)  # Default value
        self.assertFalse(torch.isnan(rul_tensor))
        
        # Check warning was printed
        mock_print.assert_called_once()
        args, _ = mock_print.call_args
        self.assertIn('Missing/invalid RUL label', args[0])
        self.assertIn('default value 1000.0', args[0])
    
    def test_none_rul_label(self):
        """Test that None RUL labels use default value."""
        y = torch.tensor([1])
        metadata = {'RUL_label': None}
        
        with patch('builtins.print') as mock_print:
            y_dict = self.task_module._build_task_labels(y, metadata, file_id=3)
        
        self.assertIn('rul_prediction', y_dict)
        rul_tensor = y_dict['rul_prediction']
        self.assertEqual(rul_tensor.item(), 1000.0)  # Default value
        self.assertFalse(torch.isnan(rul_tensor))
        
        # Check warning was printed
        mock_print.assert_called_once()
    
    def test_nan_rul_label(self):
        """Test that NaN RUL labels use default value."""
        y = torch.tensor([1])
        metadata = {'RUL_label': float('nan')}
        
        with patch('builtins.print') as mock_print:
            y_dict = self.task_module._build_task_labels(y, metadata, file_id=4)
        
        self.assertIn('rul_prediction', y_dict)
        rul_tensor = y_dict['rul_prediction']
        self.assertEqual(rul_tensor.item(), 1000.0)  # Default value
        self.assertFalse(torch.isnan(rul_tensor))
    
    def test_zero_rul_label(self):
        """Test that zero RUL labels are valid."""
        y = torch.tensor([1])
        metadata = {'RUL_label': 0.0}
        
        y_dict = self.task_module._build_task_labels(y, metadata, file_id=5)
        
        self.assertIn('rul_prediction', y_dict)
        rul_tensor = y_dict['rul_prediction']
        self.assertEqual(rul_tensor.item(), 0.0)
        self.assertFalse(torch.isnan(rul_tensor))
    
    def test_invalid_type_rul_label(self):
        """Test that invalid type RUL labels use default value."""
        y = torch.tensor([1])
        metadata = {'RUL_label': 'invalid_string'}
        
        with patch('builtins.print') as mock_print:
            y_dict = self.task_module._build_task_labels(y, metadata, file_id=6)
        
        self.assertIn('rul_prediction', y_dict)
        rul_tensor = y_dict['rul_prediction']
        self.assertEqual(rul_tensor.item(), 1000.0)  # Default value
        self.assertFalse(torch.isnan(rul_tensor))
    
    def test_warning_deduplication(self):
        """Test that warnings are only printed once per file_id."""
        y = torch.tensor([1])
        metadata = {}  # No RUL_label
        
        with patch('builtins.print') as mock_print:
            # First call should print warning
            self.task_module._build_task_labels(y, metadata, file_id=10)
            # Second call with same file_id should not print warning
            self.task_module._build_task_labels(y, metadata, file_id=10)
        
        # Should only be called once
        self.assertEqual(mock_print.call_count, 1)
    
    def test_batch_dimension_handling(self):
        """Test that RUL tensors match batch dimensions."""
        y_batch = torch.tensor([1, 2, 0])  # Batch size 3
        metadata = {'RUL_label': 750.0}
        
        y_dict = self.task_module._build_task_labels(y_batch, metadata, file_id=7)
        
        rul_tensor = y_dict['rul_prediction']
        self.assertEqual(rul_tensor.shape, torch.Size([3]))  # Match batch size
        self.assertTrue(torch.allclose(rul_tensor, torch.tensor([750.0, 750.0, 750.0])))
        self.assertFalse(torch.any(torch.isnan(rul_tensor)))
    
    def test_device_consistency(self):
        """Test that RUL tensors are on the same device as input."""
        if torch.cuda.is_available():
            y = torch.tensor([1]).cuda()
            metadata = {'RUL_label': 300.0}
            
            y_dict = self.task_module._build_task_labels(y, metadata, file_id=8)
            rul_tensor = y_dict['rul_prediction']
            
            self.assertEqual(rul_tensor.device, y.device)
        else:
            # Skip CUDA test if not available
            self.skipTest("CUDA not available")


if __name__ == '__main__':
    unittest.main()