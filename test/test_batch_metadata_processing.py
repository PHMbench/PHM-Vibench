"""
Test for batch metadata processing fix in multi-task PHM.

This test verifies that the _build_task_labels_batch method correctly
processes metadata for each sample in a batch individually, rather than
using only the first sample's metadata for all samples.

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


class MockNetwork(torch.nn.Module):
    """Mock network for testing."""
    def forward(self, x, file_id=None, task_id=None):
        B, L, C = x.shape
        # Mock outputs for different task heads
        mock_output = MagicMock()
        mock_output.classification_logits = torch.randn(B, 4)
        mock_output.anomaly_detection_logits = torch.randn(B, 1)
        mock_output.signal_prediction_logits = torch.randn(B, L, C)
        mock_output.rul_prediction_logits = torch.randn(B, 1)
        return mock_output


class TestBatchMetadataProcessing(unittest.TestCase):
    """Test batch metadata processing functionality."""
    
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
        
        # Create metadata with different RUL values for different file_ids
        self.metadata = {
            1: {'Name': 'dataset1', 'Label': 0, 'RUL_label': 100.0},
            2: {'Name': 'dataset1', 'Label': 1, 'RUL_label': 200.0}, 
            3: {'Name': 'dataset2', 'Label': 2, 'RUL_label': 300.0},
            4: {'Name': 'dataset2', 'Label': 3, 'RUL_label': 400.0},
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
    
    def test_single_sample_batch_compatibility(self):
        """Test that single sample batches work correctly (backward compatibility)."""
        # Single sample batch
        y = torch.tensor([0])
        file_ids = [1]
        
        y_dict = self.task_module._build_task_labels_batch(y, file_ids)
        
        # Should have both tasks
        self.assertIn('classification', y_dict)
        self.assertIn('rul_prediction', y_dict)
        
        # Check RUL value is correct for the single sample
        self.assertEqual(y_dict['rul_prediction'].item(), 100.0)
    
    def test_multi_sample_batch_different_rul(self):
        """Test that multi-sample batch uses correct RUL for each sample."""
        # Multi-sample batch with different file_ids and RUL values
        y = torch.tensor([0, 1, 2, 3])
        file_ids = [1, 2, 3, 4]  # Different file_ids with different RUL values
        
        y_dict = self.task_module._build_task_labels_batch(y, file_ids)
        
        # Should have both tasks
        self.assertIn('classification', y_dict)
        self.assertIn('rul_prediction', y_dict)
        
        # Check that each sample gets its correct RUL value
        expected_rul_values = torch.tensor([100.0, 200.0, 300.0, 400.0])
        torch.testing.assert_close(y_dict['rul_prediction'], expected_rul_values)
        
        print(f"✅ Multi-sample RUL values: {y_dict['rul_prediction'].tolist()}")
    
    def test_batch_with_missing_rul_values(self):
        """Test batch processing with some missing RUL values."""
        # Create metadata with missing RUL for some samples
        metadata_with_missing = {
            1: {'Name': 'dataset1', 'Label': 0, 'RUL_label': 150.0},
            2: {'Name': 'dataset1', 'Label': 1},  # Missing RUL_label
            3: {'Name': 'dataset2', 'Label': 2, 'RUL_label': None},  # None RUL_label
            4: {'Name': 'dataset2', 'Label': 3, 'RUL_label': 450.0},
        }
        
        # Create new task module with missing RUL metadata
        task_module = MultiTaskPHM(
            self.network,
            self.args_data,
            self.args_model,
            self.args_task,
            self.args_trainer,
            self.args_environment,
            metadata_with_missing
        )
        
        y = torch.tensor([0, 1, 2, 3])
        file_ids = [1, 2, 3, 4]
        
        with patch('builtins.print'):  # Suppress warnings for test
            y_dict = task_module._build_task_labels_batch(y, file_ids)
        
        # Check that missing values are replaced with default (1000.0)
        expected_rul_values = torch.tensor([150.0, 1000.0, 1000.0, 450.0])
        torch.testing.assert_close(y_dict['rul_prediction'], expected_rul_values)
        
        print(f"✅ Missing RUL handling: {y_dict['rul_prediction'].tolist()}")
    
    def test_tensor_file_ids_handling(self):
        """Test that tensor file_ids are handled correctly."""
        y = torch.tensor([0, 1, 2])
        file_ids = torch.tensor([1, 2, 3])  # Tensor format
        
        y_dict = self.task_module._build_task_labels_batch(y, file_ids)
        
        # Should convert correctly and use individual RUL values
        expected_rul_values = torch.tensor([100.0, 200.0, 300.0])
        torch.testing.assert_close(y_dict['rul_prediction'], expected_rul_values)
    
    def test_classification_labels_preserved(self):
        """Test that classification labels are preserved correctly."""
        y = torch.tensor([0, 1, 2, 3])
        file_ids = [1, 2, 3, 4]
        
        y_dict = self.task_module._build_task_labels_batch(y, file_ids)
        
        # Classification labels should be unchanged
        torch.testing.assert_close(y_dict['classification'], y)
    
    def test_anomaly_detection_labels(self):
        """Test that anomaly detection labels are computed correctly."""
        # Enable anomaly detection task
        self.args_task.enabled_tasks = ['classification', 'anomaly_detection']
        task_module = MultiTaskPHM(
            self.network,
            self.args_data,
            self.args_model,
            self.args_task,
            self.args_trainer,
            self.args_environment,
            self.metadata
        )
        
        y = torch.tensor([0, 1, 2, 3])  # 0 = normal, others = anomaly
        file_ids = [1, 2, 3, 4]
        
        y_dict = task_module._build_task_labels_batch(y, file_ids)
        
        # Check anomaly detection conversion (0=normal, >0=anomaly)
        expected_anomaly = torch.tensor([0.0, 1.0, 1.0, 1.0])
        torch.testing.assert_close(y_dict['anomaly_detection'], expected_anomaly)
    
    def test_batch_size_mismatch_handling(self):
        """Test handling of batch size and file_id count mismatch."""
        y = torch.tensor([0, 1, 2])  # Batch size 3
        file_ids = [1]  # Only one file_id
        
        # Should expand single file_id to match batch size
        y_dict = self.task_module._build_task_labels_batch(y, file_ids)
        
        # All samples should use the same RUL value (100.0)
        expected_rul_values = torch.tensor([100.0, 100.0, 100.0])
        torch.testing.assert_close(y_dict['rul_prediction'], expected_rul_values)
    
    def test_full_batch_processing_workflow(self):
        """Test the complete batch processing workflow."""
        # Create a realistic batch
        batch = {
            'x': torch.randn(4, 1024, 2),  # [B, L, C]
            'y': torch.tensor([0, 1, 2, 3]),
            'file_id': torch.tensor([1, 2, 3, 4])
        }
        
        # Test the batch metadata processing (core functionality)
        file_ids = batch['file_id'].tolist()
        y_dict = self.task_module._build_task_labels_batch(batch['y'], file_ids)
        
        # Validate batch processing results
        self.assertIn('classification', y_dict)
        self.assertIn('rul_prediction', y_dict)
        
        # Each sample should get its correct RUL value
        expected_rul_values = torch.tensor([100.0, 200.0, 300.0, 400.0])
        torch.testing.assert_close(y_dict['rul_prediction'], expected_rul_values)
        
        # Classification labels should match input
        torch.testing.assert_close(y_dict['classification'], batch['y'])
        
        print(f"✅ Full batch workflow successful, RUL values: {y_dict['rul_prediction'].tolist()}")
    
    def test_legacy_vs_new_comparison(self):
        """Compare old single-metadata vs new per-sample metadata approach."""
        y = torch.tensor([0, 1, 2, 3])
        file_ids = [1, 2, 3, 4]
        
        # Old approach (incorrect) - would use metadata from file_id=1 for all samples
        old_metadata = self.metadata[1]  # Only first sample's metadata
        old_y_dict = self.task_module._build_task_labels(y, old_metadata, 1)
        
        # New approach (correct) - uses individual metadata for each sample
        new_y_dict = self.task_module._build_task_labels_batch(y, file_ids)
        
        # Old approach: All RUL values would be 100.0 (from file_id=1)
        # New approach: RUL values should be [100.0, 200.0, 300.0, 400.0]
        
        # Verify the difference
        old_rul_all_same = torch.all(old_y_dict['rul_prediction'] == 100.0)
        self.assertTrue(old_rul_all_same, "Old method should use same RUL for all samples")
        
        expected_new_rul = torch.tensor([100.0, 200.0, 300.0, 400.0])
        torch.testing.assert_close(new_y_dict['rul_prediction'], expected_new_rul)
        
        print(f"✅ Old method (incorrect): {old_y_dict['rul_prediction'].tolist()}")
        print(f"✅ New method (correct): {new_y_dict['rul_prediction'].tolist()}")


if __name__ == '__main__':
    unittest.main(verbosity=2)